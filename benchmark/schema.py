"""Pydantic schema for the human-annotated cascade-revision benchmark.

A Case carries the LLM-sampled (unannotated) fields AND the human-annotated
gold. Sampling fills the former; annotation fills the latter.
"""

from __future__ import annotations

import json
from typing import Any, Literal

from pydantic import BaseModel, Field, computed_field

from benchmark.jsonptr import apply_patch, deep_merge


Role = Literal["user", "assistant"]


class Message(BaseModel):
    role: Role
    content: str

class Sample(BaseModel):
    """LLM-sampled, unannotated case. Output of sampling.py.

    Each assistant turn in `conversation` carries a JSON Patch of 'add' ops that
    insert that turn's new content. `current_json` reconstructs the full artifact
    by applying every turn's add-patch in order. (Legacy samples whose assistant
    turns are plain JSON fragments are still supported via deep_merge.) It is a
    computed field so model_dump/model_dump_json include it, but it is never
    stored redundantly — the conversation is the single source of truth.
    """

    id: str
    domain: str
    # the leaf-count target the sampler aimed for; the actual leaf count may
    # overshoot it (the final turn can add a multi-leaf value).
    target_leaves: int | None = None
    # litellm model string used to sample this case (e.g. "openai/gpt-5.5",
    # "local-llm"); preserved end-to-end so downstream artifacts can attribute
    # provenance. None for older samples that predate the field.
    generator: str | None = None
    conversation: list[Message]
    edit_instruction: str
    # the cascade skeleton this sample instantiates (Skeleton.theme); None for
    # samples that predate skeleton-driven sampling.
    theme: str | None = None
    # the skeleton's patterns, stored directly on the sample so the annotator
    # GUI (and any reader) can show them by reading the sample JSON, with no
    # separate skeletons.json catalog. None for samples that predate this.
    conversation_pattern: str | None = None
    edit_pattern: str | None = None
    cascade_pattern: str | None = None
    requires_master_data: bool | None = None
    # lookup record (e.g. a product catalog) generated alongside the edit and
    # rendered into the edit instruction at evaluation time, so substitution-
    # style cascades are deterministic. Only set for skeletons that declare
    # requires_master_data=True; None otherwise.
    master_data: dict[str, Any] | None = None

    @computed_field  # type: ignore[prop-decorator]
    @property
    def current_json(self) -> dict[str, Any]:
        merged: dict[str, Any] = {}
        saw_assistant = False
        for m in self.conversation:
            if m.role != "assistant":
                continue
            saw_assistant = True
            try:
                obj = json.loads(m.content)
            except json.JSONDecodeError as e:
                raise ValueError(
                    f"assistant message in case {self.id!r} is not valid JSON: {e}"
                ) from e
            if isinstance(obj, list):
                # add-patch turn: a JSON Patch of 'add' ops applied in order
                merged = apply_patch(merged, obj)
            elif isinstance(obj, dict):
                # legacy fragment turn: merged via deep_merge (back-compat)
                deep_merge(merged, obj)
            else:
                raise ValueError(
                    f"assistant message in case {self.id!r} is not a JSON object or add patch"
                )
        if not saw_assistant:
            raise ValueError(f"case {self.id!r} has no assistant message in conversation")
        return merged


#------------ 
# Annotation
#------------
class PatchOp(BaseModel):
    """RFC 6902 JSON Patch operation (subset we support).

    `kind` tags whether this op is the single DIRECT edit the user named in the
    edit_instruction, or a CASCADE op required to keep the artifact internally
    consistent. The evaluator uses it to split direct- vs cascade-recall.
    """

    op: Literal["replace", "add", "remove"]
    path: str | None = None  # single-leaf ops use this. Multi-leaf constraint
                              # ops (matcher='sum_with') use `paths` instead.
    value: Any = None
    kind: Literal["direct", "cascade"] = "cascade"
    # Acceptance criterion on the resulting leaf value. When `matcher` is None
    # (legacy default), only the exact `value` is accepted (same as `strict`).
    # Each matcher uses its own subset of the parameter fields below:
    #   - `strict`      : the resulting value must equal `value` exactly. No
    #                     extra params; equivalent to leaving matcher unset.
    #   - `date_range`  : `after` (exclusive lower) and/or `before` (inclusive
    #                     upper), both ISO dates.
    #   - `regex_match` : `pattern` (Python regex, re.search semantics).
    #   - `any`         : any resulting value is accepted (no params). Useful
    #                     to mark a leaf where the cascade has no canonical
    #                     answer and only the location matters.
    #   - `sum_with`    : multi-path constraint. Uses `paths` (a list of JSON
    #                     pointers, NOT the singular `path` field) and
    #                     `expected_sum`. The sum of the resulting values at
    #                     all `paths` (in the same model output) must equal
    #                     `expected_sum`. Use for cross-field invariants where
    #                     individual splits are ambiguous but the total is
    #                     determined (e.g., transit + attractions = budget).
    #                     The `paths` are evaluated as a SINGLE pair/group
    #                     unit (1 entry in value_recall / loc_f1 denom),
    #                     touched = any path changed, satisfied = sum holds.
    #                     `optional` is irrelevant for sum_with (constraint
    #                     semantics dominate; both-untouched is a miss iff
    #                     the sum is also wrong).
    matcher: Literal["strict", "date_range", "regex_match", "any", "sum_with"] | None = None
    after: str | None = None
    before: str | None = None
    pattern: str | None = None
    paths: list[str] | None = None
    expected_sum: float | None = None
    # If True, the leaf is NOT required to change: the LLM may leave it as it
    # was, OR change it to any value satisfying `matcher` (or to `value`).
    # Out-of-spec changes still count as over_edit. If both `matcher` is None
    # and `optional=True`, the leaf is fully excluded from scoring.
    optional: bool = False


class ValueConstraint(BaseModel):
    """A flexible (non-point) acceptance criterion for one leaf.

    Some cascade targets are not uniquely determined: e.g. a milestone with no
    explicit anchor only has to land on/before the deadline (and after the
    previous milestone), so ANY date in that window is correct. Or a
    natural-language cascade leaf may be phrased many ways ('sessions 1, 3,
    and 5' vs 'sessions 1, 3, 5'), where the meaning matches but the surface
    string differs. A point-valued gold would unfairly reject these valid
    alternatives, so such leaves carry a constraint and the scorer accepts
    any satisfying value.

    `matcher` selects the matching strategy. Each strategy uses its own subset
    of fields:
      - `date_range`  : `after` (exclusive lower bound), `before` (inclusive
                        upper bound), both ISO dates. Value must parse as ISO
                        date and lie in the bounds.
      - `regex_match` : `pattern` (a Python regex). Value must be a string and
                        `re.search(pattern, value)` must match.

    `optional=True` means the leaf may also be left unchanged (its original
    value already satisfies the constraint); `optional=False` means a change
    into the accepted set is required.
    """

    path: str
    matcher: Literal["date_range", "regex_match"] = "date_range"
    # date_range fields
    after: str | None = None       # exclusive lower bound (ISO date), or None
    before: str | None = None      # inclusive upper bound (ISO date), or None
    # regex_match fields
    pattern: str | None = None     # Python regex (re.search semantics)
    optional: bool = True


class AnnotatedSample(Sample):
    """Sample + human-annotated gold. Output of annotation.py.

    The gold is the expected JSON Patch (direct edit + cascade ops) and the
    resulting expected_json. We deliberately do NOT annotate a dependency
    graph: edge existence/direction/type is interpretation-laden, and
    graph-similarity is neither necessary nor sufficient for output
    correctness — the patch is the deterministic, falsifiable gold.

    `value_constraints` optionally relax specific leaves from point-equality to
    a range (see ValueConstraint); the scorer treats an in-range value as
    correct and does not count an in-range change as an over-edit.
    """

    expected_patch: list[PatchOp] = Field(default_factory=list)
    expected_json: dict[str, Any] | None = None
    value_constraints: list[ValueConstraint] = Field(default_factory=list)
    annotator: str = ""
    annotation_notes: str = ""
    annotated: bool = False
