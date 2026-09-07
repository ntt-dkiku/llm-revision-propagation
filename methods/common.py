"""Shared prompts and message builders for evaluation methods."""

from __future__ import annotations

import json
from abc import ABC
from dataclasses import dataclass
from typing import Any, ClassVar


SYSTEM_PROMPT = """\
You are an AI assistant for generating JSON objects over multiple turns and revising them according to user instructions.
When applying changes to generated JSON objects according to user feedback, modify only the relevant JSON elements, ensuring that the changes are neither excessive nor insufficient.
"""

USER_FEEDBACK_TEMPLATE = """\
# User feedback
{feedback}

Based on the above user feedback, please apply the necessary revisions to the JSON objects generated so far, ensuring that only the dependent elements are modified and that irrelevant elements remain unchanged.

{reference_information}

# Output format
{format_instructions}
"""

EXPLICIT_JSON_REF = """\
# Review of the generated JSON
The following is your generated JSON outputs:
{json_outputs}
"""

# The output-format slot: a JSON Patch (the deterministic gold form). This is
# the only part not inherited from v2 (v2 emitted a full revised object); the
# net-effect leaf scoring is agnostic to the patch's surface form.
PATCH_FORMAT = """\
Produce a JSON Patch (RFC 6902): a JSON array of operations that transforms the
current artifact into the corrected version. The operations are applied strictly
in order, each to the result of the previous one: when an array element is added
or removed, the indices of the later elements shift accordingly, so every path
must address the artifact state at that point in the sequence. Apply the
requested edit AND change every other field that must change to keep the artifact
internally consistent; do NOT modify unrelated fields. Each operation has the form
{"op": "replace" | "add" | "remove", "path": "<RFC 6901 JSON Pointer>", "value": <new value>}
("value" is omitted for "remove"). Output ONLY the JSON array — no prose, no code fences."""


# Self-review second pass: the model's first patch is already its previous
# assistant message, so only the new information (the applied artifact) is
# shown; the prompt points back at that message. Gold is never shown.
# Split into check + output parts so an optional reminder section can be
# inserted between them; the plain template is their exact concatenation.
SELF_REVIEW_CHECK = """\
Here is the artifact after applying the JSON Patch you just produced:

# Artifact after applying your patch
{applied}

Review your patch (your previous message) against the user feedback and the artifact. Check, in order:
1. Every field the edit forces to change (the edited field and every field that
   depends on it, directly or through a chain) is updated, and none is missed.
2. No field that should stay unchanged was modified.
3. Each new value is correct (recompute the arithmetic and re-check any
   threshold or conditional branch).
4. Every path is a valid RFC 6901 JSON Pointer addressing a field that exists in
   the artifact (so the patch applies cleanly), with the correct op
   (replace for an existing field, add for a new one, remove to delete)."""

SELF_REVIEW_OUTPUT = """\
Then output the corrected JSON Patch — the COMPLETE array that should be applied
to the ORIGINAL artifact (not a diff against your previous patch). If your patch
was already correct, output it unchanged. Output ONLY the JSON array — no prose,
no code fences."""

SELF_REVIEW_TEMPLATE = SELF_REVIEW_CHECK + "\n\n" + SELF_REVIEW_OUTPUT

# Delta-output variant: the review emits ONLY corrective operations, applied
# after the first patch, so operations that were already correct are never
# re-emitted (and cannot be corrupted by a rewrite). An empty array confirms
# the first patch. The check part replaces the blanket "recompute the
# arithmetic" with an asymmetric burden of proof: the patch is correct by
# default, and a value is corrected only on a named, definite mistake — a
# re-derivation that merely differs is explicitly not evidence (an unstable
# recomputation must not overwrite a correct value).
SELF_REVIEW_CHECK_DELTA = """\
Here is the artifact after applying the JSON Patch you just produced:

# Artifact after applying your patch
{applied}

Review this artifact against the user feedback. Treat your patch as correct by
default; you are looking only for definite errors, of these kinds:
1. a field the edit forces to change (directly or through a dependency chain)
   that still shows its old value;
2. a field that was modified although nothing forces it to change (revert it);
3. a value that contradicts how the conversation and the artifact determine it
   — correct a value only if you can name the specific mistake in it; a
   re-derivation that merely differs from the current value is not evidence of
   a mistake, so in that case keep the current value;
4. an operation from your patch whose change is absent above because its path
   did not address an existing field (re-apply it with a valid path)."""

SELF_REVIEW_OUTPUT_DELTA = """\
Then output a corrective JSON Patch containing ONLY the operations for the
definite errors you found, applied IN ORDER to the artifact shown above (the
result of your previous patch), so every path must address that artifact. If
you found no definite error, output an empty array: []. Output ONLY the JSON
array — no prose, no code fences."""


@dataclass
class MethodInputs:
    messages: list[dict[str, str]]
    estimator_trace: dict[str, Any] | None = None


class RevisionMethod(ABC):
    """Base class for benchmark revision methods."""

    name: ClassVar[str]

    async def prepare(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> MethodInputs:
        reference_information, estimator_trace = await self.build_reference_information(
            case,
            model=model,
            temperature=temperature,
            retries=retries,
            seed=seed,
        )
        return MethodInputs(
            messages=build_revision_messages(case, reference_information=reference_information),
            estimator_trace=estimator_trace,
        )

    async def build_reference_information(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> tuple[str, dict[str, Any] | None]:
        del case, model, temperature, retries, seed
        return "", None


def json_outputs_ref(case: dict[str, Any]) -> str:
    return json.dumps(case["current_json"], ensure_ascii=False, indent=2)


def feedback_with_master_data(case: dict[str, Any]) -> str:
    # When the edit introduces a new entity whose attributes cannot be derived
    # from the conversation, the sample carries a `master_data` lookup record.
    # It MUST be shown with the feedback (all methods) so the dependent values
    # are uniquely determined rather than invented. Without it, the model has no
    # way to know the new entity's attributes and can only guess.
    feedback = case["edit_instruction"]
    md = case.get("master_data")
    if md:
        feedback += (
            "\n\nReference data for the entity introduced by this edit "
            "(use these exact attributes for it):\n"
            + json.dumps(md, ensure_ascii=False, indent=2)
        )
    return feedback


def render_revision_prompt(*, feedback: str, reference_information: str) -> str:
    return USER_FEEDBACK_TEMPLATE.format(
        feedback=feedback,
        reference_information=reference_information,
        format_instructions=PATCH_FORMAT,
    )


def build_revision_messages(case: dict[str, Any], *, reference_information: str) -> list[dict[str, str]]:
    messages: list[dict[str, str]] = [{"role": "system", "content": SYSTEM_PROMPT}]
    for m in case.get("conversation", []):
        messages.append({"role": m["role"], "content": m["content"]})
    messages.append({
        "role": "user",
        "content": render_revision_prompt(
            feedback=feedback_with_master_data(case),
            reference_information=reference_information,
        ),
    })
    return messages
