"""Single-call sub-agent revision: the J / H / J+H baselines.

A revision-specialist system prompt (the "sub-agent" prompt) receives the
packaged context in a single user message and outputs the JSON Patch directly.
The three variants differ only in which context components the user message
carries:

  sub_hist       (H)    conversation history only
  sub_hist_json  (J+H)  conversation history + current JSON
  sub_json       (J)    current JSON only

The system prompt and every shared section are byte-identical across the three
cells, so any score difference is attributable to the context composition alone.

Integrity: the single call sees ONLY legitimate inputs — the conversation, the
current JSON, the edit instruction, and master_data; never the gold patch /
expected JSON / value constraints.
"""

from __future__ import annotations

import json
from typing import Any, ClassVar

from methods.common import MethodInputs, RevisionMethod, feedback_with_master_data

SUBAGENT_SYSTEM = """\
You are a revision agent for a JSON artifact that is edited over multiple turns.

You are given context for the artifact — the conversation that built it and/or its current state — and a single requested edit (and, when the edit introduces a new entity, reference data for that entity). Compute the full consequence of the edit and apply it: change the field the edit changes directly, and every other field whose value is FORCED to change because it depends on a field that the edit changes — directly or through a chain of such dependencies.

A field is FORCED to change in any of these ways, and you must follow all of them to the end of the chain:
  - it is computed or derived from a changed field (do the arithmetic; pick the correct branch of any conditional such as a threshold crossing or a clamp to a bound, and write the final value);
  - it NAMES, REFERS TO, ROUTES TO, or is ASSIGNED TO an entity that the edit changes (an owner, source, assignee, target, identifier, or label) — update the referent to the new entity even when no number changes;
  - it is a control, policy, status, or routing field that is GATED by a value which, after the edit, crosses a threshold or moves into a different band, tier, or category — switch it to the branch the new value selects.

Aim for a revision that is both COMPLETE and EXACT:
  - Complete: follow every dependency chain — value, reference, and gating — to its end; do not stop at the first level, and do not stop at numeric fields.
  - Exact: change a field only if its value genuinely becomes DIFFERENT. For every candidate, read its current value and compare it to the value you computed; if they are equal, leave it unchanged. Do not change a field merely because it is topically related.

Before returning, review the operations once: drop any operation whose value equals the current value, and add any forced change — value, reference, or gated — that you missed.

# Output format
Produce a JSON Patch (RFC 6902): a JSON array of operations that transforms the
current artifact into the corrected version. The operations are applied strictly
in order, each to the result of the previous one: when an array element is added
or removed, the indices of the later elements shift accordingly, so every path
must address the artifact state at that point in the sequence. Apply the
requested edit AND change every other field that must change to keep the artifact
internally consistent; do NOT modify unrelated fields. Each operation has the form
{"op": "replace" | "add" | "remove", "path": "<RFC 6901 JSON Pointer>", "value": <new value>}
("value" is omitted for "remove"). Output ONLY the JSON array — no prose, no code fences.\
"""


def build_subagent_user(case: dict[str, Any], *, include_history: bool, include_json: bool,
                        ref_text: str | None = None) -> str:
    parts: list[str] = []
    if include_history:
        convo_txt = "\n".join(f"[{m['role']}] {m['content']}" for m in case.get("conversation", []))
        parts.append(
            "Conversation that built the artifact (shows how each value was derived):\n" + convo_txt)
    if include_json:
        parts.append(
            "Current JSON artifact:\n" + json.dumps(case["current_json"], indent=2, ensure_ascii=False))
    parts.append("Requested edit:\n" + feedback_with_master_data(case))
    if ref_text:
        # Auxiliary reference information is appended AFTER the edit so the
        # context+edit prefix stays byte-identical to the no-ref cells.
        parts.append(ref_text)
    return "\n\n".join(parts)


class SubAgentRevisionBase(RevisionMethod):
    include_history: ClassVar[bool]
    include_json: ClassVar[bool]

    async def prepare(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> MethodInputs:
        del model, temperature, retries, seed  # single call; no internal LLM use
        user = build_subagent_user(
            case, include_history=self.include_history, include_json=self.include_json)
        return MethodInputs(messages=[
            {"role": "system", "content": SUBAGENT_SYSTEM},
            {"role": "user", "content": user},
        ])


class SubHistMethod(SubAgentRevisionBase):
    name = "sub_hist"
    include_history = True
    include_json = False


class SubHistJsonMethod(SubAgentRevisionBase):
    name = "sub_hist_json"
    include_history = True
    include_json = True


class SubJsonMethod(SubAgentRevisionBase):
    name = "sub_json"
    include_history = False
    include_json = True


class SubHistJsonReviewMethod(SubHistJsonMethod):
    """H+J with an opt-in self-review second pass (evaluator-driven). The review
    user message shows only the artifact after applying the model's own first
    patch — the patch itself is already the prior assistant message, and gold is
    never shown."""
    name = "sub_hist_json_rev"
    self_review = True


class SubHistJsonReviewRemindMethod(SubHistJsonReviewMethod):
    """Self-review plus deterministic value-free cross-check reminders
    (mirrored blocks; unmodified mates of repeated edit patterns) inserted into
    the review message only when a signal fires (methods/review_reminders.py)."""
    name = "sub_hist_json_rev_rem"
    self_review_reminders = True


class SubHistJsonReviewDeltaMethod(SubHistJsonReviewMethod):
    """Self-review whose output is ONLY the corrective operations, appended
    after the first patch (sequential semantics against the applied artifact);
    an empty array confirms the first patch. Correct operations are never
    re-emitted, so a review cannot corrupt them."""
    name = "sub_hist_json_rev_d"
    self_review_delta = True


class SubHistJsonReviewRemindDeltaMethod(SubHistJsonReviewDeltaMethod):
    """Delta-output self-review plus the deterministic cross-check reminders."""
    name = "sub_hist_json_rev_rem_d"
    self_review_reminders = True


class SubHistJsonReview3Method(SubHistJsonReviewDeltaMethod):
    """Cost-scaling arm A: THREE serial delta-review rounds. Each round sees
    the conversation so far plus the artifact after the cumulative patch and
    appends corrective ops. The evaluator records the patch after every round
    (review_stages), so 1-/2-round performance is derived from the same run."""
    name = "sub_hist_json_rev3"
    self_review_rounds = 3


class SubHistJsonReview4Method(SubHistJsonReviewDeltaMethod):
    """Serial arm extended to FOUR review rounds (5 calls total). Rounds 1-3
    replay the rev3 conversation from cache; only round 4 is new."""
    name = "sub_hist_json_rev4"
    self_review_rounds = 4


class SubHistJsonReview8Method(SubHistJsonReviewDeltaMethod):
    """Serial arm at EIGHT review rounds (9 calls; rounds 1-6 replay rev6 from
    cache) for the 122B equal-cost study."""
    name = "sub_hist_json_rev8"
    self_review_rounds = 8


class SubHistJsonReview6Method(SubHistJsonReviewDeltaMethod):
    """Serial arm at SIX review rounds (7 calls; rounds 1-4 replay rev4 from
    cache) — matches the displayed serial ladder end (x6)."""
    name = "sub_hist_json_rev6"
    self_review_rounds = 6


class SubHistJsonReview7Method(SubHistJsonReviewDeltaMethod):
    """Serial arm extended to SEVEN review rounds (8 calls total) for the
    equal-cost comparison against the judge. Rounds 1-4 replay the rev4
    conversation from cache; rounds 5-7 are new."""
    name = "sub_hist_json_rev7"
    self_review_rounds = 7
