"""Evaluate cascade-aware revision methods on annotated benchmark cases.

The registered methods (see methods/registry.py): the single-call baselines
J / H / J+H (sub_json / sub_hist / sub_hist_json, differing only in the context
shown to the model), serial reflection (sub_hist_json_rev*), and the
parallel-sampling methods (sub_hist_json_cons*_and for OR / AND / MAJ / MED,
sub_hist_json_judge* for SELECT).

For each case the model returns a JSON Patch; we apply the model patch and the
gold patch to current_json and compare their NET effect at leaf-path
granularity (robust to how the patch is expressed). Metrics (per case, then
averaged):

  - loc_precision / loc_recall / loc_f1 : changed-leaf-set agreement (location)
  - over_edit  : # leaves the model changed that gold did not
  - miss       : # leaves gold changed that the model did not
  - value_recall : fraction of gold-changed leaves set to the correct value
  - direct_recall / cascade_recall : value_recall split by the gold op `kind`

Usage:
  python -m benchmark.evaluator --annotated benchmark/data/annotated_test \
      --out results/bench/<run> --model <name> [--methods sub_hist_json sub_json]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import math
import re
import statistics
from pathlib import Path
from typing import Any

from agents.llm import complete
from benchmark.jsonptr import apply_patch, list_leaf_paths, pointer_get
from benchmark.strict_score import score_strict
from methods import METHOD_NAMES, create_method
from methods.common import (
    SELF_REVIEW_CHECK, SELF_REVIEW_CHECK_DELTA,
    SELF_REVIEW_OUTPUT, SELF_REVIEW_OUTPUT_DELTA,
)
from methods.review_reminders import build_review_reminders

_MISSING = object()

METHODS = METHOD_NAMES


# ----------------------------- leaf diffing -------------------------------


def leaf_map(obj: Any) -> dict[str, Any]:
    return {p: pointer_get(obj, p) for p in list_leaf_paths(obj)}


def changed_leaves(before: Any, after: Any) -> set[str]:
    bm, am = leaf_map(before), leaf_map(after)
    return {p for p in (set(bm) | set(am)) if bm.get(p, _MISSING) != am.get(p, _MISSING)}


# --------------------------- patch extraction -----------------------------


def _extract_json_array(text: str) -> list[Any]:
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```\s*$", "", text)
    start = text.find("[")
    if start < 0:
        raise ValueError(f"no JSON array found in: {text[:200]!r}")
    depth = 0
    end = -1
    in_str = False
    esc = False
    for i in range(start, len(text)):
        ch = text[i]
        if esc:
            esc = False
            continue
        if ch == "\\":
            esc = True
            continue
        if ch == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if end < 0:
        raise ValueError("unbalanced JSON brackets")
    arr = json.loads(text[start:end])
    if not isinstance(arr, list):
        raise ValueError("top-level JSON is not an array")
    return arr


# ------------------------------- scoring ----------------------------------


def _value_recall(changed: set[str], gold_after: dict, model_after: dict) -> float:
    if not changed:
        return 1.0
    ok = sum(1 for p in changed if model_after.get(p, _MISSING) == gold_after.get(p, _MISSING))
    return ok / len(changed)


_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _as_date(v: Any):
    if isinstance(v, str) and _DATE_RE.match(v):
        try:
            return dt.date.fromisoformat(v)
        except ValueError:
            return None
    return None


def _in_range(value: Any, vc: dict) -> bool:
    """True if `value` satisfies a date_range ValueConstraint (after < v <= before)."""
    d = _as_date(value)
    if d is None:
        return False
    after = _as_date(vc.get("after"))
    before = _as_date(vc.get("before"))
    if after is not None and not (d > after):
        return False
    if before is not None and not (d <= before):
        return False
    return True


def _matches_regex(value: Any, vc: dict) -> bool:
    """True if `value` is a string satisfying the regex_match ValueConstraint
    (re.search semantics)."""
    if not isinstance(value, str):
        return False
    pat = vc.get("pattern")
    if not pat:
        return False
    try:
        return re.search(pat, value) is not None
    except re.error:  # malformed pattern -> never matches (annotator bug)
        return False


def _satisfies_constraint(value: Any, vc: dict) -> bool:
    """Dispatch on `matcher` field. Accepts the legacy `type` key too so older
    annotated files that predate the rename still load. Unknown matchers
    silently fail (never satisfy) to keep scoring conservative.

    Only single-leaf matchers are handled here. The multi-path `sum_with`
    constraint is handled by `multi_path_constraint_satisfied` and processed
    separately in `compute()`.
    """
    matcher = vc.get("matcher") or vc.get("type") or "strict"
    if matcher == "strict":
        # The canonical value is `vc["value"]` on inline patch ops; legacy
        # standalone constraints did not carry a value, so default to equality
        # check against the gold leaf value (handled by the caller).
        return value == vc.get("value")
    if matcher == "any":
        return True
    if matcher == "date_range":
        return _in_range(value, vc)
    if matcher == "regex_match":
        return _matches_regex(value, vc)
    return False


def multi_path_constraint_satisfied(op: dict, model_after: Any) -> bool:
    """Check if a multi-path constraint op (e.g. `sum_with`) is satisfied by
    the model output. The op carries `paths` (list of JSON pointers) and a
    matcher-specific parameter (`expected_sum` for `sum_with`)."""
    matcher = op.get("matcher")
    paths = op.get("paths") or []
    if not paths:
        return False
    sentinel = object()
    if matcher == "sum_with":
        expected_sum = op.get("expected_sum")
        if expected_sum is None:
            return False
        try:
            total = 0.0
            for p in paths:
                val = pointer_get(model_after, p, default=sentinel)
                if val is sentinel:
                    return False
                total += float(val)
            return math.isclose(total, float(expected_sum), abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return False


def op_satisfied(op: dict, current: Any, model_after: Any) -> bool:
    """Check if one alternative `op` is consistent with the transition from
    `current` to `model_after`.

    Two scoring modes:

    (A) FLEX matchers (`date_range`, `regex_match`, `any`): the matcher
        describes a set of acceptable values; we check that against the
        model's leaf directly.

    (B) STRICT or REMOVE: there is a SINGLE deterministic outcome. We compute
        it by applying `op` to `current` and compare the resulting leaf state
        at `path` against `model_after`'s state at the same path. This
        correctly distinguishes:
          - object-key remove (path becomes missing) vs replace (path holds a
            different value): the partial_after state differs, so a
            replacement by the LLM does NOT satisfy a remove annotation.
          - array-index remove (array shrinks, later elements shift down):
            partial_after has the shifted state; the LLM doing an equivalent
            array shrink yields the same shifted state — both satisfy.
          - array-index remove vs same-index replace (the LLM keeps the
            array length and only changes the value at that index):
            partial_after's array is shorter, so the lengths/states differ
            and the replacement does NOT satisfy a remove annotation.
    """
    path = op["path"]
    matcher = op.get("matcher") or "strict"
    sentinel = object()

    # (A) flex matchers — check the matcher against model_after's value.
    if matcher == "any":
        return True
    if matcher in ("date_range", "regex_match"):
        val = pointer_get(model_after, path, default=sentinel)
        if val is sentinel:
            return False
        return _satisfies_constraint(val, op)

    # (B) strict / add / remove — judge by the FINAL state in `model_after`
    # against the op's target, NOT by re-applying the op to `current`. The
    # latter breaks for array ops: index removal/insertion shifts siblings, the
    # gold op may reference an index that only exists post-insert, and other
    # cascades touching the same array confound a parent-array comparison. The
    # final-state view is robust to all of these and to how the model expressed
    # the change (per-index, whole-array, replace-with-shorter, etc.).
    op_kind = op.get("op", "replace")
    segs = path.split("/")
    last = segs[-1] if len(segs) > 1 else ""
    parent_path = "/".join(segs[:-1])

    if op_kind == "remove":
        if last.isdigit():
            # array element: the element that was at `path` in `current` must no
            # longer appear in model_after's parent array (or the array is gone).
            removed_elem = pointer_get(current, path, default=sentinel)
            if removed_elem is sentinel:
                return False
            model_parent = pointer_get(model_after, parent_path, default=sentinel)
            if model_parent is sentinel:
                return True
            return isinstance(model_parent, list) and removed_elem not in model_parent
        # object key: the key must be absent in model_after.
        return pointer_get(model_after, path, default=sentinel) is sentinel

    if op_kind == "add" and (last.isdigit() or last == "-"):
        # array insert/append: the added value must be present in the array.
        model_parent = pointer_get(model_after, parent_path, default=sentinel)
        return isinstance(model_parent, list) and op.get("value") in model_parent

    # replace, or add of an object key: the model's value at `path` must equal
    # the op's target value.
    return pointer_get(model_after, path, default=sentinel) == op.get("value")


def _path_state_differs(path: str, current: Any, model_after: Any) -> bool:
    """True if the state at `path` differs between `current` and `model_after`,
    including the case of being added or removed."""
    sentinel = object()
    cur = pointer_get(current, path, default=sentinel)
    new = pointer_get(model_after, path, default=sentinel)
    return cur != new


def _op_touched(path: str, alts: list[dict], current: Any, model_after: Any) -> bool:
    """Did the model touch the location this gold op targets?

    For array-index add/remove ops the path's last segment is a numeric index
    or the append token "-", which does not resolve to a gettable leaf (and
    "-" is not even a valid index for a read). Compare at the PARENT array
    level instead — mirroring op_satisfied — so an append at "/arr/-" registers
    as touched exactly like an append at "/arr/<N>". Other ops compare at the
    op's own path."""
    segs = path.split("/")
    last = segs[-1] if len(segs) > 1 else ""
    if (last.isdigit() or last == "-") and any(
        a.get("op") in ("add", "remove") for a in alts
    ):
        return _path_state_differs("/".join(segs[:-1]), current, model_after)
    return _path_state_differs(path, current, model_after)


def _attach_legacy_constraints(
    gold_patch: list[dict], value_constraints: list[dict],
) -> list[dict]:
    """Merge a legacy `value_constraints` list (separate from expected_patch)
    onto the corresponding ops in `gold_patch`. If a constraint references a
    path not in the patch, a synthetic op is appended carrying only the
    constraint (no canonical value), so the scoring loop sees it.

    No-op when value_constraints is empty (the common case for newly-annotated
    samples that use the inline-matcher schema).
    """
    if not value_constraints:
        return gold_patch
    by_path: dict[str, dict] = {op.get("path"): op for op in gold_patch}
    out = list(gold_patch)  # shallow copy; ops will be replaced as needed
    for vc in value_constraints:
        p = vc.get("path")
        if not p:
            continue
        # Translate legacy `type` -> `matcher`.
        legacy_matcher = vc.get("matcher") or vc.get("type") or "date_range"
        spec = {
            "matcher": legacy_matcher,
            "after": vc.get("after"),
            "before": vc.get("before"),
            "pattern": vc.get("pattern"),
            "optional": bool(vc.get("optional", False)),
        }
        if p in by_path:
            # Merge into the matching op without overwriting the canonical value/op/kind.
            merged = {**by_path[p], **{k: v for k, v in spec.items() if v is not None or k == "optional"}}
            idx = out.index(by_path[p])
            out[idx] = merged
        else:
            # Synthesize a placeholder op so the scorer can see the constraint.
            out.append({
                "op": "replace", "path": p, "value": None, "kind": "cascade",
                **spec,
            })
    return out


def score(current: Any, gold_patch: list[dict], model_patch: list[dict],
          value_constraints: list[dict] | None = None) -> dict[str, Any]:
    """Net-effect path-level scoring with OR-alternative support.

    Acceptance criteria live INLINE on each gold patch op (the `matcher`,
    `after`, `before`, `pattern`, `optional` fields on PatchOp). When the
    `expected_patch` contains MULTIPLE ops at the SAME path, those ops form an
    OR alternative group — the LLM satisfies the expected cascade at that path
    if its output is consistent with AT LEAST ONE of the alternatives.

    The legacy `value_constraints` parameter is still accepted (older
    annotations that predate the inline-matcher schema) — its entries are
    folded onto matching paths in `gold_patch` before scoring.

    Per-path semantics (paths in expected_patch only):
      - Required path (any alt has optional=False): the LLM's behavior must
        satisfy at least one alt. Untouched -> miss. Touched-but-no-alt-
        satisfied -> over_edit.
      - Optional path (all alts have optional=True): the LLM may leave the
        leaf as it was. If touched: a satisfying alt is free, an out-of-spec
        change drops value_recall only (never over_edit).
    """
    # Fold any legacy `value_constraints` onto the corresponding gold op (or
    # synthesize a placeholder op so we can still inspect them uniformly).
    gold_patch = _attach_legacy_constraints(gold_patch, value_constraints or [])

    # Split gold ops into single-path ops (with `path`) and multi-path
    # constraint ops (with `paths`, e.g. matcher='sum_with'). The latter are
    # not "patches" in the RFC 6902 sense — they describe a cross-leaf
    # invariant — so they are not applied via apply_patch and are scored as
    # a single pair/group unit instead.
    single_path_ops: list[dict] = []
    constraint_ops: list[dict] = []
    for op in gold_patch:
        if op.get("paths"):
            constraint_ops.append(op)
        else:
            single_path_ops.append(op)

    # Group single-path expected ops by path; same-path ops form OR alts.
    ops_by_path: dict[str, list[dict]] = {}
    for op in single_path_ops:
        ops_by_path.setdefault(op["path"], []).append(op)

    # A path is "optional" iff ALL its alternatives are optional.
    path_optional: dict[str, bool] = {
        p: all(o.get("optional") for o in alts)
        for p, alts in ops_by_path.items()
    }

    # Compute the canonical gold artifact from the FIRST non-optional alt per
    # path; optional paths contribute no canonical change. Constraint ops are
    # NOT applied (they have no transform semantics, only validity checks).
    canonical_ops = [
        alts[0] for p, alts in ops_by_path.items() if not path_optional[p]
    ]
    gold_after = apply_patch(current, canonical_ops)
    try:
        model_after = apply_patch(current, model_patch)
        apply_ok = True
    except Exception:  # noqa: BLE001 — a malformed patch scores as "no change"
        model_after = current
        apply_ok = False

    gold_changed = changed_leaves(current, gold_after)
    model_changed = changed_leaves(current, model_after)

    # Iterate paths with OR satisfaction.
    tp = fp = fn = 0
    val_ok = 0
    val_total = 0          # required-path count contributing to value_recall denom
    flex_opt_touched = 0   # optional paths the LLM touched (regardless of satisfaction)
    flex_opt_ok = 0        # touched optional paths whose behavior satisfied an alt
    direct_outcomes: list[bool] = []      # per direct-path: did any alt satisfy?
    cascade_outcomes: list[bool] = []     # per required cascade-path: did any alt satisfy?

    for path, alts in ops_by_path.items():
        is_optional = path_optional[path]
        any_satisfied = any(op_satisfied(alt, current, model_after) for alt in alts)
        touched = _op_touched(path, alts, current, model_after)
        any_direct = any(o.get("kind") == "direct" for o in alts)

        if is_optional:
            # Optional: only contributes to flex_opt_* counters. A satisfying
            # touch boosts value_recall, an out-of-spec touch drops it, and an
            # untouched optional path has no effect.
            if touched:
                flex_opt_touched += 1
                if any_satisfied:
                    flex_opt_ok += 1
        else:
            # Required path: location and value are tracked INDEPENDENTLY.
            #   - touched (regardless of value correctness): tp (location ok)
            #   - untouched: fn (location missed)
            #   - touched but no alt satisfied: val_recall drops; NOT over_edit
            #     (over_edit is reserved for touches on UNRELATED paths)
            val_total += 1
            if touched:
                tp += 1
                if any_satisfied:
                    val_ok += 1
            else:
                fn += 1
            (direct_outcomes if any_direct else cascade_outcomes).append(any_satisfied)

    # Constraint ops (multi-path sum_with etc.): processed as ONE unit per op.
    # touched = any path changed in model_after; satisfied = constraint holds
    # (e.g. sum of values equals expected_sum). `optional` is irrelevant —
    # the constraint semantics dominate: both-untouched + constraint violated
    # is a miss; touched-or-untouched + constraint satisfied is a pass.
    constraint_paths: set[str] = set()
    for cop in constraint_ops:
        c_paths = cop.get("paths") or []
        constraint_paths.update(c_paths)
        any_touched = any(
            _path_state_differs(p, current, model_after) for p in c_paths
        )
        c_satisfied = multi_path_constraint_satisfied(cop, model_after)
        is_direct = cop.get("kind") == "direct"
        val_total += 1
        if any_touched:
            tp += 1
            if c_satisfied:
                val_ok += 1
        else:
            if c_satisfied:
                # Untouched but constraint already holds (e.g. partner absorbed
                # the change). Credit value_recall but NOT a location tp; no
                # miss either, since the invariant is satisfied as-is.
                val_ok += 1
            else:
                fn += 1
        (direct_outcomes if is_direct else cascade_outcomes).append(c_satisfied)

    # Extra over_edit: LEAF in model_changed that is NEITHER covered by a
    # gold path NOR an implied byproduct of a gold op (i.e., already in
    # gold_changed via apply_patch). Array remove ops, for example, shift
    # later indices into earlier slots and make the old last index missing —
    # those byproducts appear in gold_changed but not in ops_by_path keys.
    # Constraint op paths are also "covered" — touching them is intended.
    model_extra = model_changed - set(ops_by_path) - gold_changed - constraint_paths
    fp += len(model_extra)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    n_value = val_total + flex_opt_touched
    value_recall = (val_ok + flex_opt_ok) / n_value if n_value else 1.0

    direct_recall = (
        sum(1 for ok in direct_outcomes if ok) / len(direct_outcomes)
        if direct_outcomes else 1.0
    )
    cascade_recall = (
        sum(1 for ok in cascade_outcomes if ok) / len(cascade_outcomes)
        if cascade_outcomes else 1.0
    )

    n_flex = sum(1 for alts in ops_by_path.values() if any(o.get("matcher") for o in alts))
    return {
        "apply_ok": apply_ok,
        "loc_precision": precision,
        "loc_recall": recall,
        "loc_f1": f1,
        "over_edit": fp,
        "miss": fn,
        "value_recall": value_recall,
        "direct_recall": direct_recall,
        "cascade_recall": cascade_recall,
        "n_gold_changed": len(gold_changed),
        "n_model_changed": len(model_changed),
        "n_flex": n_flex,
    }


# --------------------------- per-case eval --------------------------------

# Generous budget: reasoning models (esp. Qwen3 with thinking) can spend many
# tokens deliberating even on small cases; too small a cap truncates before the
# patch is emitted (finish_reason=length), which scores as a parse failure.
REV_MAX_TOKENS = 32000


async def eval_one(
    case: dict[str, Any], *, method: str, model: str | None, temperature: float, max_retries: int,
    seed: int | None = None,
) -> dict[str, Any]:
    """Evaluate one case under one method.

    The returned dict captures EVERYTHING needed for post-hoc analysis: the
    fully-rendered user prompt that was sent, the model's raw answer and
    reasoning trace, the parsed patch (or [] on parse failure), the full
    token breakdown (incl. reasoning_tokens), and — for multi-call methods —
    the estimator's intermediate outputs + token usage as a nested sub-record.
    """
    current = case["current_json"]
    gold_patch = case.get("expected_patch", [])
    vcs = case.get("value_constraints", [])
    method_runner = create_method(method)
    method_inputs = await method_runner.prepare(
        case, model=model, temperature=temperature, retries=max_retries, seed=seed)
    estimator_trace = method_inputs.estimator_trace
    est_in = estimator_trace["usage"]["in"] if estimator_trace else 0
    est_out = estimator_trace["usage"]["out"] if estimator_trace else 0
    est_reasoning = estimator_trace["usage"]["reasoning"] if estimator_trace else 0

    # A method may compute its patch itself (consensus/aggregation methods):
    # prepare() returns it in estimator_trace["direct_patch"] and no final
    # reviser call is made — usage is what the method's internal calls consumed.
    if estimator_trace and "direct_patch" in estimator_trace:
        model_patch = estimator_trace["direct_patch"]
        try:
            metrics = score_strict(current, gold_patch, model_patch, vcs)
        except Exception:  # noqa: BLE001 — a malformed direct patch scores as no-op
            model_patch = []
            metrics = score_strict(current, gold_patch, model_patch, vcs)
        return {
            "id": case["id"], "domain": case.get("domain", ""), "method": method,
            "target_leaves": case.get("target_leaves"),
            "parse_ok": True,
            "user_prompt": method_inputs.messages[-1]["content"] if method_inputs.messages else "",
            "raw_answer": "",
            "reasoning_content": None,
            "finish_reason": "direct",
            "model_patch": model_patch,
            "metrics": metrics,
            "usage": {
                "in": est_in, "out": est_out, "reasoning": est_reasoning,
                "rev_in": 0, "rev_out": 0, "rev_reasoning": 0,
                "est_in": est_in, "est_out": est_out, "est_reasoning": est_reasoning,
                "rev_answer_chars": 0, "rev_reasoning_chars": 0,
                "est_answer_chars": len(estimator_trace.get("raw_answer") or ""),
                "est_reasoning_chars": len(estimator_trace.get("reasoning_content") or ""),
            },
            "estimator": estimator_trace,
        }

    messages = method_inputs.messages
    user_prompt = messages[-1]["content"]  # the rendered revision prompt
    last_err = ""
    last_raw = ""
    last_reasoning: str | None = None
    last_finish: str | None = None
    last_usage_in = 0
    last_usage_out = 0
    last_usage_reasoning = 0
    for _ in range(max_retries):
        try:
            answer, usage = await complete(
                messages=messages, temperature=temperature,
                max_tokens=REV_MAX_TOKENS, enable_thinking=True, model=model, seed=seed,
            )
            last_raw = answer
            last_reasoning = usage.reasoning
            last_finish = usage.finish_reason
            last_usage_in = usage.input_tokens
            last_usage_out = usage.output_tokens
            last_usage_reasoning = usage.reasoning_tokens
            if usage.finish_reason == "length":
                raise ValueError("hit max_tokens (truncated)")
            model_patch = _extract_json_array(answer)
            # Opt-in self-review pass(es): show the model its own patch + the
            # applied artifact and let it emit corrections. Gold/metrics are
            # never shown — each review sees only legitimate, model-derived
            # state. With self_review_rounds > 1 the review repeats serially as
            # a growing conversation; the patch AFTER EVERY round is recorded in
            # review_stages so round-k performance is derivable post hoc.
            rev2_in = rev2_out = rev2_reasoning = 0
            review_delta = review_raw = review_reasoning = None
            review_stages = review_usages = None
            rounds = int(getattr(method_runner, "self_review_rounds", 1)) \
                if getattr(method_runner, "self_review", False) else 0
            if rounds:
                review_stages = [list(model_patch)]  # stage 0 = first pass
                review_usages = []
                convo = list(messages) + [{"role": "assistant", "content": answer}]
                delta_mode = getattr(method_runner, "self_review_delta", False)
                for _round in range(rounds):
                    applied_doc = None
                    try:
                        applied_doc = apply_patch(current, model_patch)
                        applied_str = json.dumps(applied_doc, ensure_ascii=False, indent=2)
                    except Exception:  # noqa: BLE001 — unappliable patch: say so
                        applied_str = "(your patch did not apply cleanly to the artifact)"
                    reminders = ""
                    if getattr(method_runner, "self_review_reminders", False):
                        reminders = build_review_reminders(current, model_patch, applied_doc)
                    check_part = SELF_REVIEW_CHECK_DELTA if delta_mode else SELF_REVIEW_CHECK
                    out_part = SELF_REVIEW_OUTPUT_DELTA if delta_mode else SELF_REVIEW_OUTPUT
                    review_user = check_part.format(applied=applied_str) + "\n\n" + (
                        reminders + "\n\n" + out_part if reminders else out_part)
                    convo.append({"role": "user", "content": review_user})
                    answer2, usage2 = await complete(
                        messages=convo, temperature=temperature,
                        max_tokens=REV_MAX_TOKENS, enable_thinking=True, model=model, seed=seed,
                    )
                    convo.append({"role": "assistant", "content": answer2})
                    rev2_in += usage2.input_tokens
                    rev2_out += usage2.output_tokens
                    rev2_reasoning += usage2.reasoning_tokens
                    review_usages.append({"in": usage2.input_tokens, "out": usage2.output_tokens,
                                          "reasoning": usage2.reasoning_tokens})
                    if usage2.finish_reason != "length":
                        try:
                            arr2 = _extract_json_array(answer2)
                            if delta_mode:
                                # Corrective ops are appended after the current
                                # patch (RFC 6902 sequential semantics: they
                                # address the post-patch artifact). [] confirms.
                                review_delta = arr2
                                review_raw = answer2
                                review_reasoning = usage2.reasoning
                                if arr2:
                                    model_patch = model_patch + arr2
                            else:
                                model_patch = arr2
                                last_raw = answer2
                                last_reasoning = usage2.reasoning
                                last_finish = usage2.finish_reason
                        except ValueError:
                            pass  # keep current patch if review output unparseable
                    review_stages.append(list(model_patch))
            metrics = score_strict(current, gold_patch, model_patch, vcs)
            return {
                "id": case["id"], "domain": case.get("domain", ""), "method": method,
                "target_leaves": case.get("target_leaves"),
                "parse_ok": True,
                "user_prompt": user_prompt,
                "raw_answer": last_raw,
                "reasoning_content": last_reasoning,
                "finish_reason": last_finish,
                "model_patch": model_patch,
                "review_delta": review_delta,
                "review_raw_answer": review_raw,
                "review_reasoning": review_reasoning,
                "review_stages": review_stages,
                "review_usages": review_usages,
                "metrics": metrics,
                "usage": {
                    "in": last_usage_in + est_in + rev2_in,
                    "out": last_usage_out + est_out + rev2_out,
                    "reasoning": last_usage_reasoning + est_reasoning + rev2_reasoning,
                    "rev_in": last_usage_in, "rev_out": last_usage_out,
                    "rev_reasoning": last_usage_reasoning,
                    "rev2_in": rev2_in, "rev2_out": rev2_out, "rev2_reasoning": rev2_reasoning,
                    "est_in": est_in, "est_out": est_out, "est_reasoning": est_reasoning,
                    # Character-length proxies: provider-side reasoning_tokens
                    # is 0 for vLLM+openai_gptoss (text only, no token split),
                    # so length(reasoning_content) and length(raw_answer) act
                    # as a usable proxy for the reasoning vs answer ratio.
                    "rev_answer_chars": len(last_raw or ""),
                    "rev_reasoning_chars": len(last_reasoning or ""),
                    "est_answer_chars": len(estimator_trace.get("raw_answer") or "") if estimator_trace else 0,
                    "est_reasoning_chars": len(estimator_trace.get("reasoning_content") or "") if estimator_trace else 0,
                },
                "estimator": estimator_trace,
            }
        except Exception as e:  # noqa: BLE001
            last_err = f"{type(e).__name__}: {e}"
            continue
    metrics = score_strict(current, gold_patch, [], vcs)
    metrics["apply_ok"] = False
    return {
        "id": case["id"], "domain": case.get("domain", ""), "method": method,
        "target_leaves": case.get("target_leaves"),
        "parse_ok": False, "error": last_err,
        "user_prompt": user_prompt,
        "raw_answer": last_raw,
        "reasoning_content": last_reasoning,
        "finish_reason": last_finish,
        "model_patch": [],
        "metrics": metrics,
        "usage": {
            "in": last_usage_in + est_in,
            "out": last_usage_out + est_out,
            "reasoning": last_usage_reasoning + est_reasoning,
            "rev_in": last_usage_in, "rev_out": last_usage_out,
            "rev_reasoning": last_usage_reasoning,
            "est_in": est_in, "est_out": est_out, "est_reasoning": est_reasoning,
            "rev_answer_chars": len(last_raw or ""),
            "rev_reasoning_chars": len(last_reasoning or ""),
            "est_answer_chars": len(estimator_trace.get("raw_answer") or "") if estimator_trace else 0,
            "est_reasoning_chars": len(estimator_trace.get("reasoning_content") or "") if estimator_trace else 0,
        },
        "estimator": estimator_trace,
    }


# ------------------------------- batch ------------------------------------


def _mean(xs: list[float]) -> float:
    return statistics.mean(xs) if xs else 0.0


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    ms = [r["metrics"] for r in results]
    keys = ["loc_precision", "loc_recall", "loc_f1", "value_recall",
            "direct_recall", "cascade_recall", "over_edit", "miss"]
    roll = {k: _mean([m[k] for m in ms]) for k in keys}
    roll["n"] = len(results)
    # Exact-match success: no spurious edits, no missed edits, every gold-changed
    # leaf set to the correct value.
    roll["success_rate"] = _mean([
        1.0 if (m["over_edit"] == 0 and m["miss"] == 0 and m["value_recall"] >= 1.0)
        else 0.0 for m in ms])
    # Location-only success: identified exactly the right set of leaves to edit
    # (no spurious edits, no missed edits) regardless of whether the values are
    # correct. Isolates dependency-location recognition from value computation.
    roll["loc_success_rate"] = _mean([
        1.0 if (m["over_edit"] == 0 and m["miss"] == 0) else 0.0 for m in ms])
    roll["parse_fail_rate"] = _mean([0.0 if r["parse_ok"] else 1.0 for r in results])
    roll["mean_in_tok"] = _mean([r["usage"]["in"] for r in results])
    roll["mean_out_tok"] = _mean([r["usage"]["out"] for r in results])
    roll["mean_reasoning_tok"] = _mean([r["usage"].get("reasoning", 0) for r in results])
    roll["mean_rev_in_tok"] = _mean([r["usage"].get("rev_in", 0) for r in results])
    roll["mean_rev_out_tok"] = _mean([r["usage"].get("rev_out", 0) for r in results])
    roll["mean_rev_reasoning_tok"] = _mean([r["usage"].get("rev_reasoning", 0) for r in results])
    roll["mean_est_in_tok"] = _mean([r["usage"].get("est_in", 0) for r in results])
    roll["mean_est_out_tok"] = _mean([r["usage"].get("est_out", 0) for r in results])
    roll["mean_est_reasoning_tok"] = _mean([r["usage"].get("est_reasoning", 0) for r in results])
    roll["mean_rev_answer_chars"] = _mean([r["usage"].get("rev_answer_chars", 0) for r in results])
    roll["mean_rev_reasoning_chars"] = _mean([r["usage"].get("rev_reasoning_chars", 0) for r in results])
    roll["mean_est_answer_chars"] = _mean([r["usage"].get("est_answer_chars", 0) for r in results])
    roll["mean_est_reasoning_chars"] = _mean([r["usage"].get("est_reasoning_chars", 0) for r in results])
    return roll


async def eval_all(
    cases: list[dict[str, Any]], *, method: str, model: str | None,
    temperature: float, max_retries: int, concurrency: int, seed: int | None = None,
    label: str | None = None,
) -> list[dict[str, Any]]:
    sem = asyncio.Semaphore(concurrency)
    results: list[dict[str, Any] | None] = [None] * len(cases)
    tag = label or method

    async def run(i: int, case: dict[str, Any]) -> None:
        async with sem:
            results[i] = await eval_one(
                case, method=method, model=model, temperature=temperature,
                max_retries=max_retries, seed=seed)
            r = results[i]
            print(f"  [{tag}] [{i+1}/{len(cases)}] {r['id']}  f1={r['metrics']['loc_f1']:.2f} "
                  f"val={r['metrics']['value_recall']:.2f} over={r['metrics']['over_edit']} "
                  f"miss={r['metrics']['miss']}{'' if r['parse_ok'] else '  [parse-fail]'}")

    await asyncio.gather(*(run(i, c) for i, c in enumerate(cases)))
    return [r for r in results if r is not None]


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--annotated", type=Path, default=Path("benchmark/data/annotated"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", default=None)
    p.add_argument("--methods", nargs="+", default=list(METHODS), choices=METHODS,
                   help="Which methods to run (default: all).")
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--seed", type=int, default=42,
                   help="Sampling seed forwarded to the model (default: 42). "
                        "Pass -1 to disable seeding.")
    p.add_argument("--max-retries", type=int, default=2)
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--include-excluded", action="store_true",
                   help="By default cases with excluded=true in the annotated "
                        "JSON are skipped from evaluation. Pass this flag to "
                        "include them anyway.")
    args = p.parse_args()

    files = sorted(args.annotated.glob("*.json"))
    cases = []
    n_excluded = 0
    for f in files:
        c = json.loads(f.read_text())
        if not (c.get("annotated") and c.get("expected_patch")):
            continue
        if c.get("excluded") and not args.include_excluded:
            n_excluded += 1
            continue
        cases.append(c)
    if args.limit:
        cases = cases[: args.limit]
    if not cases:
        print(f"No annotated cases in {args.annotated}")
        return
    if n_excluded:
        print(f"Skipping {n_excluded} cases marked excluded "
              f"(pass --include-excluded to keep them)")
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating {len(cases)} cases x {len(args.methods)} methods "
          f"(model={args.model or 'default'})")

    rollups: dict[str, dict[str, Any]] = {}
    labels: list[str] = []
    for method in args.methods:
        label = method
        labels.append(label)
        results = asyncio.run(eval_all(
            cases, method=method, model=args.model, temperature=args.temperature,
            max_retries=args.max_retries, concurrency=args.concurrency,
            seed=(None if args.seed < 0 else args.seed), label=label))
        roll = aggregate(results)
        rollups[label] = roll
        (args.out / f"per_case_{label}.json").write_text(json.dumps(results, ensure_ascii=False, indent=2))
        (args.out / f"rollup_{label}.json").write_text(json.dumps(roll, ensure_ascii=False, indent=2))

    (args.out / "comparison.json").write_text(json.dumps(rollups, ensure_ascii=False, indent=2))
    cols = ["success_rate", "loc_success_rate", "loc_f1", "loc_precision", "loc_recall", "value_recall",
            "direct_recall", "cascade_recall", "over_edit", "miss",
            "parse_fail_rate", "mean_in_tok", "mean_out_tok", "mean_reasoning_tok",
            "mean_rev_out_tok", "mean_rev_reasoning_tok",
            "mean_est_out_tok", "mean_est_reasoning_tok"]
    print(f"\n=== comparison (model={args.model or 'default'}, n={len(cases)}) ===")
    print(f"{'metric':16s}" + "".join(f"{m:>16s}" for m in labels))
    for c in cols:
        print(f"{c:16s}" + "".join(f"{rollups[m][c]:>16.3f}" for m in labels))
    print(f"\nWrote per_case_*/rollup_*/comparison.json to {args.out}")


if __name__ == "__main__":
    main()
