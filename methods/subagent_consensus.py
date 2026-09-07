"""Cost-scaling arm B: parallel self-consistency over the H+J reviser.

The SAME reviser prompt (byte-identical to sub_hist_json) is sampled k=3 times
with derived seeds (seed, seed+1000, seed+2000); each sample's patch is applied
to the current artifact, and the applied artifacts are merged DETERMINISTICALLY
into one consensus artifact, which is handed to the evaluator as a
synthesized patch. Three pre-declared aggregation modes:

  AND  intersection — a node changes only where the samples UNANIMOUSLY agree
       on the same new state (precision-leaning);
  OR   union — a node changes where ANY sample changes it; when samples
       conflict on the same node, the EARLIEST sample's change wins
       (recall-leaning);
  MAJ  majority vote — a node changes only to a state shared by a STRICT
       majority (> k/2) of samples; with no majority the current value is
       kept.

Merge rules (pre-declared, applied recursively against the CURRENT node):
  - all samples deep-equal            -> that value;
  - dicts (current and all samples)   -> merge per key over the union of keys
        (a missing key counts as "absent"; AND keeps the current state for a
        key unless every sample agrees on its new state, OR takes the majority
        state, ties broken by the earliest sample);
  - arrays                            -> if every sample has the same length,
        merge element-wise (recursing with the current element where one
        exists); otherwise the whole array is one voting unit: AND keeps the
        current array unless samples are unanimous, OR takes the majority
        serialization (ties -> earliest sample);
  - scalars / type mismatches         -> one voting unit, same vote rule.

Every sample's raw patch is stored in the trace (samples), so consensus at
k=1 (= sample 1 = the base cell, same seed) and k=2 (samples 1+2) is derivable
post hoc from the saved run without re-calling the model.

Integrity: every sample sees ONLY the legitimate context (the same messages as
the base cell); aggregation is deterministic vote logic over the model's own
outputs — no gold, no values from any other prompt.
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, ClassVar

from agents.llm import complete
from benchmark.jsonptr import apply_patch
from methods.common import MethodInputs, RevisionMethod
from methods.subagent_revision import SUBAGENT_SYSTEM, build_subagent_user


def _majority(values: list[Any]) -> Any:
    """Most common serialized value; ties broken by earliest occurrence."""
    ser = [json.dumps(v, sort_keys=True) for v in values]
    best = Counter(ser).most_common(1)[0][0]
    return json.loads(best)


_ABSENT = object()


def _ser_state(x: Any) -> str:
    return "\x00ABSENT" if x is _ABSENT else json.dumps(x, sort_keys=True)


def _vote_unit(cur: Any, samples: list[Any], mode: str) -> Any:
    """Treat the node as one indivisible unit and decide its consensus state.
    A sample's state is its value at this node or _ABSENT; `cur` likewise."""
    cur_s = _ser_state(cur)
    states = [_ser_state(s) for s in samples]
    if mode == "and":
        # unanimity on the same new state required to move off current
        if len(set(states)) == 1 and states[0] != cur_s:
            return samples[0]
        return cur
    if mode == "or":
        # union of changes: the earliest sample that differs from current wins
        for st_, smp in zip(states, samples):
            if st_ != cur_s:
                return smp
        return cur
    # mode == "maj": strict majority (> k/2) on a new state, else keep current
    state, cnt = Counter(states).most_common(1)[0]
    if cnt * 2 > len(samples) and state != cur_s:
        for st_, smp in zip(states, samples):
            if st_ == state:
                return smp
    return cur


def merge_consensus(cur: Any, samples: list[Any], mode: str) -> Any:
    """Merge sample artifacts against the current artifact. `cur` may be
    _ABSENT (key absent before the edit); sample values may be _ABSENT
    (key absent in that sample's artifact)."""
    present = [s for s in samples if s is not _ABSENT]
    if len(present) == len(samples):
        ser = {json.dumps(s, sort_keys=True) for s in samples}
        if len(ser) == 1:
            return samples[0]
    if all(isinstance(s, dict) for s in present) and len(present) == len(samples) \
            and (isinstance(cur, dict) or cur is _ABSENT):
        cur_d = cur if isinstance(cur, dict) else {}
        keys: list[str] = list(cur_d.keys())
        for s in samples:
            for k in s:
                if k not in keys:
                    keys.append(k)
        out: dict[str, Any] = {}
        for k in keys:
            merged = merge_consensus(
                cur_d.get(k, _ABSENT),
                [s.get(k, _ABSENT) for s in samples],
                mode,
            )
            if merged is not _ABSENT:
                out[k] = merged
        return out
    if all(isinstance(s, list) for s in present) and len(present) == len(samples):
        lengths = {len(s) for s in samples}
        # Element-wise merging is only meaningful when positions still align
        # with the CURRENT array: any length change (insertion/removal) makes
        # index-based alignment undefined, so the whole array becomes one
        # voting unit.
        if len(lengths) == 1 and isinstance(cur, list) and lengths == {len(cur)}:
            out_l = []
            for i in range(len(cur)):
                merged = merge_consensus(cur[i], [s[i] for s in samples], mode)
                out_l.append(None if merged is _ABSENT else merged)
            return out_l
        return _vote_unit(cur, samples, mode)
    return _vote_unit(cur, samples, mode)


def _esc(k: str) -> str:
    return str(k).replace("~", "~0").replace("/", "~1")


def _eq(a: Any, b: Any) -> bool:
    return json.dumps(a, sort_keys=True) == json.dumps(b, sort_keys=True)


def _diff_array_ops(cur: list, new: list, base_path: str) -> list[dict[str, Any]]:
    """Element-aware ops for a changed array. The strict scorer binds INSERTED
    elements through add ops, so a length-changing array must not be expressed
    as a whole-array replace. Alignment: longest deep-equal prefix and suffix;
    the middle block becomes replaces plus adds (growth) or removes (shrink)."""
    p = 0
    while p < len(cur) and p < len(new) and _eq(cur[p], new[p]):
        p += 1
    s = 0
    while s < (len(cur) - p) and s < (len(new) - p) and _eq(cur[len(cur) - 1 - s], new[len(new) - 1 - s]):
        s += 1
    cur_mid = cur[p:len(cur) - s]
    new_mid = new[p:len(new) - s]
    ops: list[dict[str, Any]] = []
    n_common = min(len(cur_mid), len(new_mid))
    for i in range(n_common):
        if not _eq(cur_mid[i], new_mid[i]):
            ops.append({"op": "replace", "path": f"{base_path}/{p + i}", "value": new_mid[i]})
    if len(new_mid) > len(cur_mid):
        for j, v in enumerate(new_mid[n_common:]):
            ops.append({"op": "add", "path": f"{base_path}/{p + n_common + j}", "value": v})
    elif len(cur_mid) > len(new_mid):
        for _ in range(len(cur_mid) - n_common):
            ops.append({"op": "remove", "path": f"{base_path}/{p + n_common}"})
    return ops


def consensus_to_patch(current: Any, consensus: Any) -> list[dict[str, Any]]:
    """Synthesize an RFC 6902 patch transforming `current` into `consensus`.

    Dict keys are handled at top-level granularity (replace/add/remove —
    positionally stable, no index arithmetic); a changed LIST value anywhere
    under a top-level key is diffed element-aware via _diff_array_ops so that
    insertions surface as add ops (the scorer binds new elements through
    them). Nested structures are walked recursively."""
    if not isinstance(current, dict) or not isinstance(consensus, dict):
        raise ValueError("consensus_to_patch expects dict roots")

    ops: list[dict[str, Any]] = []

    def walk(cur: Any, new: Any, path: str) -> None:
        if _eq(cur, new):
            return
        if isinstance(cur, dict) and isinstance(new, dict):
            for k in cur:
                if k not in new:
                    ops.append({"op": "remove", "path": f"{path}/{_esc(k)}"})
            for k, v in new.items():
                if k not in cur:
                    ops.append({"op": "add", "path": f"{path}/{_esc(k)}", "value": v})
                else:
                    walk(cur[k], v, f"{path}/{_esc(k)}")
            return
        if isinstance(cur, list) and isinstance(new, list):
            if len(cur) == len(new):
                for i in range(len(cur)):
                    walk(cur[i], new[i], f"{path}/{i}")
            else:
                ops.extend(_diff_array_ops(cur, new, path))
            return
        ops.append({"op": "replace", "path": path, "value": new})

    walk(current, consensus, "")
    return ops


def build_consensus_patch(current: Any, consensus: Any,
                          samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Patch for the consensus artifact. If the consensus exactly equals some
    sample's applied artifact, reuse that sample's RAW patch (earliest match):
    the scorer's inserted-element binding is sensitive to op FORM, and the
    model's own ops carry it faithfully. Only true mixtures are synthesized."""
    for s in samples:
        if s.get("_applied") is not None and _eq(s["_applied"], consensus):
            return list(s["patch"])
    return consensus_to_patch(current, consensus)


def medoid_index(applied: list[Any]) -> int:
    """Index of the artifact with minimum mean leaf-disagreement to the rest
    (ties -> earliest). The zero-extra-cost selector."""
    def lf(doc, path=""):
        if isinstance(doc, dict):
            for k, v in doc.items(): yield from lf(v, f"{path}/{_esc(str(k))}")
        elif isinstance(doc, list):
            for i, v in enumerate(doc): yield from lf(v, f"{path}/{i}")
        else:
            yield path, json.dumps(doc, sort_keys=True)
    lfs = [dict(lf(a)) for a in applied]
    best_i, best_d = 0, None
    for i, li in enumerate(lfs):
        ds = []
        for j, lj in enumerate(lfs):
            if i == j: continue
            keys = set(li) | set(lj)
            ds.append(sum(1 for k in keys if li.get(k) != lj.get(k)) / max(len(keys), 1))
        d = sum(ds) / len(ds) if ds else 1.0
        if best_d is None or d < best_d:
            best_i, best_d = i, d
    return best_i


class SubAgentConsensusBase(RevisionMethod):
    mode: ClassVar[str]            # "and" | "or" | "maj"
    n_samples: ClassVar[int] = 3

    async def _collect_samples(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> tuple[list[dict[str, str]], list[dict[str, Any]], dict[str, int]]:
        user = build_subagent_user(case, include_history=True, include_json=True)
        messages = [
            {"role": "system", "content": SUBAGENT_SYSTEM},
            {"role": "user", "content": user},
        ]
        # Lazy imports avoid an import cycle. REV_MAX_TOKENS must be the
        # evaluator's value: the response cache keys on the full request, so a
        # different max_tokens breaks the sample-1 == base-cell identity.
        from benchmark.evaluator import REV_MAX_TOKENS, _extract_json_array
        usage_tot = {"in": 0, "out": 0, "reasoning": 0}
        samples: list[dict[str, Any]] = []
        for i in range(self.n_samples):
            member_seed = seed + i if (seed is not None and i > 0) else seed
            raw = ""
            reasoning = None
            patch = None
            err = ""
            s_usage = {"in": 0, "out": 0, "reasoning": 0}
            for _ in range(retries):
                try:
                    text, u = await complete(
                        messages=messages, temperature=temperature,
                        max_tokens=REV_MAX_TOKENS, enable_thinking=True,
                        model=model, seed=member_seed)
                    usage_tot["in"] += u.input_tokens
                    usage_tot["out"] += u.output_tokens
                    usage_tot["reasoning"] += u.reasoning_tokens
                    s_usage["in"] += u.input_tokens
                    s_usage["out"] += u.output_tokens
                    s_usage["reasoning"] += u.reasoning_tokens
                    raw, reasoning = text, u.reasoning
                    if u.finish_reason == "length":
                        raise ValueError("hit max_tokens (truncated)")
                    patch = _extract_json_array(text)
                    break
                except Exception as e:  # noqa: BLE001
                    err = f"{type(e).__name__}: {e}"
            applied = None
            if patch is not None:
                try:
                    applied = apply_patch(case["current_json"], patch)
                except Exception as e:  # noqa: BLE001
                    err = f"apply: {type(e).__name__}: {e}"
            samples.append({"seed": member_seed, "patch": patch, "applied_ok": applied is not None,
                            "usage": s_usage,
                            "raw_answer": raw, "reasoning_content": reasoning, "error": err,
                            "_applied": applied})
        return messages, samples, usage_tot

    async def prepare(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> MethodInputs:
        messages, samples, usage_tot = await self._collect_samples(
            case, model=model, temperature=temperature, retries=retries, seed=seed)
        current = case["current_json"]
        applied_docs = [s["_applied"] for s in samples if s["_applied"] is not None]
        if applied_docs:
            consensus = merge_consensus(current, applied_docs, self.mode)
            if consensus is _ABSENT:
                consensus = current
        else:
            consensus = current
        direct_patch = build_consensus_patch(current, consensus, samples)
        for s in samples:
            s.pop("_applied", None)
        trace = {
            "direct_patch": direct_patch,
            "mode": self.mode,
            "n_samples": self.n_samples,
            "samples": samples,
            "raw_answer": "",
            "reasoning_content": None,
            "usage": usage_tot,
            "error": "; ".join(s["error"] for s in samples if s["error"]),
        }
        return MethodInputs(messages=messages, estimator_trace=trace)


JUDGE_SYSTEM = """\
You are the selector for a revision task on a JSON artifact that is edited over multiple turns.

You are given the conversation that built the artifact, the current JSON artifact, a single requested edit (and, when the edit introduces a new entity, reference data for that entity), and several candidate JSON Patches produced independently for that edit. Exactly one candidate must be chosen.

Judge each candidate against the artifact and the requested edit: the right candidate applies the requested edit and changes every field whose value is FORCED to change as a consequence — values computed or derived from a changed field, fields that name or refer to a changed entity, and control, policy, status, or routing fields gated by a value that crosses a threshold — while leaving every other field untouched. Verify the arithmetic of changed values, check that no forced change is missing, and check that nothing unrelated is modified.

Output ONLY the number of the chosen candidate. No prose, no punctuation.\
"""


class SubHistJsonJudge5Method(SubAgentConsensusBase):
    """Selection by an LLM judge over 5 parallel samples: one extra call sees
    the legitimate context plus the candidate patches and outputs an index.
    The chosen sample's RAW patch is scored. Unparseable judge output falls
    back to the medoid (pre-declared). The judge emits a choice, never values."""
    name = "sub_hist_json_judge5"
    mode = "judge"
    n_samples = 5

    async def prepare(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> MethodInputs:
        from benchmark.evaluator import REV_MAX_TOKENS
        messages, samples, usage_tot = await self._collect_samples(
            case, model=model, temperature=temperature, retries=retries, seed=seed)
        current = case["current_json"]
        cands = [s for s in samples if s["_applied"] is not None]
        judge_raw = ""
        judge_reasoning = None
        choice = None
        if len(cands) >= 2:
            cand_txt = "\n\n".join(
                f"Candidate {i + 1}:\n" + json.dumps(s["patch"], ensure_ascii=False)
                for i, s in enumerate(cands))
            judge_user = (build_subagent_user(case, include_history=True, include_json=True)
                          + "\n\nCandidate patches:\n\n" + cand_txt)
            try:
                text, u = await complete(
                    messages=[{"role": "system", "content": JUDGE_SYSTEM},
                              {"role": "user", "content": judge_user}],
                    temperature=temperature, max_tokens=REV_MAX_TOKENS,
                    enable_thinking=True, model=model, seed=seed)
                usage_tot["in"] += u.input_tokens
                usage_tot["out"] += u.output_tokens
                usage_tot["reasoning"] += u.reasoning_tokens
                judge_raw, judge_reasoning = text, u.reasoning
                import re as _re
                m = _re.search(r"\d+", text)
                if m:
                    idx = int(m.group(0))
                    if 1 <= idx <= len(cands):
                        choice = idx - 1
            except Exception:  # noqa: BLE001 — judge is best-effort
                pass
        if choice is None and cands:
            choice = medoid_index([s["_applied"] for s in cands])
            fallback = "medoid"
        else:
            fallback = ""
        direct_patch = list(cands[choice]["patch"]) if cands else []
        for s in samples:
            s.pop("_applied", None)
        trace = {
            "direct_patch": direct_patch,
            "mode": "judge",
            "n_samples": self.n_samples,
            "samples": samples,
            "judge_choice": choice,
            "judge_fallback": fallback,
            "judge_raw": judge_raw,
            "judge_reasoning": judge_reasoning,
            "raw_answer": "",
            "reasoning_content": None,
            "usage": usage_tot,
            "error": "; ".join(s["error"] for s in samples if s["error"]),
        }
        return MethodInputs(messages=messages, estimator_trace=trace)


class SubHistJsonJudge2Method(SubHistJsonJudge5Method):
    """Judge k-ablation: first TWO samples (replayed from cache) + a new judge
    call whose candidate list holds only those two."""
    name = "sub_hist_json_judge2"
    n_samples = 2


class SubHistJsonJudge3Method(SubHistJsonJudge5Method):
    """Judge k-ablation: first THREE samples."""
    name = "sub_hist_json_judge3"
    n_samples = 3


class SubHistJsonJudge4Method(SubHistJsonJudge5Method):
    """Judge k-ablation: first FOUR samples."""
    name = "sub_hist_json_judge4"
    n_samples = 4


MERGE_SYSTEM = """\
You are a revision agent for a JSON artifact that is edited over multiple turns.

You are given context for the artifact — the conversation that built it and its current state — a single requested edit (and, when the edit introduces a new entity, reference data for that entity), and several candidate JSON Patches that were produced INDEPENDENTLY for that edit. The candidates are references only: each may be correct, partially correct, or wrong, and they may disagree with one another.

Produce the single best revision. Use the candidates as evidence — adopt the operations they agree on, choose between them where they conflict, and add or correct any operation they all missed or got wrong. Do not feel bound to any candidate: your output may match one of them, combine several, or differ from all of them where they are mistaken.

Compute the full consequence of the edit and apply it: change the field the edit changes directly, and every other field whose value is FORCED to change because it depends on a changed field — values computed or derived from a changed field (do the arithmetic; pick the correct branch of any conditional), fields that name, refer to, route to, or are assigned to a changed entity, and control, policy, status, or routing fields gated by a value that crosses a threshold or moves into a different band — while leaving every other field untouched. Verify the arithmetic of changed values, check that no forced change is missing, and check that nothing unrelated is modified.

# Output format
Produce a JSON Patch (RFC 6902): a JSON array of operations that transforms the current artifact into the corrected version, applied strictly in order. Each operation has the form {"op": "replace" | "add" | "remove", "path": "<RFC 6901 JSON Pointer>", "value": <new value>} ("value" is omitted for "remove"). Output ONLY the JSON array — no prose, no code fences.\
"""


class SubHistJsonMerge5Method(SubAgentConsensusBase):
    """Fusion over 5 parallel samples: one extra call sees the legitimate
    context plus the candidate patches and GENERATES a new patch, using the
    candidates as references (may combine and correct, not just select). Unlike
    the judge this is not in-support — it can exceed the pass@k ceiling but can
    also introduce novel errors. Gold is never shown. Falls back to the medoid
    if the merge output is unparseable (pre-declared)."""
    name = "sub_hist_json_merge5"
    mode = "merge"
    n_samples = 5

    async def prepare(
        self,
        case: dict[str, Any],
        *,
        model: str | None,
        temperature: float,
        retries: int,
        seed: int | None = None,
    ) -> MethodInputs:
        from benchmark.evaluator import REV_MAX_TOKENS, _extract_json_array
        messages, samples, usage_tot = await self._collect_samples(
            case, model=model, temperature=temperature, retries=retries, seed=seed)
        cands = [s for s in samples if s["_applied"] is not None]
        merge_raw = ""
        merge_reasoning = None
        merged_patch = None
        if len(cands) >= 2:
            cand_txt = "\n\n".join(
                f"Candidate {i + 1}:\n" + json.dumps(s["patch"], ensure_ascii=False)
                for i, s in enumerate(cands))
            merge_user = (build_subagent_user(case, include_history=True, include_json=True)
                          + "\n\nCandidate patches:\n\n" + cand_txt)
            try:
                text, u = await complete(
                    messages=[{"role": "system", "content": MERGE_SYSTEM},
                              {"role": "user", "content": merge_user}],
                    temperature=temperature, max_tokens=REV_MAX_TOKENS,
                    enable_thinking=True, model=model, seed=seed)
                usage_tot["in"] += u.input_tokens
                usage_tot["out"] += u.output_tokens
                usage_tot["reasoning"] += u.reasoning_tokens
                merge_raw, merge_reasoning = text, u.reasoning
                if u.finish_reason != "length":
                    arr = _extract_json_array(text)
                    # Accept only a well-formed op list; a generated patch with
                    # any non-dict element (the model sometimes emits stray
                    # scalars) is treated like an unparseable output and falls
                    # back to the medoid, so direct_patch is always valid.
                    if isinstance(arr, list) and all(
                            isinstance(o, dict) and "op" in o and "path" in o for o in arr):
                        merged_patch = arr
            except Exception:  # noqa: BLE001 — merge is best-effort
                pass
        if merged_patch is None:
            fallback = "medoid"
            merged_patch = list(cands[medoid_index([s["_applied"] for s in cands])]["patch"]) \
                if cands else []
        else:
            fallback = ""
        for s in samples:
            s.pop("_applied", None)
        trace = {
            "direct_patch": merged_patch,
            "mode": "merge",
            "n_samples": self.n_samples,
            "samples": samples,
            "merge_fallback": fallback,
            "merge_raw": merge_raw,
            "merge_reasoning": merge_reasoning,
            "raw_answer": merge_raw,
            "reasoning_content": merge_reasoning,
            "usage": usage_tot,
            "error": "; ".join(s["error"] for s in samples if s["error"]),
        }
        return MethodInputs(messages=messages, estimator_trace=trace)


class SubHistJsonMerge2Method(SubHistJsonMerge5Method):
    """Merge k-ablation: first TWO samples."""
    name = "sub_hist_json_merge2"
    n_samples = 2


class SubHistJsonMerge3Method(SubHistJsonMerge5Method):
    """Merge k-ablation: first THREE samples."""
    name = "sub_hist_json_merge3"
    n_samples = 3


class SubHistJsonMerge4Method(SubHistJsonMerge5Method):
    """Merge k-ablation: first FOUR samples."""
    name = "sub_hist_json_merge4"
    n_samples = 4


class SubHistJsonCons3AndMethod(SubAgentConsensusBase):
    name = "sub_hist_json_cons3_and"
    mode = "and"


class SubHistJsonCons3OrMethod(SubAgentConsensusBase):
    name = "sub_hist_json_cons3_or"
    mode = "or"


class SubHistJsonCons3MajMethod(SubAgentConsensusBase):
    name = "sub_hist_json_cons3_maj"
    mode = "maj"


class SubHistJsonCons9AndMethod(SubAgentConsensusBase):
    """Nine-sample variant (samples 1-7 replay cons7 from cache; s+7..s+8 new)
    for the 122B equal-cost study, whose judge call is worth ~3.2 samples."""
    name = "sub_hist_json_cons9_and"
    mode = "and"
    n_samples = 9


class SubHistJsonCons7AndMethod(SubAgentConsensusBase):
    """Seven-sample variant (samples 1-5 replay cons5 from cache; s+5..s+6
    new) — the displayed equal-cost ladder ends at k=7, so this generates
    exactly what is shown. MED/MAJ/OR/AND at k<=7 derive from the samples."""
    name = "sub_hist_json_cons7_and"
    mode = "and"
    n_samples = 7


class SubHistJsonCons8AndMethod(SubAgentConsensusBase):
    """Eight-sample variant for the equal-cost comparison against the judge
    (whose extra call is worth 0.9-3.2 samples depending on the model).
    Samples 1-5 replay the cons5 calls from cache; only s+5..s+7 are new.
    MED@k / MAJ@k / OR@k for k<=8 are derived from the stored samples."""
    name = "sub_hist_json_cons8_and"
    mode = "and"
    n_samples = 8


class SubHistJsonCons5AndMethod(SubAgentConsensusBase):
    """Five-sample variant (seeds s..s+4000); samples 1-3 replay the cons3
    calls from cache. One cell suffices for storage — k=1..5 x and/or/maj are
    all derived from the stored samples in analysis."""
    name = "sub_hist_json_cons5_and"
    mode = "and"
    n_samples = 5
