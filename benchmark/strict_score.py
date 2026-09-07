"""Identity-anchored ("strict") scoring for cascade edits.

Position-based net-effect scoring cannot distinguish, for an id-less array with
a structural change, between

  (1) a FAILED CASCADE on a survivor that shifted into a slot — the model edited
      the right element but computed the wrong value (a value_recall failure),
  (2) a SPURIOUS edit to a slot whose element should have stayed/been removed —
      the model changed a leaf it should not have (an over_edit),

because both surface, position-wise, as "the value at /arr/i changed to X".

We recover element identity by tracking element PROVENANCE through the RFC 6902
patch (the same offset bookkeeping the annotator GUI uses) on BOTH the gold and
the model patch, then compare per ORIGINAL-ELEMENT identity rather than per
position. No element ids and no re-annotation are needed: provenance is a
deterministic function of the existing gold/model patches. Structural ops
(add/remove) on top-level arrays are tracked; nested-array structural ops are
out of scope for now (the benchmark's structural edits are all top-level).
"""
from __future__ import annotations

import math
import re
from typing import Any

from benchmark.jsonptr import apply_patch, pointer_get

# Relative tolerance for comparing non-integral computed floats (e.g. a storage
# estimate `rows * bytes / 1024^3`). The gold carries full float64 precision
# while a model reasonably rounds, so an exact `==` would penalize the rounding
# even when the cascade arithmetic is correct. Integers (counts, costs, exact
# lookups) are still compared exactly — the tolerance applies ONLY when at least
# one side is a non-integral float. 1e-4 absorbs float64 representation noise and
# display-precision rounding (e.g. 2.0116568 vs 2.01164) but is tight enough that
# a genuine arithmetic discrepancy (e.g. a few dollars of compounded per-line
# rounding on a multi-thousand total) still surfaces as a value mismatch.
_FLOAT_REL_TOL = 1e-4

# Toggle for the exact-final-array provenance reconciliation (see score_strict).
# True in production; flipped to False only by tests to isolate its effect.
_EXACT_ARRAY_RECONCILE = True

_SENT = object()
# Sentinel string a gold add-value sub-field can hold to mark itself a FREE field
# (any model value accepted) — e.g. an inserted element's arbitrary identifier.
_ANY = "<any>"


def _norm(v: Any) -> Any:
    """Normalize a leaf value for equality. For strings, ignore trailing
    whitespace and a single trailing period so cosmetic differences like
    "Invoice X" vs "Invoice X." do not count as a value mismatch. Non-string
    values (and the _SENT sentinel) are returned unchanged."""
    if isinstance(v, str):
        s = v.rstrip()
        if s.endswith("."):
            s = s[:-1].rstrip()
        return s
    return v


def _num_eq(a: Any, b: Any) -> bool:
    """Numeric equality with rounding tolerance for non-integral floats.

    Booleans (which are ints in Python) and integer-valued numbers compare
    exactly; only when at least one operand is a non-integral float do we accept
    a relative difference up to `_FLOAT_REL_TOL`. Returns None if the pair is not
    a plain number pair (so the caller falls back to string-aware equality)."""
    if isinstance(a, bool) or isinstance(b, bool):
        return a == b
    if isinstance(a, (int, float)) and isinstance(b, (int, float)):
        if float(a).is_integer() and float(b).is_integer():
            return a == b
        return math.isclose(a, b, rel_tol=_FLOAT_REL_TOL, abs_tol=1e-9)
    return None


def _norm_eq(a: Any, b: Any) -> bool:
    """String-aware equality: trailing whitespace / a trailing period are
    ignored (symmetric); non-integral floats compare with a rounding tolerance.
    Sentinels and other non-strings fall back to `==`."""
    n = _num_eq(a, b)
    if n is not None:
        return n
    return _norm(a) == _norm(b)


def _leaves(obj: Any, pre: str = "") -> list[tuple[str, Any]]:
    out: list[tuple[str, Any]] = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            out += _leaves(v, pre + "/" + str(k).replace("~", "~0").replace("/", "~1"))
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            out += _leaves(v, pre + "/" + str(i))
    else:
        out.append((pre, obj))
    return out


def track_prov(current: Any, patch: list[dict]) -> tuple[dict, dict]:
    """Track element provenance through a patch.

    Returns (prov, removed):
      prov[array_path]    = [orig_index | None, ...] giving, for each FINAL
                            position, the ORIGINAL index of the element now there
                            (None = inserted by an `add`).
      removed[array_path] = set of ORIGINAL indices removed by a `remove`.

    Only structural (add/remove) ops move elements; replaces change values in
    place and leave provenance untouched."""
    prov: dict[str, list | None] = {}
    removed: dict[str, set] = {}

    def ensure(P: str) -> list | None:
        if P not in prov:
            arr = pointer_get(current, P, default=None)
            prov[P] = list(range(len(arr))) if isinstance(arr, list) else None
        return prov[P]

    for op in patch:
        m = re.match(r"^(.*)/(\d+|-)$", op.get("path", ""))
        if not m:
            continue
        P, seg = m.group(1), m.group(2)
        if op.get("op") == "remove":
            p = ensure(P)
            if p is None:
                continue
            idx = len(p) - 1 if seg == "-" else int(seg)
            if 0 <= idx < len(p):
                removed.setdefault(P, set()).add(p[idx])
                p.pop(idx)
        elif op.get("op") == "add":
            p = ensure(P)
            if p is None:
                continue
            idx = len(p) if seg == "-" else int(seg)
            p.insert(idx, None)
    return prov, removed


def _elem_sim(a: Any, b: Any) -> float:
    """Similarity of two array elements = fraction of leaf (path,value) pairs
    that agree. Used to recover provenance by content when the patch op
    structure does not (a wholesale array replace)."""
    la, lb = dict(_leaves(a)), dict(_leaves(b))
    keys = set(la) | set(lb)
    if not keys:
        return 1.0 if a == b else 0.0
    return sum(1 for k in keys if la.get(k) == lb.get(k)) / len(keys)


def _align_prov(orig: list, new: list, thresh: float = 0.5) -> list:
    """Content-align a model's FINAL array `new` to the `orig` array, returning
    prov: for each new index, the matched original index or None (= #NEW).
    Order-preserving maximum-similarity match (weighted LCS); a pairing needs
    similarity >= thresh, so an inserted element (low similarity to every
    original) stays #NEW while shifted survivors keep their origin identity.
    This makes provenance robust to HOW the patch was expressed — granular
    add/remove ops or a single `replace /array = [...]` give the same result."""
    n, m = len(orig), len(new)
    sim = [[_elem_sim(orig[i], new[j]) for j in range(m)] for i in range(n)]
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            best = max(dp[i - 1][j], dp[i][j - 1])
            s = sim[i - 1][j - 1]
            if s >= thresh and dp[i - 1][j - 1] + s > best:
                best = dp[i - 1][j - 1] + s
            dp[i][j] = best
    prov: list = [None] * m
    i, j = n, m
    while i > 0 and j > 0:
        s = sim[i - 1][j - 1]
        if s >= thresh and abs(dp[i][j] - (dp[i - 1][j - 1] + s)) < 1e-9:
            prov[j - 1] = i - 1
            i -= 1
            j -= 1
        elif dp[i - 1][j] >= dp[i][j - 1]:
            i -= 1
        else:
            j -= 1
    return prov


def _to_identity(final_path: str, prov: dict) -> str:
    """Rewrite a final-position leaf path into an identity-anchored path by
    replacing the (first) structural-array index with the ORIGINAL index of the
    element now there: /tasks/2/start_date -> /tasks#3/start_date."""
    for P in prov:
        if prov[P] is None:
            continue
        if final_path == P or final_path.startswith(P + "/"):
            m = re.match(r"^/(\d+)(/.*)?$", final_path[len(P):])
            if m:
                fidx = int(m.group(1))
                sub = m.group(2) or ""
                if 0 <= fidx < len(prov[P]):
                    orig = prov[P][fidx]
                    tag = f"#{orig}" if orig is not None else "#NEW"
                    return f"{P}{tag}{sub}"
    return final_path


def _ident_map(doc: Any, prov: dict) -> dict[str, Any]:
    return {_to_identity(p, prov): v for p, v in _leaves(doc)}


def strict_score(current: Any, gold_patch: list[dict], model_patch: list[dict]) -> dict:
    """Score a model patch against a gold patch by ORIGINAL-ELEMENT identity.

    Metrics:
      value_recall : fraction of required outcomes achieved. A required outcome
                     is a leaf the gold changes (gold value != original, by
                     identity) or an element the gold removes.
      miss         : required outcomes the model failed to even attempt
                     (changed leaf left at original / absent; removal not done).
      over_edit    : leaves the model changed that the gold leaves at their
                     original value, or leaves of an element the gold removed,
                     or brand-new leaves the gold never introduces.
    `_details` lists per-leaf verdicts for debugging.
    """
    gold_after = apply_patch(current, gold_patch)
    model_after = apply_patch(current, model_patch)
    g_prov, g_removed = track_prov(current, gold_patch)
    m_prov, _ = track_prov(current, model_patch)

    # Identity tagging must be CONSISTENT across O/G/M: if EITHER patch
    # restructures an array, that array's elements are keyed by origin index in
    # all three (the side that did not restructure uses identity provenance).
    arr_paths = set(g_prov) | set(m_prov)

    def _fill(prov: dict) -> dict:
        for P in arr_paths:
            if prov.get(P) is None:
                arr = pointer_get(current, P, default=None)
                prov[P] = list(range(len(arr))) if isinstance(arr, list) else None
        return prov

    O = _ident_map(current, _fill({}))
    G = _ident_map(gold_after, _fill(g_prov))
    M = _ident_map(model_after, _fill(m_prov))

    model_keeps = {
        P: {i for i in (m_prov.get(P) or []) if i is not None}
        for P in arr_paths
    }

    req_total = req_ok = 0
    over_edit = 0
    miss = 0
    details: list[tuple[str, str, str]] = []

    # 0) elements the gold REMOVES — a required outcome (direct/cascade removal)
    for P, idxs in g_removed.items():
        for i in idxs:
            req_total += 1
            if i in model_keeps.get(P, set()):
                miss += 1
                details.append(("MISS(not-removed)", f"{P}#{i}", "element kept"))
            else:
                req_ok += 1

    # 1) leaves the gold prescribes (every identity leaf in gold_after)
    for ip, gv in G.items():
        ov = O.get(ip, _SENT)
        mv = M.get(ip, _SENT)
        if ov is not _SENT and gv == ov:
            # should-stay leaf: a model change to a non-gold value = over_edit
            if mv is not _SENT and mv != gv:
                over_edit += 1
                details.append(("OVEREDIT(should-stay)", ip, f"orig={ov} model={mv}"))
            continue
        # required change (gv != ov, or this identity is new)
        req_total += 1
        if mv is _SENT:
            miss += 1
            details.append(("MISS(absent)", ip, f"gold={gv}"))
        elif mv == gv:
            req_ok += 1
        else:
            details.append(("VALUE_FAIL", ip, f"gold={gv} model={mv}"))

    # 2) leaves the model changed that the gold never prescribes -> over_edit
    #    (spurious new leaves, or edits to elements the gold removed/left as-is)
    for ip, mv in M.items():
        if ip in G:
            continue
        ov = O.get(ip, _SENT)
        if ov is _SENT:
            over_edit += 1
            details.append(("OVEREDIT(spurious-new)", ip, f"model={mv}"))
        elif mv != ov:
            over_edit += 1
            details.append(("OVEREDIT(changed-untouched)", ip, f"orig={ov} model={mv}"))

    return {
        "value_recall": round(req_ok / req_total, 4) if req_total else 1.0,
        "req_total": req_total,
        "req_ok": req_ok,
        "miss": miss,
        "over_edit": over_edit,
        "_details": details,
    }


# --------------------------------------------------------------------------- #
# Full evaluator-compatible scorer (drop-in for benchmark.evaluator.score).
# Produces the same metric dict, but every lookup is anchored to element
# identity, so it equals the position-based scorer when there is no structural
# change and correctly attributes failed cascades vs spurious edits when there
# is one.
# --------------------------------------------------------------------------- #

_ARR_TAIL = re.compile(r"^(.*)/(\d+|-)$")


def _is_array_tail(path: str) -> bool:
    return _ARR_TAIL.match(path) is not None


def _path_to_ident(path: str, prov: dict) -> str:
    for P in prov:
        if prov[P] is None:
            continue
        if path == P or path.startswith(P + "/"):
            m = re.match(r"^/(\d+)(/.*)?$", path[len(P):])
            if m:
                fidx = int(m.group(1))
                sub = m.group(2) or ""
                if 0 <= fidx < len(prov[P]):
                    o = prov[P][fidx]
                    return f"{P}#{o}{sub}" if o is not None else f"{P}#NEW{sub}"
    return path


def _gold_removals(current: Any, gold_patch: list[dict]):
    """Replay the gold patch's structural ops to resolve each array-element
    removal to its ORIGINAL index (a remove must be identified at its
    pre-removal position). Returns [(array_path, origin_index, op)]."""
    prov: dict[str, list | None] = {}

    def ensure(P):
        if P not in prov:
            arr = pointer_get(current, P, default=None)
            prov[P] = list(range(len(arr))) if isinstance(arr, list) else None
        return prov[P]

    removals: list[tuple[str, int, dict]] = []
    for op in gold_patch:
        if op.get("paths"):
            continue
        m = _ARR_TAIL.match(op.get("path", ""))
        if not m:
            continue
        P, seg = m.group(1), m.group(2)
        p = ensure(P)
        if p is None:
            continue
        if op.get("op") == "remove":
            idx = len(p) - 1 if seg == "-" else int(seg)
            if 0 <= idx < len(p):
                removals.append((P, p[idx], op))
                p.pop(idx)
        elif op.get("op") == "add":
            idx = len(p) if seg == "-" else int(seg)
            p.insert(idx, None)
    return removals


def _leaf_satisfied(op: dict, mval: Any, M: dict | None = None,
                    prov: dict | None = None) -> bool:
    """Identity-space leaf satisfaction (mirrors evaluator.op_satisfied for the
    value dimension; structural array ops are handled separately).

    `ref_equal` is a referential matcher: the leaf is satisfied iff its model
    value equals the model value at `ref_path` (another leaf), regardless of the
    literal token. Used for free identifiers — e.g. an inserted task's `id` is
    arbitrary, but the successor's `predecessorId` must point at whatever id the
    model actually chose. Needs M (identity map) and prov to resolve ref_path."""
    from benchmark.evaluator import _satisfies_constraint  # lazy: avoid cycle

    matcher = op.get("matcher") or "strict"
    if matcher == "any":
        return mval is not _SENT
    if matcher in ("date_range", "regex_match"):
        return False if mval is _SENT else _satisfies_constraint(mval, op)
    if matcher == "ref_equal":
        ref = op.get("ref_path")
        if mval is _SENT or M is None or not ref:
            return False
        rv = M.get(_to_identity(ref, prov or {}), _SENT)
        return rv is not _SENT and _norm_eq(mval, rv)
    if op.get("op") == "remove":
        return mval is _SENT  # object-key removal: leaf must be gone
    return _norm_eq(mval, op.get("value"))


def score_strict(current: Any, gold_patch: list[dict], model_patch: list[dict],
                 value_constraints: list[dict] | None = None,
                 collect_detail: bool = False) -> dict:
    """Identity-anchored drop-in replacement for evaluator.score().

    With ``collect_detail=True`` the result also carries ``detail`` — per-gold-path
    {path, kind, touched, satisfied} plus the over-edited paths — for debugging
    failures. This is purely additive and does not affect any metric."""
    from benchmark.evaluator import _attach_legacy_constraints  # lazy: avoid cycle

    gold_patch = _attach_legacy_constraints(gold_patch, value_constraints or [])

    # canonical gold artifact (first non-optional alt per identity path), for
    # gold_changed reporting only; matchers/optionals are scored via the ops.
    canonical = [o for o in gold_patch
                 if not o.get("paths") and not o.get("optional")]
    try:
        gold_after = apply_patch(current, canonical)
    except Exception:  # noqa: BLE001
        gold_after = apply_patch(current, [o for o in gold_patch if not o.get("paths")])
    try:
        model_after = apply_patch(current, model_patch)
        apply_ok = True
    except Exception:  # noqa: BLE001
        model_after, apply_ok = current, False

    g_prov, _ = track_prov(current, gold_patch)
    m_prov, _ = track_prov(current, model_patch)
    # A model may rewrite a whole array in one `replace /array = [...]` op instead
    # of granular add/remove ops; track_prov can't see element moves there, so
    # recover that array's provenance by content-aligning the final array to the
    # original. Keeps scoring robust to patch granularity (result, not op style).
    for op in model_patch:
        if op.get("op") == "replace" and isinstance(op.get("value"), list):
            P = op.get("path", "")
            orig = pointer_get(current, P, default=None)
            final = pointer_get(model_after, P, default=None)
            # Only re-derive provenance when the array LENGTH changed (an element
            # was inserted/removed). Same-length wholesale replaces are value
            # edits or in-place swaps, where positional identity is already
            # correct and re-aligning would only risk mismatches.
            if (isinstance(orig, list) and isinstance(final, list)
                    and len(orig) != len(final)):
                m_prov[P] = _align_prov(orig, final)
    arr_paths = set(g_prov) | set(m_prov)

    # Exact-final-array reconciliation. When the model produced the SAME final
    # array as the gold for a path, the model is correct there by definition —
    # there is no difference left to attribute to over/miss/value, so element
    # identity is irrelevant. Content-alignment cannot recover a stable identity
    # for a WHOLESALE replace whose elements are all-new (no textual overlap with
    # the originals), which would otherwise spuriously penalize a perfectly
    # correct model. Adopt the gold's element provenance for such paths so the
    # identity map aligns exactly. Only fires on exact list equality, so partial
    # (genuinely-wrong) arrays keep their normal scoring.
    for P in (arr_paths if _EXACT_ARRAY_RECONCILE else ()):
        if g_prov.get(P) is None:
            continue
        g_final = pointer_get(gold_after, P, default=None)
        m_final = pointer_get(model_after, P, default=None)
        if isinstance(g_final, list) and g_final == m_final:
            m_prov[P] = list(g_prov[P])

    def _fill(prov):
        for P in arr_paths:
            if prov.get(P) is None:
                arr = pointer_get(current, P, default=None)
                prov[P] = list(range(len(arr))) if isinstance(arr, list) else None
        return prov

    g_prov, m_prov = _fill(dict(g_prov)), _fill(dict(m_prov))
    O = _ident_map(current, _fill({}))
    G = _ident_map(gold_after, g_prov)
    M = _ident_map(model_after, m_prov)
    model_keeps = {P: {i for i in (m_prov.get(P) or []) if i is not None}
                   for P in arr_paths}

    # Map gold ops to identity paths with the SAME (filled, final) provenance
    # used for O/G/M, so tagging is consistent. Array-element removals are
    # resolved separately (origin index at pre-removal position); array-element
    # adds expand the inserted value into per-leaf required ops at #NEW.
    removals = _gold_removals(current, gold_patch)
    leaf_ops: list[tuple[str, dict]] = []
    constraint_ops: list[dict] = []
    for op in gold_patch:
        if op.get("paths"):
            constraint_ops.append(op)
            continue
        path = op.get("path", "")
        m = _ARR_TAIL.match(path)
        if op.get("op") == "remove" and m:
            continue  # array-element removal -> handled via `removals`
        # A dict-key removal of a whole OBJECT (remove /tasks/<name> where the
        # value is a dict) has no leaf at the op path itself: neither O nor M
        # holds the unit path, so `touched` stays False and the op lands in
        # `fn` — a phantom miss even for a perfect model (gold-vs-gold failed
        # on such cases). Expand it into per-leaf remove ops over the CURRENT
        # value's leaves, symmetric with the object-valued add/replace
        # expansion below; each leaf then scores touched (present -> gone) and
        # satisfied (absent in M).
        if op.get("op") == "remove" and not m:
            cur_v = pointer_get(current, path, default=_SENT)
            if isinstance(cur_v, dict) and cur_v:
                for sub, _v in _leaves(cur_v):
                    leaf_ops.append((_to_identity(path + sub, g_prov), {
                        "op": "remove",
                        "kind": op.get("kind", "direct"),
                        "optional": op.get("optional", False),
                    }))
                continue
        if op.get("op") == "add" and m:
            base = f"{m.group(1)}#NEW"
            for sub, v in _leaves(op.get("value")):
                lop = {
                    "op": "add", "value": v, "kind": op.get("kind", "direct"),
                    "optional": op.get("optional", False),
                }
                # a sub-field whose value is the sentinel "<any>" is a free field
                # (e.g. an inserted element's arbitrary id): accept any value.
                if v == _ANY:
                    lop["matcher"] = "any"
                leaf_ops.append((base + sub, lop))
            continue
        # A gold op may set a whole OBJECT value (e.g. replace /contact = {name,
        # email}, or add /tasks/new_task = {...} under a dict). The identity map
        # is leaf-level, so a single object-valued op can never match — it scores
        # a phantom miss while the model's real per-leaf edits count as
        # over-edits (gold-vs-gold itself failed on such cases). Expand object
        # values into per-leaf ops keyed by their object sub-path; a model that
        # edits those leaves (or writes the whole object) then matches correctly.
        # Sub-leaves whose gold value already equals the current value are
        # skipped (a no-op leaf would otherwise be miscounted as a miss; for an
        # `add` the path does not exist in `current`, so nothing is skipped).
        # Arrays are NOT auto-expanded here: element identity needs provenance,
        # so array changes must be annotated as granular add/remove/replace ops
        # (a whole-array gold replace is left as-is and will surface its own
        # scoring artifact, by design).
        if (op.get("op") in ("replace", "add") and not m and not op.get("matcher")
                and isinstance(op.get("value"), dict)):
            for sub, v in _leaves(op.get("value")):
                if _norm_eq(pointer_get(current, path + sub, default=_SENT), v):
                    continue  # no-op sub-leaf
                leaf_ops.append((_to_identity(path + sub, g_prov), {
                    "op": "replace", "value": v,
                    "kind": op.get("kind", "direct"),
                    "optional": op.get("optional", False),
                }))
            continue
        leaf_ops.append((_to_identity(path, g_prov), op))

    # group leaf ops by identity path -> OR alternatives
    by_ident: dict[str, list[dict]] = {}
    for ip, op in leaf_ops:
        by_ident.setdefault(ip, []).append(op)
    path_optional = {ip: all(o.get("optional") for o in alts)
                     for ip, alts in by_ident.items()}

    tp = fp = fn = 0
    val_ok = val_total = 0
    flex_opt_touched = flex_opt_ok = 0
    direct_outcomes: list[bool] = []
    cascade_outcomes: list[bool] = []
    detail: list[dict] = []
    over_paths: list[str] = []

    for ip, alts in by_ident.items():
        mval = M.get(ip, _SENT)
        oval = O.get(ip, _SENT)
        satisfied = any(_leaf_satisfied(a, mval, M, g_prov) for a in alts)
        touched = oval != mval
        any_direct = any(a.get("kind") == "direct" for a in alts)
        if collect_detail and not path_optional[ip]:
            detail.append({"path": ip, "kind": "direct" if any_direct else "cascade",
                           "touched": touched, "satisfied": satisfied})
        if path_optional[ip]:
            if touched:
                flex_opt_touched += 1
                if satisfied:
                    flex_opt_ok += 1
        else:
            val_total += 1
            if touched:
                tp += 1
                if satisfied:
                    val_ok += 1
            else:
                fn += 1
            (direct_outcomes if any_direct else cascade_outcomes).append(satisfied)

    # array-element removals: a required outcome (removed == satisfied)
    for P, i, op in removals:
        satisfied = i not in model_keeps.get(P, set())
        any_direct = op.get("kind") == "direct"
        if op.get("optional"):
            if not satisfied:
                flex_opt_touched += 1  # kept-when-should-go: out-of-spec touch
        else:
            val_total += 1
            if satisfied:
                tp += 1
                val_ok += 1
            else:
                fn += 1
            (direct_outcomes if any_direct else cascade_outcomes).append(satisfied)

    # multi-path constraint ops (e.g. sum_with) — map paths to identity, eval on M
    constraint_idents: set[str] = set()
    for cop in constraint_ops:
        idents = [_to_identity(p, g_prov) for p in (cop.get("paths") or [])]
        constraint_idents.update(idents)
        any_touched = any(O.get(ip, _SENT) != M.get(ip, _SENT) for ip in idents)
        c_satisfied = _constraint_satisfied(cop, idents, M)
        any_direct = cop.get("kind") == "direct"
        val_total += 1
        if any_touched:
            tp += 1
            if c_satisfied:
                val_ok += 1
        elif c_satisfied:
            val_ok += 1
        else:
            fn += 1
        (direct_outcomes if any_direct else cascade_outcomes).append(c_satisfied)

    # over_edit: an identity leaf the gold never prescribes whose model value
    # differs from the original — a spurious edit, a dangling-after-remove
    # leftover, a brand-new leaf, or a spurious removal. Scan O ∪ M so a leaf the
    # model wrongly DELETED (absent in M) is caught too. A leaf the gold itself
    # removed AND the model also removed is correct (not over_edit). Identity
    # space has no array-shift byproducts, so no byproduct subtraction is needed.
    gold_idents = set(by_ident) | constraint_idents
    for ip in set(O) | set(M):
        if ip in gold_idents:
            continue
        oval = O.get(ip, _SENT)
        mval = M.get(ip, _SENT)
        if _norm_eq(oval, mval):
            continue
        if ip not in G and mval is _SENT:
            continue  # gold removed it and the model removed it too -> correct
        fp += 1
        if collect_detail:
            over_paths.append(ip)

    precision = tp / (tp + fp) if (tp + fp) else 1.0
    recall = tp / (tp + fn) if (tp + fn) else 1.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
    n_value = val_total + flex_opt_touched
    value_recall = (val_ok + flex_opt_ok) / n_value if n_value else 1.0
    direct_recall = (sum(direct_outcomes) / len(direct_outcomes)
                     if direct_outcomes else 1.0)
    cascade_recall = (sum(cascade_outcomes) / len(cascade_outcomes)
                      if cascade_outcomes else 1.0)
    gold_changed = {ip for ip in set(O) | set(G) if O.get(ip, _SENT) != G.get(ip, _SENT)}
    model_changed = {ip for ip in set(O) | set(M) if O.get(ip, _SENT) != M.get(ip, _SENT)}
    n_flex = sum(1 for alts in by_ident.values()
                 if any(o.get("matcher") for o in alts))

    result = {
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
    if collect_detail:
        result["detail"] = {"paths": detail, "over_paths": over_paths}
    return result


def _constraint_satisfied(cop: dict, idents: list[str], M: dict) -> bool:
    import math
    matcher = cop.get("matcher")
    if matcher == "sum_with":
        expected = cop.get("expected_sum")
        if expected is None:
            return False
        try:
            total = 0.0
            for ip in idents:
                v = M.get(ip, _SENT)
                if v is _SENT:
                    return False
                total += float(v)
            return math.isclose(total, float(expected), abs_tol=1e-6)
        except (TypeError, ValueError):
            return False
    return False
