"""Group-disjoint, multi-factor stratified dev/test split for the benchmark.

The 150 annotated cases are really 50 *scenarios* (one per ``(domain, theme)``)
realised at three document sizes (``target_leaves`` 10/50/100). The three
members of a scenario share the same edit and cascade pattern — only the JSON
size differs — so splitting at the case level would leak a scenario's small
member into dev and its large member into test, making dev and test effectively
overlap. We therefore split at the *scenario* level: every scenario goes wholly
to dev or wholly to test, and dev/test never share a scenario.

A purely random scenario draw can still come out lopsided on the axes we care
about. So instead of plain random we balance the split across three per-scenario
factors so dev and the whole set have matching marginal distributions:

  * ``domain``                — proportional, treated as a (near-)hard constraint;
  * ``cascade_size``          — small/medium/large by tercile of the affected-leaf
                                count at L100 (few affected == many *protected*);
  * ``category``              — the cascade taxonomy authored on the Skeleton
                                (looked up by (domain, theme)).

Selection is a seeded balanced search: domain counts for dev are fixed by
largest-remainder proportional allocation; within that, many random candidate
scenario sets are scored by how well dev's marginals on the remaining factors
match the global proportions, and the lowest-imbalance candidate is kept. The
search optimises FEATURE balance only (never model performance) and is robust to
the seed (it converges to the same low imbalance regardless), so the split is not
sensitive to seed choice.

Usage:
  python -m benchmark.split --annotated benchmark/data/annotated \
      --dev-size 30 --seed 42 \
      --out-split benchmark/data/split_dev30_test120.json \
      --out-dev benchmark/data/annotated_dev --out-test benchmark/data/annotated_test
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from typing import Any

from benchmark.jsonptr import apply_patch, list_leaf_paths, pointer_get
from benchmark.skeleton import SKELETONS_BY_DOMAIN


# Cascade-mechanism category is authored on the Skeleton (source of truth) and
# looked up here by (domain, theme); see benchmark/skeleton.py. A case whose
# scenario is not in the registry (or any --categories override) falls back to
# "uncategorized" rather than a guess.
_CATEGORY_BY_KEY: dict[tuple[str, str], str] = {
    (sk.domain, sk.theme): (sk.category or "uncategorized")
    for skel in SKELETONS_BY_DOMAIN.values() for sk in skel
}


def lookup_category(domain: str, theme: str) -> str:
    return _CATEGORY_BY_KEY.get((domain, theme), "uncategorized")


# --- scenario feature extraction --------------------------------------------
def _domain_of(case_id: str) -> str:
    return re.sub(r"_L\d.*$", "", case_id)


def _affected_leaf_count(case: dict[str, Any]) -> int:
    cur = case["current_json"]
    gold = [o for o in case["expected_patch"]
            if o.get("path") and not o.get("matcher") and not o.get("optional")]
    try:
        after = apply_patch(cur, gold)
    except Exception:  # noqa: BLE001 — fall back to all gold ops
        after = apply_patch(cur, [o for o in case["expected_patch"] if o.get("path")])
    lc = {p: pointer_get(cur, p) for p in list_leaf_paths(cur)}
    la = {p: pointer_get(after, p) for p in list_leaf_paths(after)}
    return sum(1 for p in la if la.get(p) != lc.get(p))


def load_scenarios(annotated_dir: str) -> dict[tuple[str, str], dict[str, Any]]:
    """Group cases into scenarios keyed by (domain, theme); return per-scenario
    metadata + the member case ids by target_leaves."""
    groups: dict[tuple[str, str], dict[int, dict[str, Any]]] = defaultdict(dict)
    paths: dict[tuple[str, str], dict[int, str]] = defaultdict(dict)
    for f in sorted(glob.glob(os.path.join(annotated_dir, "*.json"))):
        d = json.load(open(f))
        key = (d["domain"], d.get("theme"))
        groups[key][d["target_leaves"]] = d
        paths[key][d["target_leaves"]] = os.path.basename(f)[:-5]

    scenarios: dict[tuple[str, str], dict[str, Any]] = {}
    for key, members in groups.items():
        dom, theme = key
        rep = members.get(max(members)) or next(iter(members.values()))  # largest tl
        scenarios[key] = {
            "domain": dom,
            "theme": theme,
            "case_ids": [paths[key][tl] for tl in sorted(paths[key])],
            "n_members": len(members),
            "affected_at_max": _affected_leaf_count(rep),
            "structural": any(o.get("op") in ("add", "remove")
                              for m in members.values() for o in m["expected_patch"]),
            "requires_master_data": any(bool(m.get("requires_master_data"))
                                        for m in members.values()),
            "category": lookup_category(dom, theme),
        }
    return scenarios


def _size_buckets(scenarios: dict[tuple[str, str], dict[str, Any]]) -> None:
    """Annotate each scenario with a small/medium/large cascade_size bucket by
    tercile of affected_at_max (mutates in place)."""
    vals = sorted(s["affected_at_max"] for s in scenarios.values())
    n = len(vals)
    t1, t2 = vals[n // 3], vals[2 * n // 3]
    for s in scenarios.values():
        a = s["affected_at_max"]
        s["cascade_size"] = "small" if a <= t1 else ("medium" if a <= t2 else "large")


# --- balanced search --------------------------------------------------------
# Soft balance factors (domain is balanced separately as a near-hard quota in
# _domain_dev_counts), so the split balances three factors in total:
# domain + cascade_size (propagation size) + category (propagation pattern).
_BALANCE_FACTORS = ("cascade_size", "category")


def _domain_dev_counts(scenarios: dict[tuple[str, str], dict[str, Any]],
                       dev_groups: int) -> dict[str, int]:
    """Largest-remainder proportional allocation of dev scenarios per domain."""
    by_dom: dict[str, int] = Counter(s["domain"] for s in scenarios.values())
    total = sum(by_dom.values())
    quota = {d: c * dev_groups / total for d, c in by_dom.items()}
    base = {d: int(q) for d, q in quota.items()}
    rem = dev_groups - sum(base.values())
    for d in sorted(by_dom, key=lambda d: (-(quota[d] - base[d]), d))[:rem]:
        base[d] += 1
    return base


def _imbalance(dev: list[dict[str, Any]], scenarios: list[dict[str, Any]]) -> float:
    """Sum over balance factors of |dev_prop(level) - global_prop(level)|."""
    score = 0.0
    nd, ng = len(dev), len(scenarios)
    for factor in _BALANCE_FACTORS:
        gc = Counter(s[factor] for s in scenarios)
        dc = Counter(s[factor] for s in dev)
        for level, gcount in gc.items():
            score += abs(dc.get(level, 0) / nd - gcount / ng)
    return score


def balanced_split(scenarios: dict[tuple[str, str], dict[str, Any]],
                   dev_groups: int, seed: int, trials: int) -> set[tuple[str, str]]:
    items = list(scenarios.items())
    all_meta = [m for _, m in items]
    by_dom: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for key, m in items:
        by_dom[m["domain"]].append(key)
    dev_counts = _domain_dev_counts(scenarios, dev_groups)

    best_keys: set[tuple[str, str]] | None = None
    best_score = float("inf")
    for t in range(trials):
        rng = random.Random((seed << 20) ^ t)
        chosen: list[tuple[str, str]] = []
        for dom, keys in by_dom.items():
            pool = list(keys)
            rng.shuffle(pool)
            chosen.extend(pool[:dev_counts[dom]])
        dev_meta = [scenarios[k] for k in chosen]
        score = _imbalance(dev_meta, all_meta)
        if score < best_score:
            best_score, best_keys = score, set(chosen)
    assert best_keys is not None
    return best_keys


# --- report + IO ------------------------------------------------------------
def _factor_report(dev: list[dict[str, Any]], test: list[dict[str, Any]]) -> dict[str, Any]:
    rep: dict[str, Any] = {}
    for factor in ("domain",) + _BALANCE_FACTORS:
        rep[factor] = {
            "dev": dict(sorted(Counter(s[factor] for s in dev).items(), key=lambda x: str(x[0]))),
            "test": dict(sorted(Counter(s[factor] for s in test).items(), key=lambda x: str(x[0]))),
        }
    return rep


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--annotated", default="benchmark/data/annotated")
    ap.add_argument("--dev-size", type=int, default=30, help="dev size in CASES (must be a multiple of the scenario size, i.e. 3)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--trials", type=int, default=20000, help="candidate draws for the balanced search")
    ap.add_argument("--categories", help="optional JSON mapping 'domain|theme' -> category (overrides inference)")
    ap.add_argument("--out-split", default="benchmark/data/split_dev30_test120.json")
    ap.add_argument("--out-dev", help="if set, copy dev case files into this dir")
    ap.add_argument("--out-test", help="if set, copy test case files into this dir")
    ap.add_argument("--apply", action="store_true",
                    help="do not re-run the balanced search; read the existing "
                         "--out-split JSON and just copy its dev/test case files "
                         "into --out-dev/--out-test")
    args = ap.parse_args()

    if args.apply:
        split = json.load(open(args.out_split))
        for ids, out in ((split["dev"], args.out_dev), (split["test"], args.out_test)):
            if not out:
                continue
            if os.path.exists(out):
                shutil.rmtree(out)
            os.makedirs(out)
            for cid in ids:
                shutil.copy(os.path.join(args.annotated, f"{cid}.json"), out)
            print(f"copied {len(ids)} cases -> {out}")
        return

    scenarios = load_scenarios(args.annotated)
    if args.categories and os.path.exists(args.categories):
        overrides = json.load(open(args.categories))
        for (dom, theme), s in scenarios.items():
            s["category"] = overrides.get(f"{dom}|{theme}", s["category"])
    _size_buckets(scenarios)

    sizes = {s["n_members"] for s in scenarios.values()}
    if len(sizes) != 1:
        raise SystemExit(f"scenarios have non-uniform member counts: {sorted(sizes)}")
    group_size = sizes.pop()
    if args.dev_size % group_size:
        raise SystemExit(f"--dev-size {args.dev_size} not a multiple of scenario size {group_size}")
    dev_groups = args.dev_size // group_size

    dev_keys = balanced_split(scenarios, dev_groups, args.seed, args.trials)

    dev_meta = [scenarios[k] for k in scenarios if k in dev_keys]
    test_meta = [scenarios[k] for k in scenarios if k not in dev_keys]
    dev_ids = sorted(cid for m in dev_meta for cid in m["case_ids"])
    test_ids = sorted(cid for m in test_meta for cid in m["case_ids"])
    assert set(dev_ids).isdisjoint(test_ids)
    # group-disjoint sanity: no (domain,theme) shared
    assert {(m["domain"], m["theme"]) for m in dev_meta}.isdisjoint(
        {(m["domain"], m["theme"]) for m in test_meta})

    report = _factor_report(dev_meta, test_meta)
    # Per-scenario manifest: every scenario with its features (incl. category) and
    # split assignment, so the category labels are auditable, not just aggregated.
    scenario_records = sorted(
        ({"domain": m["domain"], "theme": m["theme"],
          "split": "dev" if (m["domain"], m["theme"]) in dev_keys else "test",
          "category": m["category"], "cascade_size": m["cascade_size"],
          "structural": m["structural"], "requires_master_data": m["requires_master_data"],
          "affected_at_max": m["affected_at_max"], "case_ids": m["case_ids"]}
         for m in scenarios.values()),
        key=lambda r: (r["domain"], r["theme"]))
    split = {
        "method": "balanced",
        "seed": args.seed,
        "dev_size": len(dev_ids),
        "test_size": len(test_ids),
        "n_dev_scenarios": len(dev_meta),
        "imbalance": round(_imbalance(dev_meta, list(scenarios.values())), 4),
        "dev": dev_ids,
        "test": test_ids,
        "factor_distribution": report,
        "scenarios": scenario_records,
    }
    os.makedirs(os.path.dirname(args.out_split) or ".", exist_ok=True)
    json.dump(split, open(args.out_split, "w"), indent=2, ensure_ascii=False)

    # Editable category map (domain|theme -> category). Re-feed via --categories
    # to override the heuristic inference on the next run.
    cats_path = os.path.join(os.path.dirname(args.out_split) or ".", "scenario_categories.json")
    if not (args.categories and os.path.abspath(args.categories) == os.path.abspath(cats_path)):
        cat_map = {f"{m['domain']}|{m['theme']}": m["category"] for m in scenarios.values()}
        json.dump(dict(sorted(cat_map.items())), open(cats_path, "w"), indent=2, ensure_ascii=False)
        print(f"wrote editable category map -> {cats_path}")

    print(f"balanced seed={args.seed} -> dev={len(dev_ids)} test={len(test_ids)} "
          f"({len(dev_meta)} dev scenarios)  imbalance={split['imbalance']}")
    for factor, rr in report.items():
        print(f"  [{factor}]")
        print(f"    dev : {rr['dev']}")
        print(f"    test: {rr['test']}")
    print(f"wrote split -> {args.out_split}")

    for ids, out in ((dev_ids, args.out_dev), (test_ids, args.out_test)):
        if not out:
            continue
        if os.path.exists(out):
            shutil.rmtree(out)
        os.makedirs(out)
        for cid in ids:
            shutil.copy(os.path.join(args.annotated, f"{cid}.json"), out)
        print(f"copied {len(ids)} cases -> {out}")


if __name__ == "__main__":
    main()
