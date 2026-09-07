"""Paired significance tests for the revision methods (5-call budget) on the
final/ campaign: every method pair listed in PAIRS, per model.

Two tests per (model, pair), both respecting that the 5 seeds of a case are
not independent:
  - case-level paired bootstrap: cluster = test case (value = mean success
    over its 5 seeds), 10k resamples -> 95% CI and two-sided p
  - instance-level McNemar (b: A-only success, c: B-only success), exact
    binomial p, for reference

Run: docker exec -w /work revpropbench-runner python scripts/paired_tests.py"""
import json, glob, os, random, statistics as st
from math import comb
from benchmark.strict_score import score_strict
from benchmark.jsonptr import apply_patch
from methods.subagent_consensus import merge_consensus, build_consensus_patch, medoid_index

ann = {}
for f in glob.glob("benchmark/data/annotated/*.json"):
    a = json.load(open(f)); ann[a["id"]] = a
SEEDS = [0, 42, 84, 126, 168]
FULL = lambda p: dict(sub=p, scale=p, judge=p)
MODELS = [
 ("gpt-oss-20b", dict(sub="results/bench/final/testclean-sub-seed{s}-gpt-oss-20b",
                      scale="results/bench/final/testclean-scale5-seed{s}-gpt-oss-20b",
                      ext="results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b")),
 ("gpt-oss-120b", dict(**FULL("results/bench/final/testclean-full120-seed{s}-gpt-oss-120b"),
                       ext="results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b")),
 ("gpt-5.4-mini", dict(**FULL("results/bench/final/testclean-fullgpt54mini-med-seed{s}-gpt-5.4-mini"),
                       ext="results/bench/final/testclean-judgekgpt54mini-med-seed{s}-gpt-5.4-mini")),
 ("Qwen3.5-9B", dict(sub="results/bench/final/testclean-sub-q9b-seed{s}-qwen9b-llm",
                     scale="results/bench/final/testclean-scale5-q9b-seed{s}-qwen9b-llm",
                     ext="results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm")),
 ("Qwen3.5-27B", dict(**FULL("results/bench/final/testclean-full27b-seed{s}-qwen27b-llm"),
                      ext="results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm")),
 ("Qwen3.5-122B", dict(**FULL("results/bench/final/testclean-full122b-seed{s}-qwen122b-llm"),
                       ext="results/bench/final/testclean-judgek-122b-seed{s}-qwen122b-llm")),
]

def succ_of(m):
    vr = m.get("value_recall"); vr = 1.0 if vr is None else vr
    return m["over_edit"] == 0 and m["miss"] == 0 and vr >= 1.0
def sc(cid, patch):
    a = ann[cid]
    return score_strict(a["current_json"], a["expected_patch"], patch or [], a.get("value_constraints"))
def jget(f): return json.load(open(f)) if os.path.exists(f) else []

def outcomes(cfg, method, s):
    out = {}
    if method == "J+H":
        for c in jget(f"{cfg['sub'].format(s=s)}/per_case_sub_hist_json.json"):
            out[c["id"]] = succ_of(c.get("metrics") or sc(c["id"], c.get("model_patch")))
    elif method == "SELECT":
        # paper's 5-call SELECT = judge4 (4 samples + 1 selection call);
        # judge5 is 5 samples + judge = 6 calls and must NOT be used here
        for c in jget(f"{cfg['ext'].format(s=s)}/per_case_sub_hist_json_judge4.json"):
            out[c["id"]] = succ_of(c.get("metrics") or sc(c["id"], c.get("model_patch")))
    elif method == "REFLECT":
        for c in jget(f"{cfg['scale'].format(s=s)}/per_case_sub_hist_json_rev4.json"):
            stg = c.get("review_stages"); p = stg[4] if stg else c.get("model_patch")  # stage k = k reviews -> stg[4] = 5 calls
            out[c["id"]] = succ_of(sc(c["id"], p))
    else:  # MED / OR / MAJ / AND
        for c in jget(f"{cfg['scale'].format(s=s)}/per_case_sub_hist_json_cons5_and.json"):
            cid = c["id"]; cur = ann[cid]["current_json"]; P = []
            for smp in c["estimator"]["samples"][:5]:
                if smp["patch"] is None: continue
                try: P.append((smp["patch"], apply_patch(cur, smp["patch"])))
                except Exception: pass
            if not P: continue
            if method == "MED":
                p = P[medoid_index([ap for _, ap in P])][0]
            else:
                pairs = [{"patch": pp, "_applied": ap} for pp, ap in P]
                p = build_consensus_patch(cur, merge_consensus(cur, [ap for _, ap in P], method.lower()), pairs)
            out[cid] = succ_of(sc(cid, p))
    return out

def mcnemar_p(b, c):
    n = b + c
    if n == 0: return 1.0
    k = min(b, c)
    return min(sum(comb(n, i) for i in range(k + 1)) / 2 ** n * 2, 1.0)

PAIRS = [("SELECT","J+H"),("SELECT","MED"),("SELECT","REFLECT"),("SELECT","OR"),
         ("SELECT","MAJ"),("SELECT","AND"),
         ("MED","J+H"),("MED","REFLECT"),("MED","OR"),("MED","MAJ"),("MED","AND")]
NEEDED = sorted({m for p in PAIRS for m in p})
B = 10000
rng = random.Random(0)
print(f"{'model':13} {'A vs B':18} {'diff(pp)':>9} {'95% CI':>18} {'boot p':>8} {'b/c':>9} {'McNemar p':>10}")
for mn, cfg in MODELS:
    per = {m: {s: outcomes(cfg, m, s) for s in SEEDS} for m in NEEDED}
    for A, Bm in PAIRS:
        cids = sorted(set.intersection(*[set(per[A][s]) & set(per[Bm][s]) for s in SEEDS]))
        diffs = []; b = c = 0
        for cid in cids:
            da = st.mean(per[A][s][cid] for s in SEEDS)
            db = st.mean(per[Bm][s][cid] for s in SEEDS)
            diffs.append(da - db)
            for s in SEEDS:
                pa, pb = per[A][s][cid], per[Bm][s][cid]
                if pa and not pb: b += 1
                elif pb and not pa: c += 1
        n = len(diffs)
        le = ge = 0
        for _ in range(B):
            m = st.mean(diffs[rng.randrange(n)] for _ in range(n))
            if m <= 0: le += 1
            if m >= 0: ge += 1
        p_boot = max(min(2 * min(le, ge) / B, 1.0), 1 / B)
        boot = sorted(st.mean(diffs[rng.randrange(n)] for _ in range(n)) for _ in range(2000))
        lo, hi = boot[int(0.025 * 2000)], boot[int(0.975 * 2000)]
        ps = "<1e-4" if p_boot <= 1 / B else f"{p_boot:.4f}"
        print(f"{mn:13} {A+' vs '+Bm:18} {100*st.mean(diffs):9.2f} [{100*lo:+6.2f},{100*hi:+6.2f}] {ps:>8} {b:4d}/{c:<4d} {mcnemar_p(b,c):10.2e}")
