"""Failure analysis figure + numbers for the final/ 3-factor test set.

Element-level error composition: every method's final patch is re-scored with
the strict scorer, and we count the disjoint leaf-level error events it leaves,
summed over all cases (six models x five seeds). The three categories are

  miss        : a required (gold) leaf the model did not change     (= scorer 'miss')
  over edit   : a leaf the model changed that the gold leaves alone (= scorer 'over_edit')
  wrong value : a required leaf the model changed to the wrong value (= req_total - req_ok - miss)

These are disjoint leaf sets, so a case contributes to several categories
according to its actual errors (no per-case priority collapse).

Methods match the success-rate grid (5-call budget): J+H, REFLECT (serial review,
2 rounds), OR/AND/MAJ and MED (from the 5 parallel samples), SELECT (LLM judge).
Run: docker exec revpropbench-runner bash -c "cd /work && python scripts/plot_failure_analysis.py"
"""
import json, glob, os
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.strict_score import score_strict
from benchmark.jsonptr import apply_patch
from methods.subagent_consensus import merge_consensus, build_consensus_patch, medoid_index

ann = {}
for f in glob.glob("benchmark/data/annotated/*.json"):
    a = json.load(open(f)); ann[a["id"]] = a
SEEDS = [0, 42, 84, 126, 168]

# Run-file resolver: prefer the original campaign run dirs (the layout shipped
# in results/paper_runs/); fall back to the unified layout that a default
# `make bench-eval` writes (results/bench/final/seed<s>-<MODEL>, where MODEL
# may be either the served-name suffix or the make-level model name).
_ALT_SUFFIX = {"qwen9b-llm": "qwen3.5-9b", "qwen27b-llm": "qwen3.5-27b",
               "qwen122b-llm": "qwen3.5-122b-a10b"}
def runf(tpl, s, fname):
    p = f"{tpl.format(s=s)}/{fname}"
    if os.path.exists(p):
        return p
    suf = tpl.split("seed{s}-", 1)[1]
    for m in (suf, _ALT_SUFFIX.get(suf, suf)):
        q = f"results/bench/final/seed{s}-{m}/{fname}"
        if os.path.exists(q):
            return q
    return p


FULL = lambda p: dict(sub=p, scale=p, judge=p)
MODELS = [
 ("gpt-oss-20b", dict(sub="results/bench/final/testclean-sub-seed{s}-gpt-oss-20b",
                      scale="results/bench/final/testclean-scale5-seed{s}-gpt-oss-20b",
                      judge="results/bench/final/testclean-judge-gptoss-seed{s}-gpt-oss-20b",
                      judgek="results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b")),
 ("gpt-oss-120b", {**FULL("results/bench/final/testclean-full120-seed{s}-gpt-oss-120b"),
                   "judgek": "results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b"}),
 ("gpt-5.4-mini", {**FULL("results/bench/final/testclean-fullgpt54mini-med-seed{s}-gpt-5.4-mini"),
                   "judgek": "results/bench/final/testclean-judgekgpt54mini-med-seed{s}-gpt-5.4-mini"}),
 ("Qwen3.5-9B", dict(sub="results/bench/final/testclean-sub-q9b-seed{s}-qwen9b-llm",
                     scale="results/bench/final/testclean-scale5-q9b-seed{s}-qwen9b-llm",
                     judge="results/bench/final/testclean-judge-q9b-seed{s}-qwen9b-llm",
                     judgek="results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm")),
 ("Qwen3.5-27B", {**FULL("results/bench/final/testclean-full27b-seed{s}-qwen27b-llm"),
                  "judgek": "results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm"}),
 ("Qwen3.5-122B", {**FULL("results/bench/final/testclean-full122b-seed{s}-qwen122b-llm"),
                   "judgek": "results/bench/final/testclean-judgek-122b-seed{s}-qwen122b-llm"}),
]
METHODS = ["J+H", "REFLECT", "OR", "MAJ", "AND", "MED", "SELECT"]
TYPES = ["miss", "over edit", "wrong value"]
# Muted red / blue / green (softer than the box-plot primaries) for a cleaner,
# modern look, plus a fine hatch per type so the categories are distinguishable
# without color too.
TCOL = {"miss": "#e15759", "over edit": "#4e79a7", "wrong value": "#59a14f"}
THATCH = {"miss": "//", "over edit": "xx", "wrong value": ".."}

def leaf_errors(m):
    """Disjoint leaf-level error counts from a strict-scorer result.

    The three sets are disjoint by construction: ``miss`` is a required leaf left
    untouched, ``over edit`` is a non-gold leaf the model changed, and ``wrong
    value`` is a required leaf the model touched but set to the wrong value
    (touched yet unsatisfied in the per-leaf detail)."""
    wrong = sum(1 for p in m["detail"]["paths"] if p["touched"] and not p["satisfied"])
    return {"miss": m["miss"],
            "over edit": m["over_edit"],
            "wrong value": wrong}

def scored_for(cfg, method, s):
    """Re-score every method's final patch with the strict scorer (uniform leaf accounting)."""
    def jget(f):
        return json.load(open(f)) if glob.os.path.exists(f) else []
    def sc(cid, patch):
        a = ann[cid]
        return score_strict(a["current_json"], a["expected_patch"], patch,
                            a.get("value_constraints"), collect_detail=True)
    out = []
    if method == "J+H":
        for c in jget(runf(cfg["sub"], s, "per_case_sub_hist_json.json")):
            out.append(sc(c["id"], c.get("model_patch") or []))
    elif method == "SELECT":   # judge4 = 4 samples + 1 selection = 5 calls
        for c in jget(runf(cfg["judgek"], s, "per_case_sub_hist_json_judge4.json")):
            out.append(sc(c["id"], c.get("model_patch") or []))
    elif method == "REFLECT":
        for c in jget(runf(cfg["scale"], s, "per_case_sub_hist_json_rev4.json")):
            stg = c.get("review_stages"); p = stg[4] if stg else (c.get("model_patch") or [])   # x4 = 5 calls
            out.append(sc(c["id"], p))
    else:  # OR / AND / MAJ / MED from the 5 cons samples
        for c in jget(runf(cfg["scale"], s, "per_case_sub_hist_json_cons5_and.json")):
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
            out.append(sc(cid, p))
    return out

# counts[model][method][type] = total leaf-level errors, pooled over seeds
counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
for mn, cfg in MODELS:
    for meth in METHODS:
        for s in SEEDS:
            for m in scored_for(cfg, meth, s):
                for cat, v in leaf_errors(m).items():
                    counts[mn][meth][cat] += v

# ---- figure: erroneous-element composition by method (pooled over six models) ----
plt.rcParams["hatch.linewidth"] = 0.5   # finer, subtler hatch
fig, ax = plt.subplots(figsize=(6.6, 4.2))
bot = [0] * len(METHODS)
for t in TYPES:
    vals = [sum(counts[m][meth][t] for m, _ in MODELS) for meth in METHODS]
    ax.bar(range(len(METHODS)), vals, bottom=bot, color=TCOL[t], label=t,
           hatch=THATCH[t], edgecolor="white", linewidth=0.7)
    bot = [b + v for b, v in zip(bot, vals)]
ax.set_xticks(range(len(METHODS))); ax.set_xticklabels(METHODS, rotation=25, ha="right", fontsize=13)
ax.set_ylabel("# erroneous elements", fontsize=16)
ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
ax.tick_params(axis="y", labelsize=13)
ax.legend(loc="upper left", frameon=True, fontsize=13)
fig.tight_layout()
fig.savefig("results/figures/failure_analysis.png", dpi=300, bbox_inches="tight")
fig.savefig("results/figures/failure_analysis.pdf", bbox_inches="tight")
print("saved results/figures/failure_analysis.{png,pdf}")

# ---- printed numbers (element-level leaf errors) ----
mods = [m for m, _ in MODELS]
print("\n== (a) by model, J+H erroneous-element mix ==")
for m in mods:
    d = counts[m]["J+H"]; tot = sum(d.values())
    print(f"{m:13} total={tot:5d}  " + "  ".join(f"{t}:{d[t]}({100*d[t]/max(tot,1):.0f}%)" for t in TYPES))
print("\n== (b) by method, pooled over 6 models ==")
for meth in METHODS:
    d = {t: sum(counts[m][meth][t] for m, _ in MODELS) for t in TYPES}; tot = sum(d.values())
    print(f"{meth:8} total={tot:5d}  " + "  ".join(f"{t}:{d[t]}({100*d[t]/max(tot,1):.0f}%)" for t in TYPES))
