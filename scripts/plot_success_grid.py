"""2x3 grid of per-model bar charts: success rate of 9 methods on TEST.
Top row: gpt-oss-20b / gpt-oss-120b / gpt-5.4-mini.
Bottom row: Qwen3.5-9B / 27B / 122B. Bars = seed-mean, error = seed std.
Run inside the runner container from the repo root:
  docker exec revpropbench-runner bash -c "cd /work && python scripts/plot_success_grid.py"
"""
import json, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark.strict_score import score_strict
from benchmark.jsonptr import apply_patch
from methods.subagent_consensus import merge_consensus, build_consensus_patch, medoid_index

ann = {}
# Full annotated set (superset of every split) so re-scoring resolves any cid,
# whether a model's run dir is on the old or new (final/) test split.
for f in glob.glob("benchmark/data/annotated/*.json"):
    a = json.load(open(f)); ann[a["id"]] = a
def succ(r):
    vr = r.get("value_recall"); vr1 = 1.0 if vr is None else vr
    return r["over_edit"] == 0 and r["miss"] == 0 and vr1 >= 1.0

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
METHODS = ["J", "H", "J+H", "REFLECT", "OR", "AND", "MAJ", "MED", "SELECT"]
COLORS = ["#bcbd22", "#17becf", "#1f77b4", "#9edae5", "#2ca02c", "#7f7f7f",
          "#d62728", "#9467bd", "#ff7f0e"]

def model_stats(cfg):
    per = {m: {} for m in METHODS}   # method -> seed -> [bool]
    for s in SEEDS:
        for row, meth in (("J", "sub_json"), ("H", "sub_hist"), ("J+H", "sub_hist_json"),
                          ("SELECT", "sub_hist_json_judge4")):   # judge4 = 4 samples + 1 selection = 5 calls
            f = runf(cfg["judgek" if row == "SELECT" else "sub"], s, f"per_case_{meth}.json")
            if not os.path.exists(f): continue
            for c in json.load(open(f)):
                a = ann[c["id"]]
                r = c.get("metrics") or score_strict(a["current_json"], a["expected_patch"],
                                                     c.get("model_patch") or [], a.get("value_constraints"))
                per[row].setdefault(s, []).append(succ(r))
        f = runf(cfg["scale"], s, "per_case_sub_hist_json_rev4.json")
        if os.path.exists(f):
            for c in json.load(open(f)):
                a = ann[c["id"]]; stg = c.get("review_stages")
                p = stg[4] if stg else (c.get("model_patch") or [])   # x4 = base + 4 reviews = 5 calls
                per["REFLECT"].setdefault(s, []).append(succ(score_strict(a["current_json"], a["expected_patch"], p, a.get("value_constraints"))))
        f = runf(cfg["scale"], s, "per_case_sub_hist_json_cons5_and.json")
        if os.path.exists(f):
            for c in json.load(open(f)):
                cid = c["id"]; a = ann[cid]; cur = a["current_json"]
                P = []
                for smp in c["estimator"]["samples"][:5]:
                    if smp["patch"] is None: continue
                    try: P.append((smp["patch"], apply_patch(cur, smp["patch"])))
                    except Exception: pass
                if not P: continue
                pairs = [{"patch": p, "_applied": ap} for p, ap in P]
                applied = [ap for _, ap in P]
                for row, mode in (("OR", "or"), ("AND", "and"), ("MAJ", "maj")):
                    pt = build_consensus_patch(cur, merge_consensus(cur, applied, mode), pairs)
                    per[row].setdefault(s, []).append(succ(score_strict(cur, a["expected_patch"], pt, a.get("value_constraints"))))
                per["MED"].setdefault(s, []).append(succ(score_strict(cur, a["expected_patch"], P[medoid_index(applied)][0], a.get("value_constraints"))))
    out = {}
    for m in METHODS:
        means = [st.mean(v) for v in per[m].values() if v]
        out[m] = (st.mean(means), st.pstdev(means)) if means else (0, 0)
    return out

DUMP = {}   # per-model per-method (mean%, sd%) for the write-up
fig, axes = plt.subplots(2, 3, figsize=(15.5, 7.2), sharey=True)
for ax, (mname, cfg) in zip(axes.flat, MODELS):
    stats = model_stats(cfg)
    DUMP[mname] = {m: {"mean": round(stats[m][0] * 100, 2), "sd": round(stats[m][1] * 100, 2)} for m in METHODS}
    xs = range(len(METHODS))
    vals = [stats[m][0] * 100 for m in METHODS]
    errs = [stats[m][1] * 100 for m in METHODS]
    bars = ax.bar(xs, vals, yerr=errs, capsize=2.5,
                  color=COLORS, edgecolor="white", linewidth=0.6)
    best = max(vals)
    for x, (v, e) in enumerate(zip(vals, errs)):
        ax.annotate(f"{v:.1f}", (x, v + e), textcoords="offset points",
                    xytext=(0, 3), ha="center", fontsize=11,
                    fontweight="bold" if v == best else "normal")
    ax.set_xticks(list(xs))
    ax.set_xticklabels(METHODS, fontsize=13, rotation=38, ha="right")
    ax.set_title(mname, fontsize=16)
    ax.set_ylim(40, 106)
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", labelsize=13)
for ax in axes[:, 0]:
    ax.set_ylabel("completion rate (%)", fontsize=16)
fig.tight_layout()
# Swap each bar rectangle for a top-rounded box. The mutation aspect is
# derived from the axes' rendered geometry so the corner radius stays
# circular regardless of the data ranges (bottom corners fall below y=0
# only by the rounding amount, which is hidden behind the axis line).
fig.canvas.draw()
from matplotlib.patches import FancyBboxPatch, Rectangle
for ax in axes.flat:
    ext = ax.get_window_extent()
    xsp = ax.get_xlim()[1] - ax.get_xlim()[0]
    ysp = ax.get_ylim()[1] - ax.get_ylim()[0]
    aspect = (ysp / xsp) * (ext.width / ext.height)
    rdrop = 0.13 * aspect   # extend below y=0 so the bottom rounding is
    for patch in [q for q in ax.patches if type(q) is Rectangle]:   # clipped
        bb = patch.get_bbox()
        ax.add_patch(FancyBboxPatch(
            (bb.xmin, bb.ymin - rdrop), bb.width, bb.height + rdrop,
            boxstyle="round,pad=0,rounding_size=0.13", mutation_aspect=aspect,
            facecolor=patch.get_facecolor(), edgecolor="white", linewidth=0.6,
            zorder=patch.get_zorder()))
        patch.remove()
fig.savefig("results/figures/success_grid_6models.png", dpi=300)
fig.savefig("results/figures/success_grid_6models.pdf")
json.dump(DUMP, open("results/figures/success_grid_numbers.json", "w"), indent=2)
print("saved results/figures/success_grid_6models.{png,pdf} + success_grid_numbers.json")
