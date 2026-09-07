"""Cost-vs-completion 2x3 grid for the six campaign models.
Top row: gpt-oss-20b / gpt-oss-120b / gpt-5.4-mini.
Bottom row: Qwen3.5-9B / 27B / 122B.
x = measured tokens per case (input+output), y = completion rate (%).
Curves: serial x0..x4 (x0 = J+H base), OR k=1..5, MAJ k=1,3,5, AND k=1..5;
points: MED at k=5 (same cost as the 5 samples) and SELECT (5 samples +
judge call, usage from the judge runs).
Run inside the runner container from the repo root:
  docker exec revpropbench-runner bash -c "cd /work && python scripts/plot_cost_grid.py"
"""
import json, glob, os, statistics as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
from benchmark.strict_score import score_strict
from benchmark.jsonptr import apply_patch
from methods.subagent_consensus import merge_consensus, build_consensus_patch, medoid_index

ann = {}
# Full annotated set (superset of every split) so re-scoring resolves any cid,
# whether a model's run dir is on the old or new (final/) test split.
for f in glob.glob("benchmark/data/annotated/*.json"):
    a = json.load(open(f)); ann[a["id"]] = a
SEEDS = (0, 42, 84, 126, 168)

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



def sc(patch, cid):
    a = ann[cid]
    return score_strict(a["current_json"], a["expected_patch"], patch, a.get("value_constraints"))
def is_succ(r):
    vr = r.get("value_recall"); vr1 = 1.0 if vr is None else vr
    return r["over_edit"] == 0 and r["miss"] == 0 and vr1 >= 1.0

# USD per input/output token (OpenRouter, 2026-06-12); x axis = cost per case
PRICES = {
    "gpt-oss-20b": (0.029e-6, 0.140e-6),
    "gpt-oss-120b": (0.039e-6, 0.180e-6),
    "gpt-5.4-mini": (0.750e-6, 4.500e-6),
    "Qwen3.5-9B": (0.100e-6, 0.150e-6),
    "Qwen3.5-27B": (0.195e-6, 1.560e-6),
    "Qwen3.5-122B": (0.260e-6, 2.080e-6),
}

def agg(rows, prices):
    pin, pout = prices
    per_seed = {}
    for ok, s, ci, co in rows: per_seed.setdefault(s, []).append((ok, ci, co))
    srs = [100 * sum(ok for ok, _, _ in v) / len(v) for v in per_seed.values()]
    flat = [x for v in per_seed.values() for x in v]
    cost = st.mean([ci for _, ci, _ in flat]) * pin + st.mean([co for _, _, co in flat]) * pout
    return (st.mean(srs), st.pstdev(srs), cost * 100)   # US cents per case

# Per-panel y-ranges (chosen to frame each model; spans may differ).
YLIMS = {"gpt-oss-20b": (75, 95), "gpt-oss-120b": (80, 95), "gpt-5.4-mini": (85, 100),
         "Qwen3.5-9B": (60, 85), "Qwen3.5-27B": (80, 98), "Qwen3.5-122B": (80, 98)}

FULL = lambda p: dict(scale=p, judge=p)
MODELS = [
    ("gpt-oss-20b", dict(scale="results/bench/final/testclean-scale5-seed{s}-gpt-oss-20b",
                         judge="results/bench/final/testclean-judge-gptoss-seed{s}-gpt-oss-20b",
                         consx=("results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b", "sub_hist_json_cons7_and"),
                         revx=("results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b", "sub_hist_json_rev6"),
                         judgek="results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b")),
    ("gpt-oss-120b", {**FULL("results/bench/final/testclean-full120-seed{s}-gpt-oss-120b"),
                      "consx": ("results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b", "sub_hist_json_cons7_and"),
                      "revx": ("results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b", "sub_hist_json_rev6"),
                      "judgek": "results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b"}),
    ("gpt-5.4-mini", {**FULL("results/bench/final/testclean-fullgpt54mini-med-seed{s}-gpt-5.4-mini"),
                      "consx": ("results/bench/final/testclean-cons8gpt54mini-med-seed{s}-gpt-5.4-mini", "sub_hist_json_cons8_and"),
                      "revx": ("results/bench/final/testclean-rev7gpt54mini-med-seed{s}-gpt-5.4-mini", "sub_hist_json_rev7"),
                      "judgek": "results/bench/final/testclean-judgekgpt54mini-med-seed{s}-gpt-5.4-mini"}),
    ("Qwen3.5-9B", dict(scale="results/bench/final/testclean-scale5-q9b-seed{s}-qwen9b-llm",
                        judge="results/bench/final/testclean-judge-q9b-seed{s}-qwen9b-llm",
                        consx=("results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm", "sub_hist_json_cons7_and"),
                        revx=("results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm", "sub_hist_json_rev6"),
                        judgek="results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm")),
    ("Qwen3.5-27B", {**FULL("results/bench/final/testclean-full27b-seed{s}-qwen27b-llm"),
                     "consx": ("results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm", "sub_hist_json_cons8_and"),
                     "revx": ("results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm", "sub_hist_json_rev7"),
                     "judgek": "results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm"}),
    ("Qwen3.5-122B", {**FULL("results/bench/final/testclean-full122b-seed{s}-qwen122b-llm"),
                      "consx": ("results/bench/final/testclean-ext-122b-seed{s}-qwen122b-llm", "sub_hist_json_cons9_and"),
                      "revx": ("results/bench/final/testclean-ext-122b-seed{s}-qwen122b-llm", "sub_hist_json_rev8"),
                      "revx_lo": ("results/bench/final/testclean-ext-122b-seed{s}-qwen122b-llm", "sub_hist_json_rev6"),
                      "judgek": "results/bench/final/testclean-judgek-122b-seed{s}-qwen122b-llm"}),
]

DUMP = {}   # per-model per-curve (k, success%, sd, cost cents) for the write-up
fig, axes = plt.subplots(2, 3, figsize=(21, 9.75))   # same aspect as the success grid (15.5:7.2)
for ax, (mname, cfg) in zip(axes.flat, MODELS):
    def load(pat, meth):
        d = {}
        for s in SEEDS:
            f = runf(pat, s, f"per_case_{meth}.json")
            if not os.path.exists(f): continue
            for c in json.load(open(f)):
                d[(c["id"], s)] = c
        return d
    curves = {}
    revx = load(cfg["revx"][0], cfg["revx"][1]) if cfg.get("revx") else {}
    rev4 = revx if revx else load(cfg["scale"], "sub_hist_json_rev4")
    # Optional lower-round serial run, used for xk it can reach (k <= its
    # rounds), so a HIGHER-round run's context overflow on the longest cases
    # (which aborts the whole case, nulling its lower stages) does not
    # contaminate the lower xk. Each xk is read from the shortest run reaching
    # round k.
    rev_lo = load(cfg["revx_lo"][0], cfg["revx_lo"][1]) if cfg.get("revx_lo") else {}
    rev_lo_rounds = int(cfg["revx_lo"][1].split("rev")[1]) if rev_lo else -1
    # R review rounds -> review_stages has R+1 entries (x0..xR); plot range is
    # R+1 so the serial ladder reaches xR (rev6 -> x6, rev7 -> x7).
    rev_rounds = int(cfg["revx"][1].split("rev")[1]) if revx else 4
    kcut = 1 if mname == "gpt-5.4-mini" else 0   # mini: drop one k from MED/SEQ/OR/AND
    ser_rounds = rev_rounds + 1 - kcut
    if mname == "Qwen3.5-9B":
        ser_rounds = 5   # drop the two rightmost serial points (cap at x4)
    cons = load(cfg["scale"], "sub_hist_json_cons5_and")
    # The x0 anchor must equal the base cell exactly. Cases that overflowed
    # the context window during a LATER review round have review_stages=None,
    # so their round-0 state is recovered from the cache-identical first
    # consensus sample; the failure is charged to k>=1 only.
    base_pt = {}
    for (cid, s), c in cons.items():
        smp = c["estimator"]["samples"][:1]
        if smp and smp[0]["patch"] is not None:
            base_pt[(cid, s)] = (is_succ(sc(smp[0]["patch"], cid)),
                                 smp[0]["usage"]["in"], smp[0]["usage"]["out"])
    pts = []
    for k in range(0, ser_rounds):
        # x0 = the single first-pass call = sample 1 (cache-identical to the
        # MED/SELECT/consensus k=1 anchor). Read it from the SAME source
        # (base_pt) so every curve's origin coincides exactly.
        if k == 0:
            rows = [(ok, s, ci, co) for (cid, s), (ok, ci, co) in base_pt.items()]
            pts.append(agg(rows, PRICES[mname]))
            continue
        rows = []
        src = rev_lo if (rev_lo and k <= rev_lo_rounds) else rev4
        for (cid, s), c in src.items():
            u = c["usage"]
            if c.get("review_stages"):
                ci = u["rev_in"] + sum(ru["in"] for ru in c["review_usages"][:k])
                co = u["rev_out"] + sum(ru["out"] for ru in c["review_usages"][:k])
                ok = is_succ(sc(c["review_stages"][k], cid))
            else:   # context overflow at a later round: failure from k>=1 on
                ci, co = u.get("rev_in", u["in"]), u.get("rev_out", u["out"])
                ok = is_succ(sc(c.get("model_patch") or [], cid))
            rows.append((ok, s, ci, co))
        pts.append(agg(rows, PRICES[mname]))
    curves["REFLECT"] = (pts, "o-", "#1f77b4", list(range(0, ser_rounds)))
    # When a cons8 run exists (equal-cost extension), the MAJ ladder gains k=7
    # and MED becomes a k=5..8 ladder; samples 1-5 are cache-identical.
    cons8 = load(cfg["consx"][0], cfg["consx"][1]) if cfg.get("consx") else {}
    cseries_src = cons8 if cons8 else cons
    # max k = the extension run's sample count (cons7 -> 7, cons8 -> 8), else 5
    kmax = int(cfg["consx"][1].split("cons")[1].split("_")[0]) if cons8 else 5
    maj_ks = tuple(k for k in range(1, kmax + 1) if k % 2) if cons8 else (1, 3, 5)
    oa_ks = range(1, kmax + 1 - kcut)   # mini drops one k from OR/AND
    for mode, marker, color, ks, src_d in (("or", "s-", "#2ca02c", oa_ks, cseries_src),
                                           ("maj", "D-", "#d62728", maj_ks, cseries_src),
                                           ("and", "v-", "#7f7f7f", oa_ks, cseries_src)):
        pts = []
        for k in ks:
            rows = []
            for (cid, s), c in src_d.items():
                samples = c["estimator"]["samples"][:k]
                cur = ann[cid]["current_json"]
                pairs = []
                for smp in samples:
                    if smp["patch"] is None: continue
                    try: pairs.append({"patch": smp["patch"], "_applied": apply_patch(cur, smp["patch"])})
                    except Exception: pass
                applied = [sp["_applied"] for sp in pairs]
                patch = build_consensus_patch(cur, merge_consensus(cur, applied, mode), pairs) if applied else []
                rows.append((is_succ(sc(patch, cid)), s,
                             sum(s2["usage"]["in"] for s2 in samples),
                             sum(s2["usage"]["out"] for s2 in samples)))
            pts.append(agg(rows, PRICES[mname]))
        curves[f"parallel {mode.upper()}"] = (pts, marker, color, list(ks))
    med_src = cons8 if cons8 else cons
    # k=1 anchor = the single sample = the base cell; MED and SELECT both
    # start there so their growth is drawn from the 1-call origin.
    med_ks = (1,) + tuple(range(3, kmax + 1 - kcut)) if cons8 else (1, 3, 4, 5)
    pts = []
    for k in med_ks:
        rows = []
        for (cid, s), c in med_src.items():
            samples = c["estimator"]["samples"][:k]
            cur = ann[cid]["current_json"]
            P = []
            for smp in samples:
                if smp["patch"] is None: continue
                try: P.append((smp["patch"], apply_patch(cur, smp["patch"])))
                except Exception: pass
            patch = P[medoid_index([ap for _, ap in P])][0] if P else []
            rows.append((is_succ(sc(patch, cid)), s,
                         sum(s2["usage"]["in"] for s2 in samples),
                         sum(s2["usage"]["out"] for s2 in samples)))
        pts.append(agg(rows, PRICES[mname]))
    curves["MED (medoid)"] = (pts, "^-", "#9467bd", list(med_ks))
    judge_files = [(5, load(cfg["judge"], "sub_hist_json_judge5"))]
    if cfg.get("judgek"):
        judge_files = [(k, load(cfg["judgek"], f"sub_hist_json_judge{k}")) for k in (2, 3, 4)] + judge_files
    rows = []
    for (cid, s), c in med_src.items():
        smp = c["estimator"]["samples"][:1]
        if not smp or smp[0]["patch"] is None: continue
        rows.append((is_succ(sc(smp[0]["patch"], cid)), s,
                     smp[0]["usage"]["in"], smp[0]["usage"]["out"]))
    pts = [agg(rows, PRICES[mname])]   # k=1 anchor (no judge call)
    for k, jd in judge_files:
        rows = []
        for (cid, s), c in jd.items():
            mt = c.get("metrics")
            ok = is_succ(mt) if mt else is_succ(sc(c.get("model_patch") or [], cid))
            rows.append((ok, s, c["usage"]["in"], c["usage"]["out"]))
        pts.append(agg(rows, PRICES[mname]))
    curves["SELECT (k+1)"] = (pts, "*-", "#ff7f0e", [1] + [k for k, _ in judge_files])

    DUMP[mname] = {name: {"k": kvals, "success": [round(p[0], 2) for p in pts],
                          "sd": [round(p[1], 2) for p in pts], "cost_cents": [round(p[2], 4) for p in pts]}
                   for name, (pts, _, _, kvals) in curves.items()}

    base_y = curves["parallel OR"][0][0][0]
    for name, (pts, marker, color, kvals) in curves.items():
        xs = [p[2] for p in pts]; ys = [p[0] for p in pts]; es = [p[1] for p in pts]
        m = marker.rstrip("-")
        big = name.startswith("MED") or name.startswith("SELECT")
        msz = 13 if big else 5
        base_alpha = 1.0   # solid (within-budget) region fully opaque
        zb = 5 if big else 2
        # Solid region = up to a 5-CALL budget; beyond that is faded reference.
        # judge@4 = 4 samples + 1 judge = 5 calls; consensus k=5 = 5 calls;
        # serial x4 = base + 4 reviews = 5 calls. So judge fades at k>=5,
        # consensus at k>=6, serial at x>=5.
        thr = 5 if name == "REFLECT" else (5 if name.startswith("SELECT") else 6)
        n_solid = max(sum(1 for k in kvals if k < thr), 1)
        # Lines connect from the shared origin (index 0); markers/error bars are
        # drawn only for k>=1 (the origin is shown once as a black base point).
        ax.plot(xs[:n_solid], ys[:n_solid], "-", color=color, lw=1.8,
                alpha=base_alpha, zorder=zb)
        if len(xs) > 1:
            ax.errorbar(xs[1:n_solid], ys[1:n_solid], yerr=es[1:n_solid], fmt=m,
                        color=color, capsize=3, markersize=msz, linestyle="none",
                        alpha=base_alpha, zorder=zb)
        if n_solid < len(pts):
            j = n_solid - 1   # include last solid point so the line connects
            ax.plot(xs[j:], ys[j:], "-", color=color, lw=1.8, alpha=0.3, zorder=zb - 1)
            ax.errorbar(xs[n_solid:], ys[n_solid:], yerr=es[n_solid:], fmt=m,
                        color=color, capsize=3, markersize=msz, linestyle="none",
                        alpha=0.3, zorder=zb - 1)
    # Shared origin = the single first-pass call (sample 1 = J+H base), drawn
    # once as a black point with its error bar.
    ox, oy, oe = curves["MED (medoid)"][0][0][2], curves["MED (medoid)"][0][0][0], curves["MED (medoid)"][0][0][1]
    ax.errorbar([ox], [oy], yerr=[oe], fmt="o", color="black", markersize=6,
                capsize=3, zorder=7)
    ax.axhline(base_y, color="k", linestyle=":", linewidth=1.8, alpha=0.9)
    ax.set_title(mname, fontsize=20)
    ax.set_ylim(*YLIMS[mname])
    ax.margins(x=0.06)
    ax.grid(alpha=0.25)
    ax.set_axisbelow(True)
    ax.tick_params(labelsize=16)
    ax.locator_params(axis="x", nbins=6)   # cap x-ticks so dense panels (e.g. gpt-oss-20b) don't crowd
    if mname == "gpt-oss-20b":
        ax.yaxis.set_major_locator(MultipleLocator(5))   # 5-pt y ticks: 2.5-pt labels got clipped
for ax in axes[1]:
    ax.set_xlabel("API cost (US cents)", fontsize=18)
for ax in axes[:, 0]:
    ax.set_ylabel("completion rate (%)", fontsize=18)
# Legend: proxy handles showing line + marker (no error bars).
from matplotlib.lines import Line2D
_LEG = [("base (J+H)", "o", "black"), ("REFLECT", "o", "#1f77b4"),
        ("OR", "s", "#2ca02c"), ("MAJ", "D", "#d62728"),
        ("AND", "v", "#7f7f7f"), ("MED", "^", "#9467bd"),
        ("SELECT", "*", "#ff7f0e")]
handles = [Line2D([0], [0], color=c, marker=mk, linestyle="-",
                  markersize=11 if name.startswith(("MED", "SELECT")) else 7,
                  linewidth=1.8) for name, mk, c in _LEG]
axes[0, 0].legend(handles, [n for n, _, _ in _LEG], fontsize=12, loc="lower right", ncol=2)
fig.tight_layout()
fig.savefig("results/figures/cost_curve_grid_6models.png", dpi=300)
fig.savefig("results/figures/cost_curve_grid_6models.pdf")
json.dump(DUMP, open("results/figures/cost_grid_numbers.json", "w"), indent=2)
print("saved results/figures/cost_curve_grid_6models.{png,pdf} + cost_grid_numbers.json")
