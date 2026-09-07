# Benchmark statistics for the main text (run from repo root):
#   python scripts/plot_benchmark_stats_bc.py
# Emits a combined 1x3 figure AND three standalone panels (no in-image title,
# for LaTeX subfigure side-by-side use):
#   results/figures/benchmark_stats_bcd.{png,pdf}       (combined, titled, for preview)
#   results/figures/benchmark_stat_tier.{png,pdf}       (b) cascade vs artifact size
#   results/figures/benchmark_stat_domain.{png,pdf}     (c) cascade by domain
#   results/figures/benchmark_stat_pattern.{png,pdf}    (d) propagation patterns (n=50)
"""Reads the full benchmark (annotated_test + annotated_dev); n=150 samples,
50 scenarios. Panels (b)/(c) are per-sample (n=150); panel (d) is per-scenario
(n=50) since the propagation-pattern label is a scenario property."""
import json, glob, math, statistics as st
from collections import defaultdict, Counter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def jitter(n, width=0.30):   # deterministic (repeatable output; no RNG)
    if n == 1: return [0.0]
    return [(-width + 2 * width * i / (n - 1)) for i in range(n)]

FILES = sorted(glob.glob("benchmark/data/annotated/*.json"))
# Scenario propagation-pattern category per (domain|theme), taken from the
# skeleton registry (the canonical source of each scenario's category).
from benchmark.skeleton import SKELETONS_BY_DOMAIN
CATMAP = {f"{sk.domain}|{sk.theme}": sk.category
          for _sks in SKELETONS_BY_DOMAIN.values() for sk in _sks}
per_tier = defaultdict(list); per_dom = defaultdict(list); scn_cat = {}
for f in FILES:
    a = json.load(open(f))
    casc = sum(1 for o in (a.get("expected_patch") or []) if o.get("kind") == "cascade")
    per_tier[a.get("target_leaves")].append(casc)
    per_dom[a.get("domain")].append(casc)
    key = f"{a.get('domain')}|{a.get('theme')}"
    scn_cat[key] = CATMAP.get(key, "uncategorized")
cats = Counter(scn_cat.values())

TIERS = sorted(per_tier)                                   # 10, 50, 100
TIER_C = {10: "#d62728", 50: "#1f77b4", 100: "#2ca02c"}    # red / blue / green
doms = sorted(per_dom, key=lambda k: st.median(per_dom[k]))
DOM_SHORT = {"software_deployment_config": "software_config"}
CAT_ORDER = ["arithmetic", "temporal", "substitution", "threshold", "add_remove", "status_flip"]
# Two-line labels: long "<adjective> <noun>" tags spread the donut horizontally
# and leave wide side margins after bbox-tight cropping; stacking the words keeps
# the figure compact and the label text large.
CAT_LABEL = {"arithmetic": "Arithmetic\nrecompute", "temporal": "Temporal\nshift",
             "substitution": "Reference\nsubstitution", "threshold": "Threshold\ngating",
             "add_remove": "Structural\nadd/remove", "status_flip": "Status\npropagation"}
CAT_C = {"arithmetic": "#4292c6", "temporal": "#9467bd", "substitution": "#74c476",
         "threshold": "#fd8d3c", "add_remove": "#969696", "status_flip": "#e377c2"}


def draw_tier(ax, title=None):
    bx = ax.boxplot([per_tier[t] for t in TIERS], positions=range(len(TIERS)),
                    widths=0.55, patch_artist=True, showfliers=False, zorder=2)
    for p, t in zip(bx["boxes"], TIERS):
        p.set_facecolor("none"); p.set_edgecolor(TIER_C[t]); p.set_linewidth(1.8)
    for i, t in enumerate(TIERS):
        for part in ("whiskers", "caps"):
            bx[part][2 * i].set_color(TIER_C[t]); bx[part][2 * i + 1].set_color(TIER_C[t])
        bx["medians"][i].set_color(TIER_C[t]); bx["medians"][i].set_linewidth(1.8)
        ys = per_tier[t]
        ax.scatter([i + dx for dx in jitter(len(ys))], ys, s=14, color=TIER_C[t],
                   edgecolor="none", zorder=3, alpha=0.45)
    ax.set_xticks(range(len(TIERS))); ax.set_xticklabels([f"{t}" for t in TIERS])
    ax.set_xlabel("Artifact size (# elements)", fontsize=16)
    ax.set_ylabel("# revised elements", fontsize=16)
    ax.tick_params(labelsize=14)
    ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
    if title: ax.set_title(title, fontsize=15)


def draw_domain(ax, title=None):
    cb = ax.boxplot([per_dom[d] for d in doms], vert=False, widths=0.6,
                    patch_artist=True, showfliers=True)
    for p in cb["boxes"]: p.set_facecolor("#bdbdbd")
    for med in cb["medians"]: med.set_color("black"); med.set_linewidth(1.5)
    ax.set_yticks(range(1, len(doms) + 1))
    ax.set_yticklabels([DOM_SHORT.get(d, d) for d in doms], fontsize=12.5)
    ax.set_xlabel("# revised elements", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.grid(axis="x", alpha=0.25); ax.set_axisbelow(True)
    if title: ax.set_title(title, fontsize=15)


def draw_pattern(ax, title=None):
    labels = [c for c in CAT_ORDER if cats.get(c)]
    sizes = [cats[c] for c in labels]
    _, texts, autotexts = ax.pie(
        sizes, labels=[CAT_LABEL[c] for c in labels], colors=[CAT_C[c] for c in labels],
        autopct=lambda p: f"{p*sum(sizes)/100:.0f}\n({p:.0f}%)",
        pctdistance=0.79, startangle=90, counterclock=False, labeldistance=1.07,
        wedgeprops=dict(width=0.42, edgecolor="white"), textprops=dict(fontsize=13.5))
    # pctdistance = radial mid of the 0.42-wide ring (inner 0.58 .. outer 1.0 -> 0.79)
    # so the count/(%) sits in the center of each wedge band.
    for t in autotexts: t.set_fontsize(11.5)
    # Anchor each outer label just outside the ring and grow it AWAY from the
    # donut (alignment by quadrant) so labels hug the ring without overlapping it.
    for t in texts:
        x, y = t.get_position()
        ang = math.degrees(math.atan2(y, x)) % 360
        if 60 <= ang <= 120:        # top
            t.set_ha("center"); t.set_va("bottom")
        elif 240 <= ang <= 300:     # bottom
            t.set_ha("center"); t.set_va("top")
        elif x >= 0:                # right half
            t.set_ha("left"); t.set_va("center")
        else:                       # left half
            t.set_ha("right"); t.set_va("center")
    ax.set_aspect("equal")
    if title: ax.set_title(title, fontsize=15)


# combined preview (titled)
fig, ax = plt.subplots(1, 3, figsize=(19, 5.4))
draw_tier(ax[0], "(b) Cascade size scales with artifact size")
draw_domain(ax[1], "(c) Cascade size by domain")
draw_pattern(ax[2], "(d) Characteristic propagation patterns\n(per scenario, n=50)")
fig.tight_layout()
fig.savefig("results/figures/benchmark_stats_bcd.png", dpi=300)
fig.savefig("results/figures/benchmark_stats_bcd.pdf")

# standalone panels (no title -> LaTeX subcaption carries the label).
# figsizes chosen so the bbox-tight crops end up at a matched ~1.4 aspect, so
# the three subfigures have equal height when scaled to the same column width.
# tier/domain heights chosen so their bbox-tight aspect (h/w) matches the donut's
# (~0.834), so all three subfigures render at equal height when scaled to the
# same column width in the LaTeX subfigure row.
STANDALONE_SIZE = {"tier": (5.6, 4.70), "domain": (5.6, 4.70), "pattern": (4.7, 4.5)}
for name, drawer in (("tier", draw_tier), ("domain", draw_domain), ("pattern", draw_pattern)):
    f1, a1 = plt.subplots(figsize=STANDALONE_SIZE[name])
    drawer(a1)
    f1.tight_layout()
    # bbox_inches=tight expands the crop to include external pie labels that
    # would otherwise be clipped at the figure edge.
    f1.savefig(f"results/figures/benchmark_stat_{name}.png", dpi=300, bbox_inches="tight", pad_inches=0.05)
    f1.savefig(f"results/figures/benchmark_stat_{name}.pdf", bbox_inches="tight", pad_inches=0.05)
    plt.close(f1)

print("saved combined + 3 standalone panels to results/figures/")
for t in TIERS:
    v = per_tier[t]
    print(f"  tier {t:>3}: n={len(v)} median {st.median(v):.0f} mean {st.mean(v):.1f} max {max(v)}")
