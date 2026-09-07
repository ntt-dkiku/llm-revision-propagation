"""Reconstruct per-case wall-clock latency of the revision methods from the
litellm disk cache (store timestamp - vLLM `created` = per-call latency).

Method wall-clock model (deployment view), k = total LLM calls (2..5):
  J+H        = lat(single call)
  REFLECT@k  = lat(base) + sum of first k-1 serial review rounds
  MED@k      = max(lat of first k parallel samples)      (medoid pick is local)
  SELECT@k   = max(lat of k-1 parallel samples) + lat(judge)  (k>=3; judge{k-1} run)

Joins: cache entry -> per_case via sha1 of message content (samples / base) or
reasoning_content (judge). When the judge has no reasoning text to join (API
model), its prompt tokens are estimated from the saved usage and matched to the
nearest bare-digit-output cache entry. gpt-5.4-mini latencies carry a clock
offset (server vs local); corrected by subtracting min(lat)-1s, approximate.
REFLECT's intermediate rounds store tokens only, so their latency is a
per-model regression estimate (latency ~ completion tokens).

Run from the repo root on the host: python3 scripts/latency_recon.py
Needs .litellm_cache/ and results/bench/final/. Dumps per-case values to
results/latency_recon.json (the tables can be re-printed from that file alone).
"""
import json, sqlite3, pickle, hashlib, os, statistics as st
from collections import defaultdict

TARGET = {  # cache model string -> display name
    "local-llm": "gpt-oss-20b", "gptoss120-llm": "gpt-oss-120b",
    "gpt-5.4-mini-2026-03-17": "gpt-5.4-mini",
    "qwen9b-llm": "Qwen3.5-9B", "qwen27b-llm": "Qwen3.5-27B", "qwen122b-llm": "Qwen3.5-122B",
}
NEEDLES = {k: f'"model":"{k}"'.encode() for k in TARGET}
KS = [2, 3, 4, 5]

def sha(s): return hashlib.sha1((s or "").encode()).hexdigest()

# ---------- pass 1: index the cache ----------
bycontent = defaultdict(list)   # (model, sha(content)) -> [(lat, out_tok)]
byreason = defaultdict(list)    # (model, sha(reasoning)) -> [(lat, out_tok)]
digit_calls = defaultdict(list) # model -> [(lat, prompt_tok)] for bare 1-5 outputs (judges)
all_lat = defaultdict(list)     # model -> [(lat, out_tok)] for regression/offset
db = sqlite3.connect("file:.litellm_cache/cache.db?mode=ro&immutable=1", uri=True)
n_seen = n_used = n_badfile = 0
for val, fn in db.execute("SELECT value, filename FROM Cache"):
    if val is None:
        if fn is None:
            continue
        try:
            val = open(f".litellm_cache/{fn}", "rb").read()
        except OSError:
            n_badfile += 1
            continue
    n_seen += 1
    if not any(nd in val for nd in NEEDLES.values()):
        continue
    try:
        d = pickle.loads(val); r = json.loads(d["response"])
    except Exception:
        continue
    m = r.get("model")
    if m not in TARGET:
        continue
    lat = d["timestamp"] - r["created"]
    if lat < -2 or lat > 3600:
        continue
    msg = (r.get("choices") or [{}])[0].get("message") or {}
    content = msg.get("content") or ""
    reasoning = msg.get("reasoning_content") or ""
    usage = r.get("usage") or {}
    out = usage.get("completion_tokens") or 0
    mn = TARGET[m]; n_used += 1
    bycontent[(mn, sha(content))].append((lat, out))
    if reasoning:
        byreason[(mn, sha(reasoning))].append((lat, out))
    if content.strip() in {"1", "2", "3", "4", "5"}:
        digit_calls[mn].append((lat, usage.get("prompt_tokens") or 0))
    all_lat[mn].append((lat, out))
print(f"cache pass: scanned={n_seen} indexed={n_used} unreadable_files={n_badfile}")

# clock-offset correction (API model): local clock vs provider `created`
OFFSET = {mn: 0.0 for mn in set(TARGET.values())}
mini = "gpt-5.4-mini"
if all_lat[mini]:
    OFFSET[mini] = min(l for l, _ in all_lat[mini]) - 1.0
    print(f"gpt-5.4-mini clock offset estimate: {OFFSET[mini]:.1f}s (subtracted; approximate)")

def fix(mn, lat): return max(lat - OFFSET[mn], 0.3)

# per-model regression lat ~ a + b*out_tok (for REFLECT round estimates)
REG = {}
for mn, pts in all_lat.items():
    xs = [o for _, o in pts]; ys = [fix(mn, l) for l, _ in pts]
    mx, my = st.mean(xs), st.mean(ys)
    b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / max(sum((x - mx) ** 2 for x in xs), 1e-9)
    REG[mn] = (my - b * mx, b)

def lookup(idx, mn, text, out_tok):
    """Latency of the cache entry matching this output text; closest-token entry on collision."""
    c = idx.get((mn, sha(text)))
    if not c:
        return None
    best = min(c, key=lambda t: abs(t[1] - (out_tok or t[1])))
    return fix(mn, best[0])

def judge_lat(mn, est):
    """Judge-call latency: reasoning-text join, else nearest-prompt-token digit call."""
    jl = lookup(byreason, mn, est.get("judge_reasoning"), None)
    if jl is not None:
        return jl
    if not digit_calls[mn]:
        return None
    samples = est.get("samples") or []
    ein = (est.get("usage") or {}).get("in") or 0
    jin = ein - sum((s.get("usage") or {}).get("in", 0) for s in samples)
    if jin > 0:
        best = min(digit_calls[mn], key=lambda t: abs(t[1] - jin))
        return fix(mn, best[0])
    return fix(mn, st.median(l for l, _ in digit_calls[mn]))

# ---------- pass 2: per-case reconstruction ----------
SEEDS = [0, 42, 84, 126, 168]
FULL = lambda p: dict(sub=p, scale=p, judge=p)
MODELS = [  # display name, dirs for base/scale/judge5, dir for judge2-4 (ext)
 ("gpt-oss-20b", dict(sub="results/bench/final/testclean-sub-seed{s}-gpt-oss-20b",
                      scale="results/bench/final/testclean-scale5-seed{s}-gpt-oss-20b",
                      judge="results/bench/final/testclean-judge-gptoss-seed{s}-gpt-oss-20b",
                      ext="results/bench/final/testclean-ext-20b-seed{s}-gpt-oss-20b")),
 ("gpt-oss-120b", dict(**FULL("results/bench/final/testclean-full120-seed{s}-gpt-oss-120b"),
                       ext="results/bench/final/testclean-ext-120b-seed{s}-gpt-oss-120b")),
 ("gpt-5.4-mini", dict(**FULL("results/bench/final/testclean-fullgpt54mini-med-seed{s}-gpt-5.4-mini"),
                       ext="results/bench/final/testclean-judgekgpt54mini-med-seed{s}-gpt-5.4-mini")),
 ("Qwen3.5-9B", dict(sub="results/bench/final/testclean-sub-q9b-seed{s}-qwen9b-llm",
                     scale="results/bench/final/testclean-scale5-q9b-seed{s}-qwen9b-llm",
                     judge="results/bench/final/testclean-judge-q9b-seed{s}-qwen9b-llm",
                     ext="results/bench/final/testclean-ext-9b-seed{s}-qwen9b-llm")),
 ("Qwen3.5-27B", dict(**FULL("results/bench/final/testclean-full27b-seed{s}-qwen27b-llm"),
                      ext="results/bench/final/testclean-ext-27b-seed{s}-qwen27b-llm")),
 ("Qwen3.5-122B", dict(**FULL("results/bench/final/testclean-full122b-seed{s}-qwen122b-llm"),
                       ext="results/bench/final/testclean-judgek-122b-seed{s}-qwen122b-llm")),
]

def jget(f): return json.load(open(f)) if os.path.exists(f) else []

wall = defaultdict(lambda: defaultdict(list))  # model -> method -> [wall_s per case]
cov = defaultdict(lambda: [0, 0])              # (model, method) -> [joined, total]

def sample_lats(mn, samples):
    ls = [lookup(bycontent, mn, sm.get("raw_answer"), (sm.get("usage") or {}).get("out"))
          for sm in samples]
    return ls if ls and all(x is not None for x in ls) else None

for mn, cfg in MODELS:
    a, b = REG[mn]
    for s in SEEDS:
        # J+H
        for c in jget(f"{cfg['sub'].format(s=s)}/per_case_sub_hist_json.json"):
            l = lookup(bycontent, mn, c.get("raw_answer"), (c.get("usage") or {}).get("out"))
            cov[(mn, "J+H")][1] += 1
            if l is not None:
                wall[mn]["J+H"].append(l); cov[(mn, "J+H")][0] += 1
        # REFLECT@k and MED@k from the scale dir (rev4 / cons5 records)
        for c in jget(f"{cfg['scale'].format(s=s)}/per_case_sub_hist_json_rev4.json"):
            base = lookup(bycontent, mn, c.get("raw_answer"), (c.get("usage") or {}).get("out"))
            rus = c.get("review_usages") or []
            for k in KS:
                cov[(mn, f"REFLECT@{k}")][1] += 1
                if base is None or len(rus) < k - 1:
                    continue
                rounds = sum(max(a + b * (u.get("out") or 0), 0.3) for u in rus[:k - 1])
                wall[mn][f"REFLECT@{k}"].append(base + rounds); cov[(mn, f"REFLECT@{k}")][0] += 1
        for c in jget(f"{cfg['scale'].format(s=s)}/per_case_sub_hist_json_cons5_and.json"):
            all5 = sample_lats(mn, c["estimator"]["samples"][:5])
            for k in KS:
                cov[(mn, f"MED@{k}")][1] += 1
                if all5 is not None and len(all5) >= k:
                    wall[mn][f"MED@{k}"].append(max(all5[:k])); cov[(mn, f"MED@{k}")][0] += 1
        # SELECT@k (k total calls) = the judge{k-1} run: its k-1 samples + its
        # judge (which judged exactly those k-1 candidates). judge{n} = n samples
        # + 1 judge = n+1 calls, so SELECT@2 (judge1) does not exist and the
        # paper's five-call SELECT is judge4. judge2-4 live in the ext dirs.
        for k in KS:
            if k < 3:
                continue
            d = cfg["ext"]
            for c in jget(f"{d.format(s=s)}/per_case_sub_hist_json_judge{k - 1}.json"):
                est = c["estimator"]
                ls = sample_lats(mn, est["samples"][:k - 1])
                jl = judge_lat(mn, est)
                cov[(mn, f"SELECT@{k}")][1] += 1
                if ls is not None and jl is not None:
                    wall[mn][f"SELECT@{k}"].append(max(ls) + jl); cov[(mn, f"SELECT@{k}")][0] += 1

# ---------- report ----------
GROUPS = [("REFLECT", [f"REFLECT@{k}" for k in KS]),
          ("MED", [f"MED@{k}" for k in KS]),
          ("SELECT", [f"SELECT@{k}" for k in KS])]
print("\n== join coverage: listing only methods NOT fully joined ==")
for mn, _ in MODELS:
    bad = [f"{m}:{cov[(mn,m)][0]}/{cov[(mn,m)][1]}" for m in ["J+H"] + [x for _, g in GROUPS for x in g]
           if cov[(mn, m)][0] != cov[(mn, m)][1]]
    print(f"  {mn:13} " + ("all full" if not bad else "  ".join(bad)))

print("\n== wall-clock per case (seconds), mean [p50] ==")
for gname, cols in GROUPS:
    print(f"\n-- {gname} --")
    print(f"{'model':13} {'J+H':>15}" + "".join(f"{c:>16}" for c in cols))
    for mn, _ in MODELS:
        cells = [f"{st.mean(wall[mn]['J+H']):6.1f} [{st.median(wall[mn]['J+H']):5.1f}]"]
        for c in cols:
            v = wall[mn][c]
            cells.append(f"{st.mean(v):6.1f} [{st.median(v):5.1f}]" if v else "--")
        print(f"{mn:13} {cells[0]:>15}" + "".join(f"{x:>16}" for x in cells[1:]))

print("\n== slowdown vs the same model's J+H (mean ratio) ==")
allcols = [x for _, g in GROUPS for x in g]
print(f"{'model':13}" + "".join(f"{c:>11}" for c in allcols))
for mn, _ in MODELS:
    base = st.mean(wall[mn]["J+H"])
    row = "".join(f"{st.mean(wall[mn][c])/base:10.2f}x" if wall[mn][c] else f"{'--':>11}" for c in allcols)
    print(f"{mn:13}" + row)

json.dump({mn: dict(wall[mn]) for mn, _ in MODELS}, open("results/latency_recon.json", "w"))
print("\n(REFLECT rounds are token-regression estimates; gpt-5.4-mini offset-corrected, approximate.)")
print("dumped results/latency_recon.json")
