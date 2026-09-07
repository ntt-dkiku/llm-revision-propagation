---
name: run-cascade-benchmark
description: >-
  How to stand up the local vLLM model server(s) and run the human-annotated
  cascade-revision benchmark (benchmark/) end to end with the Makefile. Use
  this when someone wants to evaluate a model/method on the JSON-Patch cascade
  task: starting/stopping vLLM, the `make bench-eval` command and its variables
  (model, seed, temperature, method subset, alternate endpoint), where outputs
  land, how to read the comparison table, and how to aggregate multiple seeds.
---

# Running the cascade-revision benchmark

The benchmark scores a model's ability to apply a single revision request to a
JSON artifact AND propagate every dependent (cascade) change, with no spurious
edits. Everything runs through the `Makefile` and a containerized runner; you do
not call Python directly.

## 0. One-time setup

```bash
make env            # creates docker/.env from docker/.env.example if missing
make split-dataset  # materializes annotated_dev/ (30) and annotated_test/ (120)
```

`make bench-eval` and `make figures` bring the runner container up on their
own (`make runner-up` also works standalone). Commercial-model keys (only if
evaluating OpenAI/Anthropic models) go in `docker/.env` as `OPENAI_API_KEY` /
`ANTHROPIC_API_KEY`; re-run `make runner-up` after editing so the container
picks them up.

`make split-dataset` re-runs the deterministic balanced split search (seed 42)
and reproduces the paper's split; `CACHE=true` skips the search and just copies
the case files listed in the tracked `benchmark/data/split_dev30_test120.json`.

## 1. Start a local model server (vLLM)

The five local paper models have presets — `make vllm-up MODEL=<name>` serves
each with the paper's settings on its own port:

| `MODEL` (alias) | port | served name | GPUs |
|---|---|---|---|
| `gpt-oss-20b` | 8001 | `local-llm` | 1 |
| `gpt-oss-120b` | 8004 | `gptoss120-llm` | 1 |
| `qwen3.5-9b` (`qwen9b-llm`) | 8003 | `qwen9b-llm` | 1 |
| `qwen3.5-27b` (`qwen27b-llm`) | 8005 | `qwen27b-llm` | 1 |
| `qwen3.5-122b-a10b` (`qwen122b-llm`) | 8002 | `qwen122b-llm` | 4 (TP=4) |

```bash
make vllm-up MODEL=gpt-oss-20b     # start; weights load over a few minutes
make smoke-llm                     # curl the /v1 endpoints to confirm it's up
docker logs -f revpropbench-vllm-gptoss20   # watch load progress (container name per compose)
make vllm-down MODEL=gpt-oss-20b   # stop (use the same MODEL)
```

Pass `VLLM_GPUS=<ids>` as a make variable to select GPUs (default `0`; the
122B preset defaults to `0,1,2,3`) — e.g.
`make vllm-up MODEL=qwen3.5-27b VLLM_GPUS=1`. It must be a make argument, not
a shell environment variable: the Makefile includes `docker/.env`, whose
`VLLM_GPUS` value overrides the environment. Since the ports differ, several
presets can run at once on different GPUs.

For any other local model, set the `VLLM_*` values in `docker/.env` and run
`make vllm-up` without `MODEL` (generic compose); then evaluate with
`MODEL=<VLLM_SERVED_NAME value>`.

Health check before evaluating (must return the served name):

```bash
curl -s http://localhost:8001/v1/models | grep -o '"id":"[^"]*"'
```

## 2. Run an evaluation

```bash
make bench-eval MODEL=gpt-oss-20b BENCH_SEED=42
```

`MODEL` selects the endpoint/served-name via the preset mapping above (the
served-name aliases work too and keep the run-dir suffix identical to the
paper's campaign dirs). Any other value is passed to LiteLLM verbatim (hosted
providers, e.g. `MODEL=anthropic/claude-sonnet-5`); `MODEL=gpt-5.4-mini` maps
to hosted OpenAI.

| variable | default | meaning |
|---|---|---|
| `BENCH_ANNOTATED` | `benchmark/data/annotated_test` | case dir; `annotated_dev` (30) for development, `benchmark/data/annotated` for all 150 |
| `BENCH_RUN_DIR` | `results/bench/final/seed$(BENCH_SEED)` | output dir; `-$(MODEL)` is appended automatically, so the default becomes `final/seed<seed>-<MODEL>` — the layout `make figures` reads |
| `BENCH_SEED` | 42 | sampling seed (decode + cache key) |
| `BENCH_TEMP` | 0.6 | temperature (the paper's local-model setting) |
| `BENCH_CONCURRENCY` | 8 | cases in flight at once |
| `BENCH_METHODS` | (all) | space-separated method subset |
| `BENCH_REASONING_EFFORT` | (provider default) | OpenAI-style reasoning effort for reasoning models |
| `BENCH_EVAL_URL` | (per-MODEL default) | override endpoint, e.g. a second instance on another port |

Paper protocol: seeds `0 42 84 126 168`, test split. For `gpt-5.4-mini` add
`BENCH_TEMP=1.0 BENCH_REASONING_EFFORT=medium` (the paper's setting for it).

**Protocol discipline:** develop and analyze on `annotated_dev` only; use
`annotated_test` for final scores, not for tuning.

### Method names (`BENCH_METHODS`)

Baselines: `sub_json` (J), `sub_hist` (H), `sub_hist_json` (J+H).
Test-time compute at the paper's five calls: `sub_hist_json_rev4` (REFLECT),
`sub_hist_json_cons5_and` (its stored samples also yield OR / MAJ / MED),
`sub_hist_json_judge4` (SELECT = 4 samples + 1 selection).
Call-budget variants for the cost curves: `sub_hist_json_rev3/6/7/8`,
`sub_hist_json_cons3_{and,or,maj}`, `sub_hist_json_cons7/8/9_and`,
`sub_hist_json_judge2/3/5`, `sub_hist_json_merge2-5`. Full list:
`docker exec revpropbench-runner python -c "from methods import METHOD_NAMES; print(METHOD_NAMES)"`.

### Multi-model / multi-seed campaigns

Each preset has its own port, so different models evaluate in parallel (one
`make vllm-up` per model on separate GPUs, then one `make bench-eval` each).
Seeds sharing one server run sequentially. The LiteLLM disk cache keys on
(messages, model, seed, temperature), so re-running an identical config is
instant. For extra instances of the same model, launch another server on a
free port and point the eval at it with `BENCH_EVAL_URL`.

## 3. Outputs and reading them

Each run dir (`<BENCH_RUN_DIR>-<MODEL>/`) contains:
- `per_case_<method>.json` — one record per case: `model_patch`, `metrics`,
  `usage`, `raw_answer`, `reasoning_content` (multi-call methods also carry an
  `estimator` sub-record with their intermediate samples).
- `rollup_<method>.json` — aggregated metrics for that method.
- `comparison.json` — all methods' rollups side by side.

The run also prints a comparison table (rows = metrics, columns = methods). Key
metrics:
- `success_rate` — fraction of cases that are FULLY correct: `over_edit == 0`
  AND `miss == 0` AND `value_recall >= 1.0`. This is the headline number
  (called completion rate in the paper).
- `loc_success_rate` — location-only exact match, ignoring value correctness.
- `over_edit` / `miss` — mean spurious edits / missed gold leaves per case.
- `loc_f1` / `loc_precision` / `loc_recall`, `value_recall`,
  `direct_recall` / `cascade_recall` (split by edit kind), `parse_fail_rate`,
  `mean_in_tok` / `mean_out_tok`.

Re-print a saved run's table:

```bash
docker exec revpropbench-runner python -c "import json; d=json.load(open('results/bench/final/seed42-gpt-oss-20b/comparison.json')); [print(m, round(v['success_rate'],3), round(v['loc_success_rate'],3)) for m,v in d.items()]"
```

## 4. Aggregate success_rate across seeds (mean ± sd)

Run inside the container (`docker exec revpropbench-runner python - <<'PY'`):

```python
import json, statistics as st
seeds = [0, 42, 84, 126, 168]
method = "sub_hist_json"
prefix = "results/bench/final/seed{s}-gpt-oss-20b"   # match BENCH_RUN_DIR + MODEL
sr = [json.load(open(f"{prefix.format(s=s)}/comparison.json"))[method]["success_rate"]
      for s in seeds]
print(f"{method}: {st.mean(sr):.3f} +/- {st.pstdev(sr):.3f}  per-seed {[round(x,3) for x in sr]}")
```

To recompute metrics from saved `model_patch` without re-calling the model
(e.g. after a scorer change), use `benchmark.strict_score.score_strict(current,
expected_patch, model_patch, value_constraints)` over the `per_case_*` files and
the matching `benchmark/data/annotated*/<id>.json`.

## 5. Figures, tests, teardown

```bash
make extract-paper-runs  # unpack the paper's evaluation outputs into results/bench/final/
make figures             # regenerate the paper figures into results/figures/
make test                # pytest (scorer regression + gold-idempotence), ignores old/
make stop                # vllm-down + runner-down
```

The figure scripts read `results/bench/final/`: the paper's campaign run dirs
(from `make extract-paper-runs`) take precedence; otherwise they fall back to
the `seed<s>-<MODEL>` dirs a default `make bench-eval` writes.

## Gotchas

- `BENCH_METHODS` choices are the registry names above (J / H / J+H etc. are
  paper labels, not CLI names).
- Don't compare numbers across different prompt families; only compare methods
  that share the same system prompt and context sections (all `sub_*` methods do).
- A run dir already containing `comparison.json` for the same config re-runs
  cheaply from the LiteLLM cache.
- The evaluator's default split is `annotated_test`; run `make split-dataset`
  once after cloning or the case dir will be missing.
