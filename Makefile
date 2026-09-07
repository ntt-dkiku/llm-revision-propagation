.PHONY: help env vllm-up vllm-down smoke-llm \
        cooldown-check sync refresh-lock refresh-lock-now \
        runner-build runner-up eval-shell runner-down \
        annotator-up annotator-down annotator-logs \
        dataset split-dataset bench-sample bench-eval extract-paper-runs figures \
        test lint stop

SHELL := /bin/bash

ENV_FILE := docker/.env
ifeq (,$(wildcard $(ENV_FILE)))
$(shell cp docker/.env.example $(ENV_FILE))
endif

include $(ENV_FILE)
export

# `make vllm-up MODEL=<name>` serves a preset paper model from its dedicated
# compose (the five local paper models all have presets). Any other MODEL
# (including unset) uses the generic env-driven compose (set the VLLM_* values
# in docker/.env).
MODEL ?= gpt-oss-20b
ifeq ($(MODEL),gpt-oss-20b)
  VLLM_COMPOSE := -f docker/docker-compose.vllm-gptoss20.yaml
  VLLM_CONTAINER := revpropbench-vllm-gptoss20
else ifeq ($(MODEL),gpt-oss-120b)
  VLLM_COMPOSE := -f docker/docker-compose.vllm-gptoss120.yaml
  VLLM_CONTAINER := revpropbench-vllm-gptoss120
else ifneq (,$(filter $(MODEL),qwen3.5-9b qwen9b-llm))
  VLLM_COMPOSE := -f docker/docker-compose.vllm-qwen9b.yaml
  VLLM_CONTAINER := revpropbench-vllm-qwen9b
else ifneq (,$(filter $(MODEL),qwen3.5-27b qwen27b-llm))
  VLLM_COMPOSE := -f docker/docker-compose.vllm-qwen27b.yaml
  VLLM_CONTAINER := revpropbench-vllm-qwen27b
else ifneq (,$(filter $(MODEL),qwen3.5-122b-a10b qwen122b-llm))
  VLLM_COMPOSE := -f docker/docker-compose.vllm-qwen122b.yaml
  VLLM_CONTAINER := revpropbench-vllm-qwen122b
else
  VLLM_COMPOSE := -f docker/docker-compose.vllm.yaml
  VLLM_CONTAINER := revpropbench-vllm
endif
RUNNER_COMPOSE := -f docker/docker-compose.runner.yaml
ANNOTATOR_COMPOSE := -f docker/docker-compose.annotator.yaml

# Annotation GUI host port (browser → http://localhost:$(ANNOTATOR_PORT)).
ANNOTATOR_PORT ?= 5001

# vLLM endpoints (host ports), matching each preset model's compose port.
VLLM_URL_GPT20B    ?= http://host.docker.internal:8001/v1
VLLM_MODEL_GPT20B  ?= local-llm
VLLM_URL_QWEN9B    ?= http://host.docker.internal:8003/v1
VLLM_MODEL_QWEN9B  ?= qwen9b-llm
VLLM_URL_GPT120B   ?= http://host.docker.internal:8004/v1
VLLM_MODEL_GPT120B ?= gptoss120-llm
VLLM_URL_QWEN27B   ?= http://host.docker.internal:8005/v1
VLLM_MODEL_QWEN27B ?= qwen27b-llm
VLLM_URL_QWEN122B  ?= http://host.docker.internal:8002/v1
VLLM_MODEL_QWEN122B ?= qwen122b-llm

# Default benchmark paths.
BENCH_SAMPLES   ?= benchmark/data/samples
# Default to the 120-case test split (the paper's evaluation set; materialize
# it first with `make split-dataset`). Use benchmark/data/annotated for all 150
# cases or benchmark/data/annotated_dev for the 30-case dev split.
BENCH_ANNOTATED ?= benchmark/data/annotated_test
BENCH_LEAVES    ?= 8
BENCH_SEED      ?= 42
BENCH_TEMP      ?= 0.6
BENCH_CONCURRENCY ?= 8
# Override the eval vLLM endpoint (e.g. a second gpt-oss-20b instance on :8003
# for parallel multi-seed runs); empty = use the per-MODEL default endpoint.
BENCH_EVAL_URL  ?=
# empty = all registered methods; else a space-separated subset,
# e.g. BENCH_METHODS="sub_hist_json sub_json"
BENCH_METHODS   ?=
# empty = provider default; else low/medium/high, forwarded as the OpenAI-style
# reasoning_effort (part of the cache key) for reasoning models (e.g. gpt-5.x).
BENCH_REASONING_EFFORT ?=
# empty = all domains; else a single domain e.g. shopping_cart
BENCH_DOMAIN    ?=

# Official dataset generation (one sample per skeleton, deterministic ids).
DATASET_LEAVES      ?= 50
DATASET_SEED        ?= 42
DATASET_MODEL       ?= openai/gpt-5.5
DATASET_OUT         ?= benchmark/data/raw
DATASET_CONCURRENCY ?= 30
# empty = all domains; else a single domain e.g. data_pipeline
DATASET_DOMAIN      ?=

# Default run dir: the unified layout the figure scripts fall back to, so a
# plain `make bench-eval MODEL=<m> BENCH_SEED=<s>` feeds `make figures` directly
# (the final dir name becomes final/seed<seed>-<MODEL>).
BENCH_RUN_DIR ?= results/bench/final/seed$(BENCH_SEED)

# Convenience macro: run a shell command inside the revpropbench-runner container.
DOCKER_EXEC := docker exec revpropbench-runner bash -c

help:
	@echo "Infra:"
	@echo "  make env              # copy docker/.env.example -> docker/.env if missing"
	@echo "  make vllm-up [MODEL=gpt-oss-20b|gpt-oss-120b|qwen3.5-9b|qwen3.5-27b|qwen3.5-122b-a10b]  # start a local vLLM server (preset model; else docker/.env)"
	@echo "  make vllm-down [MODEL=...]   # stop that vLLM server (use the same MODEL)"
	@echo "  make smoke-llm        # curl vLLM /v1 endpoints"
	@echo "  make sync             # cooldown-check + uv sync --frozen"
	@echo "  make cooldown-check   # 7-day supply-chain age guard"
	@echo "  make refresh-lock     # uv lock --upgrade"
	@echo "  make runner-build     # build the runner image"
	@echo "  make runner-up        # start runner container in background"
	@echo "  make eval-shell       # exec a bash shell inside the runner"
	@echo "  make runner-down      # stop runner container"
	@echo "  make annotator-up     # start annotation GUI (http://localhost:$(ANNOTATOR_PORT))"
	@echo "  make annotator-logs   # follow annotation GUI logs"
	@echo "  make annotator-down   # stop annotation GUI"
	@echo "  make stop             # vllm-down + runner-down"
	@echo ""
	@echo "Benchmark:"
	@echo "  make dataset [DATASET_LEAVES=50 DATASET_DOMAIN=data_pipeline DATASET_SEED=42 DATASET_MODEL=openai/gpt-5.5 DATASET_OUT=benchmark/data/raw]"
	@echo "                          # runner-up + generate one sample per skeleton (deterministic ids)"
	@echo "  make bench-sample [BENCH_LEAVES=\"10 50\" BENCH_DOMAIN=shopping_cart MODEL=gpt-oss-20b]  # ad-hoc sampling via a chosen model"
	@echo "  make annotator-up       # annotate in the GUI"
	@echo "  make bench-eval MODEL=gpt-oss-20b [BENCH_SEED=42 BENCH_TEMP=0.6 BENCH_CONCURRENCY=8 BENCH_ANNOTATED=... BENCH_RUN_DIR=... BENCH_EVAL_URL=http://host.docker.internal:8003/v1 BENCH_METHODS=\"sub_hist_json sub_json\"]"
	@echo "                          # eval annotated cases (BENCH_EVAL_URL overrides the endpoint, e.g. a 2nd gpt-oss-20b instance for parallel multi-seed)"
	@echo "  make split-dataset [CACHE=true]  # materialize benchmark/data/annotated_{dev,test}/ (CACHE=true: copy from the tracked split, no search)"
	@echo "  make extract-paper-runs # extract the paper's evaluation outputs into results/bench/final/"
	@echo "  make figures            # regenerate the paper figures from results/bench/ runs"
	@echo ""
	@echo "Vars: DATASET_LEAVES=$(DATASET_LEAVES) DATASET_DOMAIN=$(DATASET_DOMAIN) DATASET_MODEL=$(DATASET_MODEL) DATASET_OUT=$(DATASET_OUT)"
	@echo "      BENCH_SAMPLES=$(BENCH_SAMPLES) BENCH_ANNOTATED=$(BENCH_ANNOTATED) BENCH_RUN_DIR=$(BENCH_RUN_DIR)"

env:
	@test -f $(ENV_FILE) || cp docker/.env.example $(ENV_FILE)
	@echo "wrote $(ENV_FILE)"

#---------------- local LLM stack ----------------
vllm-up:
	docker compose $(VLLM_COMPOSE) up -d
	@echo "vLLM starting; logs: docker logs -f $(VLLM_CONTAINER)"

vllm-down:
	-docker compose $(VLLM_COMPOSE) down

smoke-llm:
	./scripts/smoke_llm.sh

#---------------- supply-chain cooldown ----------------
cooldown-check:
	bash scripts/cooldown_age_check.sh

sync: cooldown-check
	uv sync --frozen

refresh-lock:
	uv lock --upgrade
	bash scripts/cooldown_age_check.sh

refresh-lock-now:
	@echo "[refresh-lock-now] WARNING: bypassing 1-week cooldown."
	uv lock --upgrade --exclude-newer "$$(date -u +%Y-%m-%dT%H:%M:%SZ)"

#---------------- runner container ----------------
runner-build:
	docker compose $(RUNNER_COMPOSE) build

runner-up: runner-build
	docker compose $(RUNNER_COMPOSE) up -d

eval-shell:
	docker compose $(RUNNER_COMPOSE) exec runner bash

runner-down:
	-docker compose $(RUNNER_COMPOSE) down

#---------------- annotation GUI ----------------
annotator-up:
	ANNOTATOR_PORT=$(ANNOTATOR_PORT) docker compose $(ANNOTATOR_COMPOSE) up -d
	@echo "Annotator GUI → http://localhost:$(ANNOTATOR_PORT)"

annotator-logs:
	docker compose $(ANNOTATOR_COMPOSE) logs -f

annotator-down:
	-docker compose $(ANNOTATOR_COMPOSE) down

#---------------- benchmark: model selection (litellm) ----------------
# Convenience names map to a local vLLM endpoint; any other MODEL value is
# passed to litellm verbatim (e.g. MODEL=openai/gpt-5.5 → real OpenAI). Hosted
# keys (OPENAI_API_KEY / ANTHROPIC_API_KEY) come from docker/.env, loaded into
# the runner container via env_file (re-run `make runner-up` after editing).
ifeq ($(MODEL),gpt-oss-20b)
  _EVAL_URL := $(VLLM_URL_GPT20B)
  _EVAL_NAME := $(VLLM_MODEL_GPT20B)
else ifneq (,$(filter $(MODEL),qwen3.5-9b qwen9b-llm))
  _EVAL_URL := $(VLLM_URL_QWEN9B)
  _EVAL_NAME := $(VLLM_MODEL_QWEN9B)
else ifneq (,$(filter $(MODEL),qwen3.5-27b qwen27b-llm))
  _EVAL_URL := $(VLLM_URL_QWEN27B)
  _EVAL_NAME := $(VLLM_MODEL_QWEN27B)
else ifneq (,$(filter $(MODEL),qwen3.5-122b-a10b qwen122b-llm))
  _EVAL_URL := $(VLLM_URL_QWEN122B)
  _EVAL_NAME := $(VLLM_MODEL_QWEN122B)
else ifeq ($(MODEL),gpt-oss-120b)
  _EVAL_URL := $(VLLM_URL_GPT120B)
  _EVAL_NAME := $(VLLM_MODEL_GPT120B)
else ifeq ($(MODEL),gpt-5.4-mini)
  # Hosted OpenAI. Force the openai/ prefix so complete() routes to the real
  # provider: a bare "gpt-5.4-mini" has no "/" and would be treated as a local
  # vLLM served name and sent to VLLM_BASE_URL (the dead :8001), giving a
  # "Connection error". The run-dir suffix stays "gpt-5.4-mini" (from MODEL);
  # the litellm cache key is unchanged (litellm prepends openai/ either way),
  # so already-evaluated cases still hit cache. Use BENCH_TEMP=1.0.
  _EVAL_URL :=
  _EVAL_NAME := openai/gpt-5.4-mini
else
  # literal litellm model string (hosted provider); no local api_base
  _EVAL_URL :=
  _EVAL_NAME := $(MODEL)
endif

#---------------- benchmark: official dataset generation ----------------
# Generate the benchmark dataset: one sample per cascade skeleton at each
# --leaves value, with deterministic ids {domain}_L{tl}_{NN}. Defaults to the
# hosted gpt-5.5 (needs OPENAI_API_KEY in docker/.env) and writes to
# benchmark/data/raw. Brings the runner container up first, so a single command
# stands up docker and runs generation end to end:
#   make dataset
#   make dataset DATASET_LEAVES=10 DATASET_OUT=benchmark/data/samples_tl10
#   make dataset DATASET_DOMAIN=data_pipeline DATASET_LEAVES=10
dataset: runner-up
	$(DOCKER_EXEC) "cd /work && \
	  python -m benchmark.sampling $(if $(DATASET_DOMAIN),--domain $(DATASET_DOMAIN),) \
	    --leaves $(DATASET_LEAVES) --seed $(DATASET_SEED) \
	    --model $(DATASET_MODEL) --out $(DATASET_OUT) \
	    --concurrency $(DATASET_CONCURRENCY)"

#---------------- benchmark: ad-hoc sample + eval ----------------
# Generate unannotated samples with a chosen MODEL (annotate them in the GUI
# afterwards). Like `dataset` but routes through the MODEL convenience mapping
# (local vLLM endpoints), and assumes the runner is already up.
bench-sample:
	$(DOCKER_EXEC) "cd /work && \
	  VLLM_BASE_URL=$(_EVAL_URL) \
	  python -m benchmark.sampling $(if $(BENCH_DOMAIN),--domain $(BENCH_DOMAIN),) \
	    --leaves $(BENCH_LEAVES) \
	    --seed $(BENCH_SEED) --model $(_EVAL_NAME) --out $(BENCH_SAMPLES)"

# Evaluate a model's cascade-aware revision on the annotated cases.
bench-eval: runner-up
	$(DOCKER_EXEC) "cd /work && \
	  VLLM_BASE_URL=$(if $(BENCH_EVAL_URL),$(BENCH_EVAL_URL),$(_EVAL_URL)) \
	  $(if $(BENCH_REASONING_EFFORT),LLM_REASONING_EFFORT=$(BENCH_REASONING_EFFORT),) \
	  python -m benchmark.evaluator --annotated $(BENCH_ANNOTATED) \
	    --out $(BENCH_RUN_DIR)-$(MODEL) --model $(_EVAL_NAME) \
	    --seed $(BENCH_SEED) --temperature $(BENCH_TEMP) \
	    --concurrency $(BENCH_CONCURRENCY)$(if $(BENCH_METHODS), --methods $(BENCH_METHODS),)"

#---------------- figures ----------------
# Regenerate the paper figures from the evaluated runs under results/bench/.
# Requires the runner container (make runner-up) and the run dirs to be present.
# Materialize the dev/test split dirs (annotated_dev/ 30, annotated_test/ 120).
# By default this re-runs the balanced split search (deterministic, seed 42) and
# rewrites the split definition (benchmark/data/split_dev30_test120.json).
# With CACHE=true it skips the search and just copies the case files listed in
# the tracked split definition.
split-dataset: runner-up
	$(DOCKER_EXEC) "cd /work && python -m benchmark.split \
	  $(if $(filter true True TRUE 1,$(CACHE)),--apply,) \
	  --out-dev benchmark/data/annotated_dev --out-test benchmark/data/annotated_test"

# Extract the paper's evaluation outputs (results/paper_runs/*.tar.gz) into
# results/bench/final/, where `make figures` reads from. Explicit on purpose:
# extracted paper runs take precedence over same-named user runs.
extract-paper-runs:
	@mkdir -p results/bench
	@for f in results/paper_runs/*.tar.gz; do \
	  echo "extracting $$f"; tar -xzf $$f -C results/bench/; done
	@echo "paper runs extracted to results/bench/final/"

figures: runner-up
	$(DOCKER_EXEC) "cd /work && \
	  mkdir -p results/figures && \
	  python scripts/plot_benchmark_stats_bc.py && \
	  python scripts/plot_success_grid.py && \
	  python scripts/plot_failure_analysis.py && \
	  python scripts/plot_cost_grid.py"
	@echo "figures written to results/figures/"

#---------------- tests & lint ----------------
test:
	$(DOCKER_EXEC) "cd /work && python -m pytest -q --ignore=old"

lint:
	uv run ruff check .
	uv run ruff format --check .

stop: vllm-down runner-down
