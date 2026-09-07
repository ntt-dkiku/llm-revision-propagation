#!/usr/bin/env bash
# Smoke-test the local vLLM server (OpenAI-compatible). Exits non-zero on
# first failure.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
ENV_FILE="${ROOT}/docker/.env"
if [[ -f "$ENV_FILE" ]]; then set -a; source "$ENV_FILE"; set +a; fi

: "${VLLM_HOST_PORT:=8001}"
: "${VLLM_SERVED_NAME:=local-llm}"

VLLM_URL="http://localhost:${VLLM_HOST_PORT}"

echo "[smoke] vLLM /v1/models on ${VLLM_URL}"
curl -fsS "${VLLM_URL}/v1/models" | head -c 400; echo

echo "[smoke] vLLM /v1/chat/completions"
curl -fsS "${VLLM_URL}/v1/chat/completions" \
  -H "Content-Type: application/json" \
  -d "{\"model\":\"${VLLM_SERVED_NAME}\",\"max_tokens\":32,\"messages\":[{\"role\":\"user\",\"content\":\"Say PONG.\"}]}" \
  | head -c 800; echo

echo "[smoke] OK"
