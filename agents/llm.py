"""LLM client: a thin async wrapper over ``litellm.acompletion``.

A single entry point (`complete`) reaches OpenAI / local vLLM / Ollama; the
backend is chosen from the model string (see `complete` for the convention).
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import litellm

litellm.drop_params = True          # silently drop provider-unsupported params
litellm.suppress_debug_info = True  # quiet per-call banner logging

DEFAULT_VLLM_URL = "http://localhost:8001/v1"
DEFAULT_MODEL = "local-llm"

# Disk-backed response cache (keyed on the full request: model + messages +
# params incl. seed). Speeds up re-runs and makes seeded generation free to
# replay. Opt out with LITELLM_CACHE=0.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".litellm_cache"
# litellm does not forward a size limit to diskcache, whose 1 GiB default
# silently LRU-evicts entries once exceeded (observed: evicted samples got
# regenerated on re-runs, breaking byte-identity between runs).
_CACHE_SIZE_LIMIT = 50 * 2**30


def _make_disk_cache():
    from litellm import Cache

    cache = Cache(type="disk", disk_cache_dir=str(_CACHE_DIR))
    cache.cache.disk_cache.reset("size_limit", _CACHE_SIZE_LIMIT)
    return cache


if os.environ.get("LITELLM_CACHE", "1") != "0":
    try:
        litellm.cache = _make_disk_cache()
    except Exception:  # noqa: BLE001 — caching is best-effort, never fatal
        pass


def clear_llm_cache() -> bool:
    """Delete the on-disk LLM response cache. Returns True if it existed."""
    import shutil

    if _CACHE_DIR.is_dir():
        shutil.rmtree(_CACHE_DIR, ignore_errors=True)
        if litellm.cache is not None:
            try:
                litellm.cache = _make_disk_cache()
            except Exception:  # noqa: BLE001
                pass
        return True
    return False


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip())


def configure_local_llm() -> dict[str, str]:
    """Resolve env-driven vLLM endpoint config + return it for callers.

    Priority for the base URL:
      1. VLLM_BASE_URL              (explicit)
      2. host.docker.internal:VLLM_HOST_PORT  (when running in the runner container)
      3. localhost:VLLM_HOST_PORT   (when running on the host)
      4. DEFAULT_VLLM_URL           (fallback)

    This only resolves the LOCAL vLLM endpoint + default served name; hosted
    providers (OpenAI/etc.) are reached by passing a provider-prefixed model
    to `complete`, which litellm routes on its own.
    """
    project_env = Path(__file__).resolve().parent.parent / "docker" / ".env"
    _load_dotenv(project_env)

    explicit = os.environ.get("VLLM_BASE_URL")
    if explicit:
        base = explicit
    else:
        port = os.environ.get("VLLM_HOST_PORT", "8001")
        # If we're inside a Linux container that has /.dockerenv, prefer host.docker.internal.
        host = "host.docker.internal" if Path("/.dockerenv").exists() else "localhost"
        base = f"http://{host}:{port}/v1"
    if not base.rstrip("/").endswith("/v1"):
        base = base.rstrip("/") + "/v1"
    model = os.environ.get("VLLM_SERVED_NAME") or DEFAULT_MODEL
    out = {"VLLM_BASE_URL": base, "VLLM_MODEL": model}
    for k, v in out.items():
        os.environ[k] = v
    return out


@dataclass
class CompletionUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0        # If reported separately by the provider
                                     # (OpenAI o-series, gpt-oss via vLLM with
                                     # openai_gptoss parser, etc.); 0 otherwise.
    finish_reason: str | None = None
    reasoning: str | None = None     # Qwen3 think content, stripped from `content`
    raw_content: str | None = None   # Original content (before think-strip), for debugging


def _split_qwen3_thinking(text: str, *, enable_thinking: bool) -> tuple[str, str | None]:
    """Strip Qwen3 reasoning from assistant content.

    Qwen3.6 with `enable_thinking=True` emits the response in two parts:
      <thinking content...>\\n</think>\\n\\n<final answer>

    Note: there is NO opening `<think>` tag — vLLM's chat template emits the
    `<think>` token only on the input side, so the assistant content starts
    raw and ends with `</think>` followed by the visible answer. We therefore
    take everything AFTER the LAST `</think>` as the answer, and everything
    before it (including the marker) as the reasoning trace.

    When thinking is enabled but NO `</think>` is found, the model was almost
    certainly truncated mid-think (hit max_tokens before reaching the answer).
    In that case we return an empty answer + the whole text as reasoning so
    the giant think trace doesn't pollute the multi-turn message history.
    """
    if not enable_thinking:
        return text, None
    if "</think>" not in text:
        return "", text
    head, _sep, tail = text.rpartition("</think>")
    return tail.strip(), head.strip()


async def complete(
    *,
    messages: list[dict[str, Any]],
    model: str | None = None,
    api_base: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = 30000,
    enable_thinking: bool | None = True,
    reasoning_effort: str | None = None,
    timeout_s: float = 1200.0,
    seed: int | None = None,
) -> tuple[str, CompletionUsage]:
    """Single chat completion via litellm, across OpenAI / local vLLM / Ollama.

    Model-string convention (litellm provider routing):
      - bare name (e.g. "local-llm", "qwen-llm") → treated as a local vLLM
        served name: routed as ``openai/<name>`` against ``VLLM_BASE_URL``
        with a throwaway key.
      - provider-prefixed (e.g. "openai/gpt-5.5", "ollama/llama3") → routed by
        litellm to that provider; the key comes from ``api_key`` if given,
        else litellm's own env lookup (``OPENAI_API_KEY`` etc.).

    `enable_thinking` toggles the vLLM-only ``chat_template_kwargs`` reasoning
    switch; it is sent (via ``extra_body``) ONLY to self-hosted endpoints
    (those with an ``api_base``), never to hosted OpenAI. Pass ``None`` to omit
    it entirely. The reasoning trace is stripped from the returned answer and
    kept on ``CompletionUsage.reasoning``.
    """
    cfg = configure_local_llm()
    m = model or cfg["VLLM_MODEL"]
    if "/" not in m:
        # bare served name → local vLLM via the OpenAI-compatible endpoint
        litellm_model = f"openai/{m}"
        resolved_base = api_base or cfg["VLLM_BASE_URL"]
        key = api_key or "EMPTY"  # vLLM ignores the key but the client needs one
    else:
        litellm_model = m
        resolved_base = api_base  # None → litellm uses the provider's default host
        key = api_key  # None → litellm reads the provider's env key (OPENAI_API_KEY, ...)

    kwargs: dict[str, Any] = {
        "model": litellm_model,
        "messages": messages,
        "temperature": temperature,
        "timeout": timeout_s,
    }
    if max_tokens is not None:
        kwargs["max_tokens"] = max_tokens
    if resolved_base:
        kwargs["api_base"] = resolved_base
    if key:
        kwargs["api_key"] = key
    if seed is not None:
        kwargs["seed"] = seed  # litellm forwards as the OpenAI-style `seed`
    if litellm.cache is not None:
        kwargs["caching"] = True  # a global litellm.cache alone is not enough

    # Provider-specific extras go in extra_body (forwarded verbatim, not
    # filtered by drop_params). The vLLM thinking switch only makes sense for
    # self-hosted endpoints, so gate it on having an api_base.
    extra_body: dict[str, Any] = {}
    if enable_thinking is not None and resolved_base:
        extra_body["chat_template_kwargs"] = {"enable_thinking": enable_thinking}
    if extra_body:
        kwargs["extra_body"] = extra_body

    # gpt-oss reasoning effort (low/medium/high). Forwarded as the OpenAI-style
    # top-level `reasoning_effort` so it is part of the cache key. An env default
    # lets a whole run opt in without threading the param through every call site.
    effort = reasoning_effort or os.getenv("LLM_REASONING_EFFORT")
    if effort:
        kwargs["reasoning_effort"] = effort

    resp = await litellm.acompletion(**kwargs)
    choice = resp.choices[0]
    msg = choice.message
    raw_text = msg.content or ""

    # Reasoning models surface the think trace differently. gpt-oss / hosted
    # OpenAI expose it via a separate `reasoning_content` field (handled first).
    # Qwen3 on vLLM instead emits it INLINE, terminated by `</think>`, with no
    # separate field — so the inline-split heuristic is applied ONLY for the
    # Qwen family with thinking enabled. Any other model never emits `</think>`,
    # and applying the split would swallow the whole answer as "reasoning".
    server_reasoning = getattr(msg, "reasoning_content", None) or getattr(msg, "reasoning", None)
    is_qwen = "qwen" in litellm_model.lower()
    if server_reasoning:
        answer, reasoning = raw_text, server_reasoning
    elif is_qwen and enable_thinking:
        answer, reasoning = _split_qwen3_thinking(raw_text, enable_thinking=True)
    else:
        answer, reasoning = raw_text, None

    usage = getattr(resp, "usage", None)
    # Reasoning models surface a token breakdown via
    # `completion_tokens_details.reasoning_tokens` (OpenAI-style). vLLM with
    # `openai_gptoss` reasoning parser forwards it the same way. If the
    # provider doesn't expose it, fall back to 0.
    details = getattr(usage, "completion_tokens_details", None)
    reasoning_tokens = int(getattr(details, "reasoning_tokens", 0) or 0) if details else 0
    return answer, CompletionUsage(
        input_tokens=int(getattr(usage, "prompt_tokens", 0) or 0),
        output_tokens=int(getattr(usage, "completion_tokens", 0) or 0),
        reasoning_tokens=reasoning_tokens,
        finish_reason=getattr(choice, "finish_reason", None),
        reasoning=reasoning,
        raw_content=raw_text,
    )
