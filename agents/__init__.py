"""Shared agent runtime — vLLM /v1/chat/completions direct.

Only the LLM client is exposed now; the legacy session/core harness moved to
`old/v2/agents/` when the project pivoted to the benchmark/ pipeline.
"""

from agents.llm import CompletionUsage, complete, configure_local_llm

__all__ = [
    # LLM client (the only piece the benchmark/ pipeline needs)
    "complete",
    "CompletionUsage",
    "configure_local_llm",
]
