"""benchmark/: human-grounded JSON-Patch cascade-revision benchmark.

Pipeline:
  1. sampling.py  — LLM generates an unannotated case (conversation, JSON,
                    single-target edit instruction).
  2. annotation.py — human annotates expected_patch +
                     must_not_touch on top of the LLM-sampled case.
"""
