"""Unified evaluation pipeline (Day 4).

Merges the two layers into one report:
  - deterministic checks (evaluators.py)  -- free, instant, fully reproducible
  - LLM judges (judge.py)                 -- semantic, needs a key, ~3 calls

Because both layers return the same CheckResult type, merging is trivial.
`use_llm=False` runs the free layer only -- handy for the benchmark, where we
want a stable, reproducible number that costs nothing to re-run.
"""
from __future__ import annotations

from typing import Callable, Optional

from evaluators import CHECKS, FAIL, PASS, WARN, _RANK
from schema import Trace


def evaluate_full(trace: Trace, llm_fn: Optional[Callable] = None,
                  use_llm: bool = True) -> dict:
    results = [c(trace) for c in CHECKS]                 # deterministic layer
    if use_llm:
        from judge import judge_trace                    # imported lazily
        results += judge_trace(trace, llm_fn) if llm_fn else judge_trace(trace)

    verdict = max((r.status for r in results), key=lambda s: _RANK[s])
    fired = [r.name for r in results if r.status != PASS]
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "verdict": verdict,
        "fired_checks": fired,
        "results": [r.to_dict() for r in results],
    }
