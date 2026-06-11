"""LLM-as-judge evaluators (Day 3).

The deterministic layer catches structural failures for free. These three
judges handle the *semantic* calls it can't:

  1. tool_appropriateness - was each tool call actually warranted, given the
                            goal and what was already known? (catches the
                            "correct answer, pointless extra calls" case)
  2. faithfulness         - is every claim in the final answer backed by a
                            tool observation, or did the agent make something
                            up? (hallucination detection)
  3. goal_completion      - ignoring wording, did the answer truly satisfy the
                            user's goal? (resolves brittle substring matching)

Each judge returns a CheckResult (same type as the deterministic layer) so
the two sets merge trivially on Day 4. Each judge accepts an injectable
`llm_fn`, so tests can run with a fake judge and no API key.

This is the direct extension of your LLM Evaluator project: same LLM-as-judge
pattern, but scoring a multi-step trajectory instead of a single response.
"""
from __future__ import annotations

import json
import re
from typing import Callable

from evaluators import CheckResult, FAIL, PASS, WARN
from llm import groq_llm
from schema import Trace
from tools import tools_description


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _extract_json(text: str) -> dict:
    """Pull the first {...} block out of a model response and parse it.
    Robust to code fences and preamble."""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if not m:
        raise ValueError(f"no JSON object in judge output: {text[:120]!r}")
    return json.loads(m.group(0))


def _render(trace: Trace) -> str:
    """Human-readable trajectory for the judge prompt."""
    lines = [f"User goal: {trace.user_goal}", "", "Trajectory:"]
    for s in trace.steps:
        if s.action is not None:
            lines.append(f"  Step {s.index}: thought={s.thought!r}")
            lines.append(f"           tool={s.action.tool_name}  args={s.action.args}")
            lines.append(f"           observation={s.observation!r}")
        else:
            lines.append(f"  Step {s.index}: (final) thought={s.thought!r}")
    lines.append("")
    lines.append(f"Final answer: {trace.final_answer!r}")
    return "\n".join(lines)


def _ask(llm_fn: Callable, system: str, user: str) -> dict:
    """Call the judge LLM and parse its JSON, degrading gracefully on error."""
    raw = llm_fn(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        stop=None,
    )
    return _extract_json(raw)


def _verdict_to_status(v: str) -> str:
    return {"pass": PASS, "partial": WARN, "warn": WARN, "fail": FAIL}.get(
        (v or "").lower(), WARN
    )


# --------------------------------------------------------------------------
# judge 1: tool appropriateness
# --------------------------------------------------------------------------
SYS_TOOL = """You are a strict evaluator of AI agent behaviour. You judge whether each tool call an agent made was appropriate and NECESSARY, given the user's goal and what was already known at that point in the run.

A tool call is INAPPROPRIATE if: the needed information was already available from an earlier step, the call does not help achieve the goal, or a different tool was clearly the right choice.

Respond ONLY with JSON of this exact shape:
{"steps": [{"index": <int>, "appropriate": <true|false>, "reason": "<short>"}], "summary": "<one sentence>"}"""


def judge_tool_appropriateness(trace: Trace, llm_fn: Callable = groq_llm) -> CheckResult:
    tool_steps = [s for s in trace.steps if s.action is not None]
    if not tool_steps:
        return CheckResult("tool_appropriateness", PASS, "No tool calls to judge.")
    user = (
        f"Available tools:\n{tools_description()}\n\n{_render(trace)}\n\n"
        "Judge each tool-calling step."
    )
    try:
        data = _ask(llm_fn, SYS_TOOL, user)
    except Exception as e:  # noqa: BLE001
        return CheckResult("tool_appropriateness", WARN, f"Judge error: {e}")
    bad = [s for s in data.get("steps", []) if s.get("appropriate") is False]
    if bad:
        idxs = [s.get("index") for s in bad]
        reason = bad[0].get("reason", "")
        return CheckResult("tool_appropriateness", WARN,
                           f"{len(bad)} unnecessary/incorrect tool call(s). e.g. {reason}",
                           [i for i in idxs if isinstance(i, int)])
    return CheckResult("tool_appropriateness", PASS, "Every tool call was warranted.")


# --------------------------------------------------------------------------
# judge 2: faithfulness (hallucination detection)
# --------------------------------------------------------------------------
SYS_FAITH = """You check whether an AI agent's final answer is fully supported by the tool observations in its trajectory. Any factual claim in the answer that is NOT backed by an observation is a faithfulness violation (a hallucination). General politeness or hedging is fine; specific facts must be grounded.

Respond ONLY with JSON of this exact shape:
{"verdict": "pass"|"fail", "unsupported_claims": ["<claim>", ...], "reasoning": "<short>"}"""


def judge_faithfulness(trace: Trace, llm_fn: Callable = groq_llm) -> CheckResult:
    if not (trace.final_answer or "").strip():
        return CheckResult("faithfulness", PASS, "No answer to check.")
    try:
        data = _ask(llm_fn, SYS_FAITH, _render(trace))
    except Exception as e:  # noqa: BLE001
        return CheckResult("faithfulness", WARN, f"Judge error: {e}")
    unsupported = data.get("unsupported_claims", []) or []
    if _verdict_to_status(data.get("verdict")) == FAIL or unsupported:
        claims = "; ".join(str(c) for c in unsupported[:2])
        return CheckResult("faithfulness", FAIL,
                           f"Answer contains unsupported claim(s): {claims}")
    return CheckResult("faithfulness", PASS, "Answer is fully grounded in observations.")


# --------------------------------------------------------------------------
# judge 3: goal completion (semantic)
# --------------------------------------------------------------------------
SYS_GOAL = """You judge whether an AI agent's final answer correctly and completely satisfies the user's goal. Judge substance, not wording -- a correct answer phrased differently from any expected text still passes. If no answer was given, it fails. If it answers partially, say "partial".

Respond ONLY with JSON of this exact shape:
{"verdict": "pass"|"partial"|"fail", "reasoning": "<short>"}"""


def judge_goal_completion(trace: Trace, llm_fn: Callable = groq_llm) -> CheckResult:
    if not (trace.final_answer or "").strip():
        return CheckResult("goal_completion", FAIL, "No answer was produced.")
    try:
        data = _ask(llm_fn, SYS_GOAL, _render(trace))
    except Exception as e:  # noqa: BLE001
        return CheckResult("goal_completion", WARN, f"Judge error: {e}")
    status = _verdict_to_status(data.get("verdict"))
    return CheckResult("goal_completion", status,
                       data.get("reasoning", "")[:200] or "Judged goal completion.")


JUDGES: list[Callable[[Trace, Callable], CheckResult]] = [
    judge_tool_appropriateness,
    judge_faithfulness,
    judge_goal_completion,
]


def judge_trace(trace: Trace, llm_fn: Callable = groq_llm) -> list[CheckResult]:
    """Run all three LLM judges on one trace."""
    return [j(trace, llm_fn) for j in JUDGES]
