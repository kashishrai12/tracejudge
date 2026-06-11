"""Deterministic evaluators (Day 2).

Each check is a pure function: it reads a Trace and returns a CheckResult.
No LLM, no network -- fast and free. These catch the structural failures:
runs that never finish, wrong/extra tool calls, wasteful paths, repeated
actions, and figures in the answer that no tool ever returned.

The subtler, semantic judgments (was this the *appropriate* tool? did the
model hallucinate an observation's meaning?) are deliberately left for the
LLM-as-judge layer on Day 3. Keeping the cheap checks separate is the same
tiered idea as your log-classifier: let the free layer handle what it can.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from schema import Trace
from tools import TOOLS

PASS, WARN, FAIL = "pass", "warn", "fail"
# severity ordering so we can compute an overall verdict
_RANK = {PASS: 0, WARN: 1, FAIL: 2}


@dataclass
class CheckResult:
    name: str
    status: str           # PASS | WARN | FAIL
    message: str
    steps: list[int] = field(default_factory=list)   # implicated step indices

    def to_dict(self) -> dict:
        return {"name": self.name, "status": self.status,
                "message": self.message, "steps": self.steps}


# --------------------------------------------------------------------------
# individual checks
# --------------------------------------------------------------------------
def _tool_steps(trace: Trace):
    """(index, ToolCall) for every step that actually called a tool."""
    return [(s.index, s.action) for s in trace.steps if s.action is not None]


def check_termination(trace: Trace) -> CheckResult:
    """Did the run end with a real answer, or die without one?"""
    if trace.stopped_reason == "final_answer" and (trace.final_answer or "").strip():
        return CheckResult("termination", PASS, "Run ended with a final answer.")
    if trace.stopped_reason == "max_steps":
        return CheckResult("termination", FAIL,
                           "Hit the step limit without producing an answer.")
    return CheckResult("termination", FAIL,
                       f"Run stopped ('{trace.stopped_reason}') with no usable answer.")


def check_tool_validity(trace: Trace) -> CheckResult:
    """Every tool call must name a real tool and supply sane arguments."""
    bad_tool, bad_args = [], []
    for idx, call in _tool_steps(trace):
        if call.tool_name not in TOOLS:
            bad_tool.append(idx)
            continue
        valid_keys = set(TOOLS[call.tool_name]["args"].keys())
        provided = set(call.args.keys())
        if provided - valid_keys:               # unknown argument
            bad_args.append(idx)
        if valid_keys - provided:               # missing expected argument
            bad_args.append(idx)
    if bad_tool:
        return CheckResult("tool_validity", FAIL,
                           "Called a tool that does not exist.", sorted(set(bad_tool)))
    if bad_args:
        return CheckResult("tool_validity", WARN,
                           "Tool call(s) had missing or unexpected arguments.",
                           sorted(set(bad_args)))
    return CheckResult("tool_validity", PASS, "All tool calls are valid.")


def check_expected_tools(trace: Trace) -> CheckResult:
    """Compare tools used against the task's gold tool set."""
    expected = trace.expected.get("tools")
    if not expected:
        return CheckResult("expected_tools", PASS, "No gold tool set specified.")
    used_order = []
    for _, call in _tool_steps(trace):
        if call.tool_name not in used_order:
            used_order.append(call.tool_name)
    used, exp = set(used_order), set(expected)
    missing, extra = exp - used, used - exp
    if missing:
        return CheckResult("expected_tools", FAIL,
                           f"Did not call required tool(s): {sorted(missing)}.")
    if extra:
        return CheckResult("expected_tools", WARN,
                           f"Called unnecessary tool(s): {sorted(extra)} (over-calling).")
    return CheckResult("expected_tools", PASS, "Used exactly the expected tools.")


def check_step_efficiency(trace: Trace) -> CheckResult:
    """Flag trajectories that take more tool calls than the task warrants."""
    n = len(_tool_steps(trace))
    expected = trace.expected.get("tools")
    budget = len(expected) if expected else 2
    ceiling = budget + 1                          # one step of slack -> WARN above this
    hard_ceiling = 2 * budget + 2                 # clearly broken -> FAIL above this
    if n > hard_ceiling:
        return CheckResult("step_efficiency", FAIL,
                           f"{n} tool calls for a ~{budget}-step task (severely wasteful).")
    if n > ceiling:
        return CheckResult("step_efficiency", WARN,
                           f"{n} tool calls for a ~{budget}-step task (wasteful path).")
    return CheckResult("step_efficiency", PASS, f"{n} tool call(s) -- efficient.")


def check_redundant_actions(trace: Trace) -> CheckResult:
    """Detect the exact same tool+args being called more than once."""
    seen, repeats = {}, []
    for idx, call in _tool_steps(trace):
        key = (call.tool_name, tuple(sorted(call.args.items())))
        if key in seen:
            repeats.append(idx)
        else:
            seen[key] = idx
    if repeats:
        return CheckResult("redundant_actions", WARN,
                           "Repeated an identical tool call.", sorted(repeats))
    return CheckResult("redundant_actions", PASS, "No repeated calls.")


def check_answer_grounding(trace: Trace) -> CheckResult:
    """Cheap hallucination proxy: every number in the answer should appear
    in some observation. A figure that came from nowhere is a red flag."""
    if not (trace.final_answer or "").strip():
        return CheckResult("answer_grounding", PASS, "No answer to check.")
    obs_text = " ".join(s.observation or "" for s in trace.steps)
    obs_nums = set(re.findall(r"\d+", obs_text))
    ans_nums = set(re.findall(r"\d+", trace.final_answer))
    ungrounded = sorted(n for n in ans_nums if n not in obs_nums)
    if ungrounded:
        return CheckResult("answer_grounding", WARN,
                           f"Answer has figure(s) not in any tool output: {ungrounded} "
                           f"(possible fabrication).")
    return CheckResult("answer_grounding", PASS, "All figures trace to tool outputs.")


def check_outcome_match(trace: Trace) -> CheckResult:
    """Heuristic outcome check via the gold substring. Brittle on purpose --
    a WARN here often just means the wording differs, which is exactly the
    kind of call the Day 3 LLM judge will make properly."""
    needle = trace.expected.get("answer_contains")
    if not needle:
        return CheckResult("outcome_match", PASS, "No gold substring specified.")
    if not (trace.final_answer or "").strip():
        return CheckResult("outcome_match", FAIL, "No answer to match against gold.")
    if needle.lower() in trace.final_answer.lower():
        return CheckResult("outcome_match", PASS, f"Answer contains gold '{needle}'.")
    return CheckResult("outcome_match", WARN,
                       f"Answer does not literally contain '{needle}' "
                       f"(semantic check deferred to Day 3).")


CHECKS: list[Callable[[Trace], CheckResult]] = [
    check_termination,
    check_tool_validity,
    check_expected_tools,
    check_step_efficiency,
    check_redundant_actions,
    check_answer_grounding,
    check_outcome_match,
]


def evaluate_trace(trace: Trace) -> dict:
    """Run all checks and roll up an overall verdict."""
    results = [c(trace) for c in CHECKS]
    verdict = max((r.status for r in results), key=lambda s: _RANK[s])
    return {
        "trace_id": trace.trace_id,
        "task_id": trace.task_id,
        "verdict": verdict,
        "results": [r.to_dict() for r in results],
    }
