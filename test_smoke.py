"""Smoke test: verify the agent loop, parser, tool execution, and trace
serialization work -- using a scripted fake LLM, so no API key is needed.

Run:  python test_smoke.py
"""
from __future__ import annotations

import json

from agent import run_react_agent
from tasks import TASKS


class FakeLLM:
    """Returns canned ReAct steps in order, ignoring the actual prompt."""

    def __init__(self, script: list[str]):
        self.script = script
        self.i = 0

    def __call__(self, messages, stop=None) -> str:
        out = self.script[self.i]
        self.i += 1
        return out


def test_single_tool():
    fake = FakeLLM([
        'Thought: I should look up the order.\nAction: order_lookup\nAction Input: {"order_id": "ORD1002"}',
        "Thought: I have the status now.\nFinal Answer: Your order ORD1002 is In Transit.",
    ])
    trace = run_react_agent(TASKS[0], llm_fn=fake)

    assert trace.stopped_reason == "final_answer"
    assert trace.final_answer and "In Transit" in trace.final_answer
    assert trace.steps[0].action.tool_name == "order_lookup"
    assert "In Transit" in trace.steps[0].observation
    # round-trips cleanly to/from JSON
    json.dumps(trace.model_dump())
    print("PASS  single-tool trajectory")


def test_multi_tool():
    fake = FakeLLM([
        'Thought: Check the order first.\nAction: order_lookup\nAction Input: {"order_id": "ORD1003"}',
        'Thought: It was cancelled, check refund.\nAction: refund_status\nAction Input: {"order_id": "ORD1003"}',
        "Thought: Done.\nFinal Answer: ORD1003 was cancelled; Refund Initiated, expected in 5 days.",
    ])
    trace = run_react_agent(TASKS[3], llm_fn=fake)

    tools_used = [s.action.tool_name for s in trace.steps if s.action]
    assert tools_used == ["order_lookup", "refund_status"]
    assert trace.stopped_reason == "final_answer"
    print("PASS  multi-tool trajectory")


def test_bad_format_then_recover():
    fake = FakeLLM([
        "I think I'll just look it up.",  # malformed: no Action / Final Answer
        'Thought: Use the right format.\nAction: order_lookup\nAction Input: {"order_id": "ORD1002"}',
        "Thought: Got it.\nFinal Answer: In Transit.",
    ])
    trace = run_react_agent(TASKS[0], llm_fn=fake)

    assert trace.steps[0].error is not None      # first step recorded as a parse error
    assert trace.steps[1].action.tool_name == "order_lookup"
    assert trace.stopped_reason == "final_answer"
    print("PASS  malformed-step recovery")


if __name__ == "__main__":
    test_single_tool()
    test_multi_tool()
    test_bad_format_then_recover()
    print("\nAll smoke tests passed.")
