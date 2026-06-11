"""The trace schema: one clean data structure for an agent run.

Everything downstream (the deterministic checks and the LLM-as-judge
evaluators you build on Days 2-3) reads a `Trace`. Getting this right is the
whole foundation, so keep it stable.

A run is: a user goal -> an ordered list of steps -> a final answer.
Each step is the classic ReAct unit: a thought, an action (tool call), and
the observation (tool result) that came back.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field


class ToolCall(BaseModel):
    tool_name: str
    args: dict[str, Any] = Field(default_factory=dict)


class Step(BaseModel):
    index: int
    thought: str = ""
    # action is None when the step is a final answer or a parse failure
    action: Optional[ToolCall] = None
    observation: Optional[str] = None
    # set when the model's output could not be parsed or the tool errored
    error: Optional[str] = None


class Trace(BaseModel):
    trace_id: str
    task_id: str
    user_goal: str
    steps: list[Step] = Field(default_factory=list)
    final_answer: Optional[str] = None
    # "final_answer" | "max_steps" | "error"
    stopped_reason: str = "unknown"
    model: str = ""
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    # gold/expected info copied from the task, used by the evaluator later
    expected: dict[str, Any] = Field(default_factory=dict)
