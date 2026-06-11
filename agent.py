"""A minimal ReAct agent that emits a transparent, loggable trajectory.

Design choice: we use the classic *text* ReAct format (Thought / Action /
Action Input / Final Answer) rather than the model's native tool-calling
JSON. The text trace is easy to log, read, and -- crucially -- evaluate
step-by-step later. (The tradeoff: text parsing is a little more fragile
than native tool calls. For an evaluation harness, transparency wins.)

The loop:
  1. Show the model the question + everything so far (the "scratchpad").
  2. The model returns ONE step: either an Action, or a Final Answer.
  3. If it's an Action, run the tool, append the Observation, loop.
  4. Stop on Final Answer, on max_steps, or record a parse error and retry.
"""
from __future__ import annotations

import json
import re
import uuid
from typing import Callable

from llm import DEFAULT_MODEL, groq_llm
from schema import Step, ToolCall, Trace
from tools import run_tool, tools_description

SYSTEM_PROMPT = """You are a customer-support agent. Answer the user's question by reasoning step by step and using tools when needed.

You have access to these tools:
{tools}

Use EXACTLY this format, one step at a time:

Thought: <your reasoning about what to do next>
Action: <one tool name from the list above>
Action Input: <a JSON object with the tool's arguments>

After each Action you will be shown an Observation. Continue with more
Thought/Action steps as needed. When you have enough information, respond with:

Thought: <final reasoning>
Final Answer: <your answer to the user>

Output only ONE step at a time. Never write the Observation yourself."""

THOUGHT_RE = re.compile(r"Thought:\s*(.+)")
ACTION_RE = re.compile(r"Action:\s*(.+)")
INPUT_RE = re.compile(r"Action Input:\s*(\{.*\})", re.DOTALL)
FINAL_RE = re.compile(r"Final Answer:\s*(.+)", re.DOTALL)


def _parse(text: str):
    """Parse one model step into (thought, tool_name, args, final, error)."""
    thought_m = THOUGHT_RE.search(text)
    thought = thought_m.group(1).strip().splitlines()[0] if thought_m else ""

    final_m = FINAL_RE.search(text)
    if final_m:
        return thought, None, None, final_m.group(1).strip(), None

    action_m = ACTION_RE.search(text)
    if not action_m:
        return thought, None, None, None, "no Action or Final Answer found"
    tool_name = action_m.group(1).strip().strip("`").splitlines()[0].strip()

    input_m = INPUT_RE.search(text)
    if not input_m:
        return thought, tool_name, None, None, "no parsable Action Input JSON"
    try:
        args = json.loads(input_m.group(1))
    except json.JSONDecodeError as e:
        return thought, tool_name, None, None, f"invalid JSON in Action Input: {e}"
    return thought, tool_name, args, None, None


def run_react_agent(
    task: dict,
    llm_fn: Callable = groq_llm,
    model: str = DEFAULT_MODEL,
    max_steps: int = 6,
) -> Trace:
    """Run the agent on one task and return a fully populated Trace."""
    goal = task["goal"]
    trace = Trace(
        trace_id=str(uuid.uuid4())[:8],
        task_id=task["id"],
        user_goal=goal,
        model=model,
        expected=task.get("expected", {}),
    )

    system = {"role": "system", "content": SYSTEM_PROMPT.format(tools=tools_description())}
    scratchpad = f"Question: {goal}\n"

    for i in range(max_steps):
        user = {"role": "user", "content": scratchpad}
        raw = llm_fn([system, user], stop=["Observation:"])
        thought, tool_name, args, final, error = _parse(raw)

        # Case 1: the model is done
        if final is not None:
            trace.steps.append(Step(index=i, thought=thought))
            trace.final_answer = final
            trace.stopped_reason = "final_answer"
            return trace

        # Case 2: the model output was malformed -> record and let it retry
        if error:
            trace.steps.append(Step(index=i, thought=thought, error=error))
            scratchpad += raw + "\nObservation: Could not parse your step. Use the exact format.\n"
            continue

        # Case 3: a real tool call -> execute and append the observation
        observation = run_tool(tool_name, args or {})
        trace.steps.append(
            Step(
                index=i,
                thought=thought,
                action=ToolCall(tool_name=tool_name, args=args or {}),
                observation=observation,
            )
        )
        scratchpad += (
            f"Thought: {thought}\nAction: {tool_name}\n"
            f"Action Input: {json.dumps(args)}\nObservation: {observation}\n"
        )

    trace.stopped_reason = "max_steps"
    return trace
