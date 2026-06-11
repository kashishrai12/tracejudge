"""Failure injection + benchmark generation (Day 4).

The credibility engine of the whole project. We start from clean, correct
trajectories, then programmatically corrupt copies to create KNOWN failures
with KNOWN labels. Running the evaluator over this labelled set tells us, with
real numbers, how reliably it catches each failure type -- and how often it
cries wolf on clean runs.

Each injector returns (corrupted_trace, failure_type, expected_check):
the check we expect to fire. That lets the benchmark measure not just "did it
notice something wrong" but "did it diagnose the RIGHT thing".
"""
from __future__ import annotations

import json
from pathlib import Path

from schema import Step, ToolCall, Trace

# --------------------------------------------------------------------------
# canonical CLEAN trajectories (the negatives)
# --------------------------------------------------------------------------
def _clean_traces() -> list[Trace]:
    return [
        Trace(
            trace_id="clean_status", task_id="order_status",
            user_goal="What's the status of my order ORD1002?",
            steps=[
                Step(index=0, thought="Look up the order.",
                     action=ToolCall(tool_name="order_lookup", args={"order_id": "ORD1002"}),
                     observation="Order ORD1002: Yoga Mat, status=In Transit, amount=Rs899, ordered on 2026-06-08."),
                Step(index=1, thought="I have the status."),
            ],
            final_answer="Your order ORD1002 (Yoga Mat) is In Transit. Amount Rs899.",
            stopped_reason="final_answer",
            expected={"tools": ["order_lookup"], "answer_contains": "In Transit"},
        ),
        Trace(
            trace_id="clean_refund", task_id="refund",
            user_goal="When will I get my refund for ORD1003?",
            steps=[
                Step(index=0, thought="Check the refund.",
                     action=ToolCall(tool_name="refund_status", args={"order_id": "ORD1003"}),
                     observation="Refund for ORD1003: Refund Initiated, amount=Rs3199, expected in 5 business days."),
                Step(index=1, thought="Done."),
            ],
            final_answer="Your refund for ORD1003 is initiated, expected within 5 business days (Rs3199).",
            stopped_reason="final_answer",
            expected={"tools": ["refund_status"], "answer_contains": "5"},
        ),
        Trace(
            trace_id="clean_multi", task_id="multi",
            user_goal="Did ORD1003 ship, and if cancelled what's the refund status?",
            steps=[
                Step(index=0, thought="Check the order.",
                     action=ToolCall(tool_name="order_lookup", args={"order_id": "ORD1003"}),
                     observation="Order ORD1003: Coffee Grinder, status=Cancelled, amount=Rs3199, ordered on 2026-06-01."),
                Step(index=1, thought="It was cancelled, check refund.",
                     action=ToolCall(tool_name="refund_status", args={"order_id": "ORD1003"}),
                     observation="Refund for ORD1003: Refund Initiated, amount=Rs3199, expected in 5 business days."),
                Step(index=2, thought="Answer."),
            ],
            final_answer="ORD1003 was cancelled and did not ship. Refund of Rs3199 initiated, expected in 5 business days.",
            stopped_reason="final_answer",
            expected={"tools": ["order_lookup", "refund_status"], "answer_contains": "Refund"},
        ),
    ]


# --------------------------------------------------------------------------
# injectors -- each returns (trace, failure_type, expected_check)
# --------------------------------------------------------------------------
def _first_tool_step(tr: Trace) -> Step:
    return next(s for s in tr.steps if s.action is not None)


def inject_drop_answer(tr: Trace):
    t = tr.model_copy(deep=True)
    t.final_answer = None
    t.stopped_reason = "max_steps"
    t.steps = [s for s in t.steps if s.action is not None]   # drop the final step
    return t, "dropped_answer", "termination"


def inject_unknown_tool(tr: Trace):
    t = tr.model_copy(deep=True)
    _first_tool_step(t).action.tool_name = "frobnicate"
    return t, "nonexistent_tool", "tool_validity"


def inject_missing_arg(tr: Trace):
    t = tr.model_copy(deep=True)
    _first_tool_step(t).action.args = {}
    return t, "missing_argument", "tool_validity"


def inject_wrong_tool(tr: Trace):
    t = tr.model_copy(deep=True)
    # swap to a real but wrong tool (not in this task's expected set)
    expected = set(tr.expected.get("tools", []))
    wrong = next(x for x in ["faq_search", "refund_status", "order_lookup"] if x not in expected)
    _first_tool_step(t).action.tool_name = wrong
    return t, "wrong_tool", "expected_tools"


def inject_redundant(tr: Trace):
    t = tr.model_copy(deep=True)
    fs = _first_tool_step(t)
    dup = fs.model_copy(deep=True)
    idx = t.steps.index(fs)
    t.steps.insert(idx + 1, dup)
    for i, s in enumerate(t.steps):           # renumber
        s.index = i
    return t, "redundant_loop", "redundant_actions"


def inject_extra_tool(tr: Trace):
    t = tr.model_copy(deep=True)
    extra = Step(index=0, thought="Unnecessary extra check.",
                 action=ToolCall(tool_name="faq_search", args={"query": "random"}),
                 observation="No FAQ matched that query.")
    # only counts as 'extra' if faq_search isn't expected
    if "faq_search" in set(tr.expected.get("tools", [])):
        return inject_redundant(tr)
    t.steps.insert(len(t.steps) - 1, extra)
    for i, s in enumerate(t.steps):
        s.index = i
    return t, "unnecessary_tool", "expected_tools"


def inject_fabricated_number(tr: Trace):
    t = tr.model_copy(deep=True)
    t.final_answer = (t.final_answer or "") + " A processing fee of Rs77777 was applied."
    return t, "fabricated_figure", "answer_grounding"


INJECTORS = [
    inject_drop_answer,
    inject_unknown_tool,
    inject_missing_arg,
    inject_wrong_tool,
    inject_redundant,
    inject_extra_tool,
    inject_fabricated_number,
]


def generate_benchmark(out_dir: str = "benchmark") -> None:
    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    labels = {}
    clean = _clean_traces()

    # negatives: the clean traces themselves
    for tr in clean:
        name = f"clean__{tr.task_id}.json"
        (out / name).write_text(json.dumps(tr.model_dump(), indent=2))
        labels[name] = {"label": "clean", "failure_type": None, "expected_check": None}

    # positives: every injector applied to every clean trace
    for tr in clean:
        for inj in INJECTORS:
            corrupted, ftype, check = inj(tr)
            name = f"bad__{tr.task_id}__{ftype}.json"
            (out / name).write_text(json.dumps(corrupted.model_dump(), indent=2))
            labels[name] = {"label": "bad", "failure_type": ftype, "expected_check": check}

    (out / "labels.json").write_text(json.dumps(labels, indent=2))
    n_bad = sum(1 for v in labels.values() if v["label"] == "bad")
    n_clean = sum(1 for v in labels.values() if v["label"] == "clean")
    print(f"Wrote benchmark to ./{out_dir}/  ({n_clean} clean, {n_bad} injected failures)")


if __name__ == "__main__":
    generate_benchmark()
