"""Day 1 entry point.

Runs the toy ReAct agent on every sample task and saves each trajectory as a
JSON file under traces/. Those JSON files are the input to everything you
build next.

Usage:
    python run_agent.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv

from agent import run_react_agent
from tasks import TASKS

load_dotenv()
OUT = Path("traces")


def main() -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("Set GROQ_API_KEY in a .env file first (see .env.example).")

    OUT.mkdir(exist_ok=True)
    for task in TASKS:
        print(f"\n=== Running {task['id']} ===")
        trace = run_react_agent(task)
        path = OUT / f"{task['id']}.json"
        path.write_text(json.dumps(trace.model_dump(), indent=2))
        print(f"  steps={len(trace.steps)}  stopped={trace.stopped_reason}")
        print(f"  final_answer: {trace.final_answer}")
        print(f"  saved -> {path}")

    print(f"\nDone. {len(TASKS)} traces written to ./{OUT}/")


if __name__ == "__main__":
    main()
