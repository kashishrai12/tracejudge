"""Day 3 entry point: run the three LLM judges over traces/*.json.

Needs a Groq key (these judges call the LLM). Cost is ~3 calls per trace.

Usage:
    python judge_traces.py            # reads ./traces
    python judge_traces.py somedir
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

from judge import judge_trace
from schema import Trace

load_dotenv()
TAG = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}


def main() -> None:
    if "GROQ_API_KEY" not in os.environ:
        raise SystemExit("Set GROQ_API_KEY in .env first (the judges call the LLM).")

    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("traces")
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"No trace files in ./{src}/. Run run_agent.py first.")

    reports = []
    for f in files:
        trace = Trace(**json.loads(f.read_text()))
        results = judge_trace(trace)
        reports.append({
            "task_id": trace.task_id,
            "results": [r.to_dict() for r in results],
        })
        print(f"\n{'='*70}\n{trace.task_id}\n  goal: {trace.user_goal}\n  {'-'*66}")
        for r in results:
            where = f"  (steps {r.steps})" if r.steps else ""
            print(f"  {TAG[r.status]:7} {r.name:20} {r.message}{where}")

    out = Path("reports")
    out.mkdir(exist_ok=True)
    (out / "judge_report.json").write_text(json.dumps(reports, indent=2))
    print(f"\n{'='*70}\nFull report -> reports/judge_report.json")


if __name__ == "__main__":
    main()
