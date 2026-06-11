"""Day 2 entry point.

Reads every traces/*.json, runs the deterministic evaluators, prints a
per-trace verdict report to the console, and saves a machine-readable
report to reports/eval_report.json (used later by the Day 5 dashboard).

Usage:
    python evaluate_traces.py            # reads ./traces
    python evaluate_traces.py somedir    # reads ./somedir
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from evaluators import evaluate_trace
from schema import Trace

TAG = {"pass": "[PASS]", "warn": "[WARN]", "fail": "[FAIL]"}


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("traces")
    files = sorted(src.glob("*.json"))
    if not files:
        raise SystemExit(f"No trace files found in ./{src}/. Run run_agent.py first.")

    reports, counts = [], {"pass": 0, "warn": 0, "fail": 0}
    for f in files:
        trace = Trace(**json.loads(f.read_text()))
        report = evaluate_trace(trace)
        reports.append(report)
        counts[report["verdict"]] += 1

        print(f"\n{'='*70}")
        print(f"{report['task_id']}   ->   VERDICT: {TAG[report['verdict']]}")
        print(f"  goal: {trace.user_goal}")
        print(f"  {'-'*66}")
        for r in report["results"]:
            where = f"  (steps {r['steps']})" if r["steps"] else ""
            print(f"  {TAG[r['status']]:7} {r['name']:18} {r['message']}{where}")

    out_dir = Path("reports")
    out_dir.mkdir(exist_ok=True)
    (out_dir / "eval_report.json").write_text(json.dumps(reports, indent=2))

    print(f"\n{'='*70}")
    print(f"SUMMARY: {counts['pass']} pass, {counts['warn']} warn, "
          f"{counts['fail']} fail  (of {len(files)} traces)")
    print(f"Full report -> reports/eval_report.json")


if __name__ == "__main__":
    main()
