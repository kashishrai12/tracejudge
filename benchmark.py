"""Benchmark runner (Day 4) -- produces the headline numbers.

Runs the evaluator over the labelled benchmark set and reports:
  - Detection rate (recall): of injected failures, how many were flagged.
  - Targeted accuracy:       how many fired the *correct* diagnostic check.
  - False-positive rate:     of clean traces, how many were wrongly flagged.
  - Per-failure-type breakdown.

Uses the deterministic layer only (use_llm=False) so the number is stable,
free, and reproducible -- exactly what you want to quote in an email.
Add --llm to include the judge layer.

Usage:
    python inject_failures.py      # generate the benchmark first
    python benchmark.py            # then score it
    python benchmark.py --llm      # include LLM judges
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

from pipeline import evaluate_full
from schema import Trace

BENCH = Path("benchmark")


def main() -> None:
    use_llm = "--llm" in sys.argv
    labels = json.loads((BENCH / "labels.json").read_text())

    detected = 0          # injected failures flagged as non-pass
    targeted = 0          # injected failures whose expected check fired
    n_bad = 0
    false_pos = 0
    n_clean = 0
    by_type = defaultdict(lambda: {"total": 0, "detected": 0, "targeted": 0})

    for name, meta in labels.items():
        trace = Trace(**json.loads((BENCH / name).read_text()))
        report = evaluate_full(trace, use_llm=use_llm)
        flagged = report["verdict"] != "pass"
        fired = set(report["fired_checks"])

        if meta["label"] == "bad":
            n_bad += 1
            ftype = meta["failure_type"]
            by_type[ftype]["total"] += 1
            if flagged:
                detected += 1
                by_type[ftype]["detected"] += 1
            if meta["expected_check"] in fired:
                targeted += 1
                by_type[ftype]["targeted"] += 1
        else:
            n_clean += 1
            if flagged:
                false_pos += 1

    print(f"\n{'='*64}\nTraceJudge benchmark  ({'deterministic + LLM' if use_llm else 'deterministic only'})")
    print(f"{'='*64}")
    print(f"Injected failures detected : {detected}/{n_bad}  "
          f"({100*detected/n_bad:.0f}% recall)")
    print(f"Correct check fired        : {targeted}/{n_bad}  "
          f"({100*targeted/n_bad:.0f}% targeted accuracy)")
    print(f"False positives on clean   : {false_pos}/{n_clean}  "
          f"({100*(n_clean-false_pos)/n_clean:.0f}% clean-pass rate)")
    print(f"\nBy failure type:")
    print(f"  {'type':22} {'detected':>10} {'targeted':>10}")
    for ftype, d in sorted(by_type.items()):
        print(f"  {ftype:22} {d['detected']:>4}/{d['total']:<5} {d['targeted']:>4}/{d['total']:<5}")

    Path("reports").mkdir(exist_ok=True)
    Path("reports/benchmark_report.json").write_text(json.dumps({
        "mode": "det+llm" if use_llm else "det",
        "recall": detected / n_bad,
        "targeted_accuracy": targeted / n_bad,
        "false_positive_rate": false_pos / n_clean,
        "by_type": dict(by_type),
    }, indent=2))
    print(f"\nFull report -> reports/benchmark_report.json")


if __name__ == "__main__":
    main()
