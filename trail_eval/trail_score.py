"""Score a per-span error judge against TRAIL's human annotations.

Checkpointed + resumable: every span verdict is cached, so rate-limit errors
or a crash never lose progress -- just re-run. Run from inside the cloned
trail-benchmark folder.

Usage:
    python trail_score.py 10      # smoke run
    python trail_score.py         # full run (resumes from cache)
"""
import json
import os
import re
import sys
import time
from pathlib import Path

from dotenv import find_dotenv, load_dotenv
from groq import Groq

from trail_adapter import load_trail

load_dotenv(find_dotenv(usecwd=True))
# 8b-instant has higher free-tier limits; 70b is a stronger judge. Either works.
MODEL = "llama-3.1-8b-instant"
CACHE = Path("trail_cache.jsonl")
client = Groq(api_key=os.environ["GROQ_API_KEY"])

SYS = """You are an expert at debugging AI agent execution traces. You are shown the
CURRENT step plus the PREVIOUS step for context, and you judge ONLY the current step.

Flag the current step as an error if it shows any of: a hallucination or fabricated fact,
a wrong or poorly chosen tool, a tool call with bad arguments, deviation from the user's
goal, ignoring an instruction, a formatting error, mishandling of earlier context,
needless repetition of a previous action, or a failed/poor information retrieval.

Be willing to flag: missing a real error is worse than an occasional false alarm.
Judge the CURRENT step only.

Respond ONLY with JSON: {"is_error": true|false, "reason": "<short>"}"""


def load_cache():
    c = {}
    if CACHE.exists():
        for line in CACHE.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                c[r["key"]] = r["is_error"]
    return c


def judge_span(task, step, prev=None):
    prev_block = ""
    if prev:
        prev_block = (f"PREVIOUS step (context only):\n  name: {prev['name']}\n"
                      f"  output: {prev['output'][:400]}\n\n")
    user = (f"Task: {task[:600]}\n\n{prev_block}"
            f"CURRENT step (judge THIS one):\n  name: {step['name']}\n"
            f"  arguments: {step['args'][:600]}\n  output: {step['output'][:600]}\n\n"
            f"Does the CURRENT step contain an error?")
    for attempt in range(4):
        try:
            r = client.chat.completions.create(
                model=MODEL, temperature=0, max_tokens=150, timeout=30,
                messages=[{"role": "system", "content": SYS},
                          {"role": "user", "content": user}])
            txt = r.choices[0].message.content or ""
            m = re.search(r"\{.*\}", txt, re.DOTALL)
            return bool(json.loads(m.group(0)).get("is_error", False))
        except Exception as e:  # noqa: BLE001
            print(f"      [retry {attempt+1}] {str(e)[:90]}")
            time.sleep(30 if ("429" in str(e) or "rate" in str(e).lower()) else 3)
    return False


def task_context(steps):
    for s in steps:
        if "question" in (s["args"] + s["output"]).lower() or "task" in (s["args"] + s["output"]).lower():
            return (s["args"] + " " + s["output"])[:500]
    return steps[0]["output"][:500] if steps else ""


def _f1(tp, fp, fn):
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return 2 * p * r / (p + r) if p + r else 0.0


def main():
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else None
    data = load_trail(limit=limit)
    cache = load_cache()
    cf = CACHE.open("a", encoding="utf-8")
    TP = FP = FN = TN = 0
    total = sum(len(t["steps"]) for t in data)
    done = 0
    for t in data:
        ctx = task_context(t["steps"])
        gold = t["gold_error_spans"]
        prev = None
        for step in t["steps"]:
            time.sleep(2.0)  # throttle to stay under per-minute free-tier limits
            key = f"{t['trace_id']}:{step['span_id']}"
            if key in cache:
                pred = cache[key]
            else:
                pred = judge_span(ctx, step, prev)
                cf.write(json.dumps({"key": key, "is_error": pred}) + "\n")
                cf.flush()
                cache[key] = pred
            prev = step
            is_gold = step["span_id"] in gold
            if pred and is_gold: TP += 1
            elif pred and not is_gold: FP += 1
            elif not pred and is_gold: FN += 1
            else: TN += 1
            done += 1
        print(f"  {t['trace_id'][:12]}  ({done}/{total})  running F1={_f1(TP,FP,FN):.3f}")
    cf.close()
    prec = TP / (TP + FP) if TP + FP else 0.0
    rec = TP / (TP + FN) if TP + FN else 0.0
    print(f"\nspans judged : {TP+FP+FN+TN}\nprecision : {prec:.3f}\n"
          f"recall    : {rec:.3f}\nF1        : {_f1(TP,FP,FN):.3f}")


if __name__ == "__main__":
    main()
