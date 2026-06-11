"""Apply a deterministic span-type filter, then score (exact + windowed).
Skips framework/orchestration spans that never carry real errors -- the same
tiered-routing idea as the log-classifier: let cheap signals handle what they
can before the LLM. Reads the cache; no API calls."""
import json
import re
from pathlib import Path
from trail_adapter import load_trail

cache_file = "trail_cache_70b.jsonl" if Path("trail_cache_70b.jsonl").exists() else "trail_cache.jsonl"
cache = {}
for line in Path(cache_file).read_text(encoding="utf-8").splitlines():
    if line.strip():
        r = json.loads(line)
        cache[r["key"]] = r["is_error"]

data = load_trail(limit=50)

SKIP_EXACT = {
    "main", "CodeAgent.run", "ToolCallingAgent.run", "answer_single_question",
    "get_examples_to_answer", "create_agent_hierarchy", "FinalAnswerTool",
}


def is_judgeable(name):
    name = name or ""
    if re.fullmatch(r"Step \d+", name):
        return False
    return name not in SKIP_EXACT


def score(window, apply_filter):
    TP = FP = FN = 0
    for t in data:
        steps, gold = t["steps"], t["gold_error_spans"]
        gold_pos = [i for i, s in enumerate(steps) if s["span_id"] in gold]
        pred_pos, judged = [], False
        for i, s in enumerate(steps):
            key = f"{t['trace_id']}:{s['span_id']}"
            if key not in cache:
                continue
            judged = True
            if apply_filter and not is_judgeable(s["name"]):
                continue
            if cache[key]:
                pred_pos.append(i)
        if not judged:
            continue
        for p in pred_pos:
            TP += 1 if any(abs(p - g) <= window for g in gold_pos) else 0
            FP += 0 if any(abs(p - g) <= window for g in gold_pos) else 1
        for g in gold_pos:
            if not any(abs(p - g) <= window for p in pred_pos):
                FN += 1
    p = TP / (TP + FP) if TP + FP else 0
    r = TP / (TP + FN) if TP + FN else 0
    f = 2 * p * r / (p + r) if p + r else 0
    return p, r, f, TP, FP, FN


print(f"using cache: {cache_file}\n")
for apply_filter in (False, True):
    print("=== WITH filter ===" if apply_filter else "=== no filter (baseline) ===")
    print(f"{'match':<14}{'precision':>10}{'recall':>9}{'F1':>8}")
    for w in (0, 1, 2):
        p, r, f, TP, FP, FN = score(w, apply_filter)
        label = "exact" if w == 0 else f"within {w} step"
        print(f"{label:<14}{p:>10.3f}{r:>9.3f}{f:>8.3f}   (TP={TP} FP={FP} FN={FN})")
    print()
