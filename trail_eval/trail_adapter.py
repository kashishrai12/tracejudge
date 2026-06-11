"""Flatten TRAIL OpenTelemetry traces into ordered steps + gold error spans.

TRAIL (Patronus AI) is a public benchmark of human-annotated agent traces:
https://github.com/patronus-ai/trail-benchmark

Run these scripts from inside the cloned trail-benchmark folder (so the
relative paths below resolve), or adjust DATA/ANNO to point at it.
"""
import glob
import json
import os

DATA = "benchmarking/data/GAIA"
ANNO = "benchmarking/processed_annotations_gaia"


def _flatten(trace):
    """Depth-first walk of the span tree -> ordered list of spans."""
    flat = []

    def walk(span):
        flat.append(span)
        for c in span.get("child_spans", []):
            walk(c)

    for s in trace.get("spans", []):
        walk(s)
    return flat


def _span_text(span):
    """Pull readable input/output text from a span.

    Model-call spans (LiteLLMModel.__call__) store the real content in
    span_attributes['input.value'/'output.value'] as JSON-stringified message
    lists (OpenInference convention). Tool/function spans use
    logs[].body.function.*. We try both.
    """
    name = span.get("span_name", "")
    attrs = span.get("span_attributes", {}) or {}

    def _messages_to_text(raw):
        if not raw:
            return ""
        try:
            obj = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            return str(raw)
        msgs = obj.get("messages", obj) if isinstance(obj, dict) else obj
        if isinstance(msgs, list):
            parts = []
            for m in msgs:
                if isinstance(m, dict):
                    c = m.get("content", "")
                    if isinstance(c, list):
                        c = " ".join(b.get("text", "") for b in c if isinstance(b, dict))
                    parts.append(f"{m.get('role','')}: {c}")
                else:
                    parts.append(str(m))
            return "\n".join(parts)
        return str(msgs)

    args = _messages_to_text(attrs.get("input.value", ""))
    out = _messages_to_text(attrs.get("output.value", ""))

    if not args and not out:
        for lg in span.get("logs", []):
            body = lg.get("body", {}) or {}
            if body.get("function.name"):
                name = body.get("function.name")
                args = str(body.get("function.arguments", ""))
                out = str(body.get("function.output", ""))
                break

    return name, args, out


def load_trail(limit=None):
    """Yield dicts: {trace_id, steps:[{span_id,name,args,output}],
    gold_error_spans:set, categories}."""
    files = sorted(glob.glob(f"{DATA}/*.json"))
    if limit:
        files = files[:limit]
    out = []
    for f in files:
        tid = os.path.splitext(os.path.basename(f))[0]
        ann_path = os.path.join(ANNO, f"{tid}.json")
        if not os.path.exists(ann_path):
            continue
        trace = json.load(open(f, encoding="utf-8"))
        ann = json.load(open(ann_path, encoding="utf-8"))
        steps = []
        for s in _flatten(trace):
            name, a, o = _span_text(s)
            steps.append({"span_id": s.get("span_id"), "name": name,
                          "args": a[:500], "output": o[:500]})
        errors = ann.get("errors", []) or []
        out.append({
            "trace_id": tid,
            "steps": steps,
            "gold_error_spans": {e.get("location") for e in errors if e.get("location")},
            "categories": sorted({e.get("category") for e in errors if e.get("category")}),
        })
    return out
