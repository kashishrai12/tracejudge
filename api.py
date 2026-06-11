"""TraceJudge API — agent-trajectory evaluator.

POST a trace to /evaluate and get a step-level verdict back. The deterministic
layer runs by default (free, no key). Pass ?use_llm=true to add the LLM-judge
layer (requires GROQ_API_KEY set as a Space secret).
"""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from pipeline import evaluate_full
from schema import Trace

app = FastAPI(title="TraceJudge", description="Agent-trajectory evaluator")

EXAMPLE = """{
  "trace_id": "demo1",
  "task_id": "order_status",
  "user_goal": "What's the status of my order ORD1002?",
  "steps": [
    {"index": 0, "thought": "Look it up.",
     "action": {"tool_name": "order_lookup", "args": {"order_id": "ORD1002"}},
     "observation": "Order ORD1002: Yoga Mat, status=In Transit, amount=Rs899."},
    {"index": 1, "thought": "Unnecessary extra check.",
     "action": {"tool_name": "faq_search", "args": {"query": "transit"}},
     "observation": "Orders can be cancelled before they ship."},
    {"index": 2, "thought": "Answer."}
  ],
  "final_answer": "Your order ORD1002 (Yoga Mat) is In Transit.",
  "stopped_reason": "final_answer",
  "expected": {"tools": ["order_lookup"], "answer_contains": "In Transit"}
}"""

LANDING = f"""<!doctype html><html><head><meta charset="utf-8">
<title>TraceJudge</title>
<style>
 body{{font-family:system-ui,sans-serif;max-width:760px;margin:40px auto;padding:0 20px;
       line-height:1.55;color:#1a1a2e}}
 code,pre{{background:#f4f4f8;border-radius:6px}} pre{{padding:16px;overflow:auto;font-size:13px}}
 code{{padding:2px 5px}} a{{color:#4338ca}} h1{{margin-bottom:4px}}
 .tag{{color:#666;font-size:14px}}
</style></head><body>
<h1>TraceJudge</h1>
<p class="tag">An agent-trajectory evaluator. Scores a whole agent run, not just its final answer.</p>
<p>Most evaluators grade a single question and answer. TraceJudge reads an agent's full
multi-step trajectory and flags <b>which step</b> went wrong and <b>why</b> — catching
failures a final-answer-only check misses (e.g. a correct answer reached via a wasteful path).</p>
<p><b>Try it:</b> open the interactive API at <a href="/docs">/docs</a>, expand
<code>POST /evaluate</code>, click <b>Try it out</b>, and paste the example below.</p>
<p>Or from a terminal:</p>
<pre>curl -X POST "%API%/evaluate" -H "Content-Type: application/json" -d @trace.json</pre>
<p>Example trace (the agent answers correctly but calls an unnecessary tool — watch it
get flagged WARN):</p>
<pre>{EXAMPLE}</pre>
</body></html>"""


@app.get("/", response_class=HTMLResponse)
def landing() -> str:
    return LANDING


@app.get("/health")
def health() -> dict:
    return {"service": "TraceJudge", "status": "ok"}


@app.post("/evaluate")
def evaluate(trace: Trace, use_llm: bool = False) -> dict:
    """Evaluate one agent trajectory. use_llm=true adds the LLM-judge layer."""
    return evaluate_full(trace, use_llm=use_llm)
