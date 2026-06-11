# TraceJudge

**An agent-trajectory evaluator.** Most LLM evaluators grade a single
question→answer pair. TraceJudge reads an agent's full multi-step *trajectory*
(thought → tool call → observation → … → answer) and flags **which step** went
wrong and **why** — catching failures a final-answer-only check misses, like a
correct answer reached through a wasteful or wrong path.

🔗 **Live demo:** https://huggingface.co/spaces/kashishrai/tracejudge
(landing page) 

🔗 **Live demo:** [https://kashishrai-tracejudge.hf.space/]
(interactive API at `/docs`)

---

## Why

An agent can reach the right final answer while calling the wrong tools,
hallucinating a tool result, looping, or burning needless cost — and a
final-answer-only evaluator scores all of that as a pass. TraceJudge evaluates
the whole trajectory and localizes the failure.

## Architecture

A two-layer, tiered evaluator (cheap signals first, LLM only where needed):

- **Trace schema** (`schema.py`) — one data model for an agent run:
  `user_goal → steps[] → final_answer`, each step a `thought` + `action`
  (`{tool_name, args}`) + `observation`.
- **Toy ReAct agent** (`agent.py`, `tools.py`, `tasks.py`) — a minimal agent over
  a 3-tool mock support backend, used to generate trajectories.
- **Layer 1 — deterministic checks** (`evaluators.py`) — free, instant,
  reproducible: termination, tool validity, expected-tool set, step efficiency,
  redundant actions, answer grounding, outcome match.
- **Layer 2 — LLM-as-judge** (`judge.py`) — the semantic calls layer 1 can't make:
  tool *appropriateness*, faithfulness (hallucination), and goal completion.
- **Pipeline + API** (`pipeline.py`, `api.py`) — both layers return the same result
  type and merge into one verdict, served over FastAPI (`POST /evaluate`).
- **Benchmark** (`inject_failures.py`, `benchmark.py`) — injects labelled failures
  into clean traces to measure detection rate and false positives (a sanity check
  confirming the checks route known failures correctly with zero false positives
  on clean runs).

## Results — validated on TRAIL (external human labels)

The headline validation is against [TRAIL](https://github.com/patronus-ai/trail-benchmark)
(Patronus AI), a public benchmark of human-annotated agent traces. TRAIL is
hard — the best frontier models score ~11% on its full joint task. TraceJudge's
judge layer was evaluated on a 50-trace GAIA-split slice (~1,300 spans) for
binary span-level error detection:

| Configuration | Exact P | Exact R | Exact F1 | ±1-span F1 |
|---|---|---|---|---|
| Per-span judge (8B) | 0.12 | 0.76 | 0.21 | 0.51 |
| **+ deterministic span-type filter** | **0.22** | **0.76** | **0.34** | **0.52** |

Key findings (honest framing):
- **High recall, low precision** — the judge rarely *misses* a real error
  (recall 0.76, rising to ~0.98 at ±1-span localization) but over-flags.
- **Model size lifts recall, not precision** — an 8B→70B judge raised recall
  (0.78→0.90 on a 25-trace slice) but left precision flat, so the bottleneck is
  *context granularity, not model capability*.
- **A deterministic span-type filter doubled exact-match precision at zero recall
  cost** — skipping framework/orchestration spans (the same tiered-routing idea as
  a log classifier) removed ~440 false positives without losing a single true
  positive.

Numbers are a held-out slice with an 8B/70B judge, reported at exact and windowed
(±1-span) localization — not the full benchmark. See `trail_eval/`.

## Quickstart

```bash
pip install -r requirements.txt
python test_smoke.py                 # verify (no API key needed)
cp .env.example .env                 # add your Groq key
python run_agent.py                  # generate traces
python evaluate_traces.py            # deterministic checks
python judge_traces.py               # + LLM-judge layer
```

Run the API locally:
```bash
uvicorn api:app --reload             # docs at http://127.0.0.1:8000/docs
```

## Deploy

Containerized via the included `Dockerfile`; deployed on Hugging Face Spaces.

## Layout

```
schema, tools, tasks, llm, agent      core: data model + toy agent
evaluators, judge, pipeline           the two-layer evaluator
api, Dockerfile                       service + container
run_agent, evaluate_traces,           runners
  judge_traces, inject_failures, benchmark
trail_eval/                           TRAIL benchmark validation
```
