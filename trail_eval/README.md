# TRAIL validation

Validates TraceJudge's LLM-judge layer against [TRAIL](https://github.com/patronus-ai/trail-benchmark),
a public benchmark of 148 human-annotated agent execution traces.

## Setup
1. Clone the benchmark (the data is not vendored here):
   `git clone https://github.com/patronus-ai/trail-benchmark`
2. Copy these four scripts into the cloned `trail-benchmark/` folder.
3. Put your `GROQ_API_KEY` in a `.env` (in the parent folder is fine).

## Run
- `python trail_score.py 50`   - judge each span, cache verdicts (resumable)
- `python score_relaxed.py`    - score with exact / +-1 / +-2 span localization
- `python score_filtered.py`   - add the deterministic span-type filter

See the main README for the results these produce.
