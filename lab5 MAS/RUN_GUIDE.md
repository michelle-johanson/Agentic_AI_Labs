# Lab 5 run guide

The extension is the shared fact-checked debate history in `debateMAS.py`. This guide covers the environment used on 5 November 2026 and the commands that reproduce the checks. It does not submit the assignment.

## Environment

The repository had no virtual environment. This session created one at the repository root:

`Agentic_AI_Labs/.venv`

- Python 3.14.7
- Ollama client 0.34.4, listening at `http://127.0.0.1:11434`
- Model `qwen2.5:7b`, Ollama id `845dbda0ea48`, size 4.7 GB

`qwen2.5:7b` was not installed. It was downloaded with `ollama pull qwen2.5:7b`. Ollama was not already running; it was started with `ollama serve`.

Installed package versions are pinned in `requirements-lab5.txt`. The packages the script imports are:

| Package | Version |
| --- | --- |
| langchain | 1.4.8 |
| langchain-core | 1.7.6 |
| langchain-ollama | 1.1.0 |
| langchain-tavily | 0.2.18 |
| langgraph | 1.2.19 |
| langsmith | 0.14.17 |
| python-dotenv | 1.2.3 |
| typing_extensions | 4.16.0 |
| python-docx | 1.2.0 |

Recreate the environment from the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r "lab5 MAS/requirements-lab5.txt"
ollama serve
ollama pull qwen2.5:7b
```

## Environment variable names

Put real values in `Agentic_AI_Labs/.env`. That file is gitignored. Placeholders are in `.env.example`. Do not put keys in source, logs, or the memo.

- `TAVILY_API_KEY`
- `LANGSMITH_API_KEY`
- `LANGSMITH_TRACING` (`true` when tracing)
- `LANGSMITH_ENDPOINT` (`https://api.smith.langchain.com`)
- `LANGSMITH_PROJECT` (the runner sets this to `is590r-lab5-debate` for a live run)

`eval_runner.py --check-ready` reports whether each key is missing, a placeholder, or present. It does not print the value.

The local `.env` now has both keys. Do not print them. The six completed runs from 5 November 2026 are in `evidence/runs/`, and their public trace URLs are in `evidence/comparison.md`.

## Baseline

The instructor script was found at:

`/Users/jensroylance/Desktop/School Stuff/Fall 2026/IS590R Agentic AI/Labs/lab5 - MAS/debateMAS.py`

Its SHA-256 is `574feba4d9a873c7b343536d8dcdfaf45d46dc0bc640ee5117f96343315c06d3`.

`baseline/instructor_debateMAS.py` is a byte-identical copy. `baseline/instructor_debateMAS_runnable.py` is the same file with `draw_mermaid_png` commented out so diagram generation cannot block a run. Its SHA-256 is `def3a4391ae0aa86c9d51561005abf52eaf3c57ea481dabf0ef7839ed8901310`. No debate prompt, case, model, or round limit was changed in that copy.

The merged script's SHA-256 is `3f16cb0f47b734413070a54fa1ded40827a9767fd63e83cb398574b10c22afba`. Its diagram call is also commented out. `_append_trace()` writes a JSONL file only when `DEBATE_TRACE_PATH` is set. With that variable unset, a normal run does not write a trace file and does not change the debate text.

## Offline checks

From `lab5 MAS`:

```bash
../.venv/bin/python offline_checks.py
```

The saved log is `evidence/offline/offline_checks.txt`. These checks use stub agents on the compiled merged graph. They are not live Ollama, Tavily, or LangSmith results. The latest run passed 7/7.

## Live runs

Check readiness first:

```bash
../.venv/bin/python eval_runner.py --check-ready
```

After both API keys are in `.env`, and while `ollama serve` is running, execute one process per run. The default timeout is 1500 seconds. A timed-out or failed run is saved with `status` `failed` and is not a final successful evaluation.

```bash
../.venv/bin/python eval_runner.py --variant baseline --run-id baseline-01
../.venv/bin/python eval_runner.py --variant baseline --run-id baseline-02
../.venv/bin/python eval_runner.py --variant baseline --run-id baseline-03
../.venv/bin/python eval_runner.py --variant merged --run-id merged-01
../.venv/bin/python eval_runner.py --variant merged --run-id merged-02
../.venv/bin/python eval_runner.py --variant merged --run-id merged-03
```

Each completed or failed run writes:

- `evidence/runs/<run-id>/console.txt`
- `evidence/runs/<run-id>/trace.jsonl`
- `evidence/runs/<run-id>/final_state.json`
- `evidence/runs/<run-id>/meta.json`

`evidence/manifest.json` lists those runs. The runner records the source hash, package versions, model name, elapsed seconds, tool calls, and token-usage fields only when the response objects actually include `usage_metadata`.

When `LANGSMITH_API_KEY` is present, the runner turns on tracing for project `is590r-lab5-debate`, waits for the LangSmith tracers to flush, and looks up a root run whose name or metadata `run_id` matches the command. It then creates a public share URL and requests that URL without an Authorization header. It does not share a trace whose stored input or output contains `tvly-` or `lsv2_`. Setting the environment variable is not, by itself, recorded as proof that tracing worked. `meta.json` field `tracing.verified` is true only after an unauthenticated request returns status 200.

## Manual steps still required

1. Create `Agentic_AI_Labs/.env` with `TAVILY_API_KEY` and `LANGSMITH_API_KEY`.
2. Run the six commands above.
3. Open each public URL in a browser that is not logged into LangSmith and confirm it loads.
4. Screenshot the first-round checker report, its presence in both second-round advocate inputs, and the moderator input. Caption each image with the run id, participant, and what it shows.
5. Replace the blocker language in the memo and in `IMPROVEMENTS.md` with those measured results, including unfavorable ones.
6. Do not upload the whole project publicly from this session.
