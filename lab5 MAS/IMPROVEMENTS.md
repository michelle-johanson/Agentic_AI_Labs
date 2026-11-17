# Lab 5 improvement

## One change we extended

We added a **shared fact-checked debate history** to `debateMAS.py`. After each round, a fact-checker reads the accumulated arguments and earlier findings, investigates selected claims, and appends its report to the same history. Both advocates receive those findings before their next turn, and the moderator receives the full record before recommending BUILD or BUY.

This is one feedback process: preserve the debate, check claims in context, and make the findings available to every later participant. It is not a return to the search-record-only variant, and it does not add a new agent architecture.

## 1. What we changed in the code

### Shared state

- `DebateState.history: list[str]` stores chronological entries labeled `Round N - BUILD`, `Round N - BUY`, and `Round N - FACT CHECK`.
- `format_history(state)` joins those entries for participant prompts.
- `agent_position_a()` and `agent_position_b()` read the entire history and append their new arguments. Their system prompts tell them to correct or qualify claims challenged by evidence.
- `position_a` and `position_b` still hold the latest arguments for the final-output print statements.
- Initial state includes `history: []`. Reports live in `history`. There is no separate `fact_check_notes` list.

### Checking and saving findings

- A `fact_checker` agent has the existing `web_search` tool.
- Its prompt prioritizes one specific, decision-relevant claim per side from the latest round. It uses earlier history to identify unresolved or repeated claims and is instructed to avoid redundant checks unless a claim is contested or new evidence appears.
- The checker is instructed to identify each selected claim, label it supported, contradicted, or unverifiable, and explain the finding with a returned source URL. These instructions are prompt-based. The code does not enforce a claim count or verdict format.
- `run_fact_checker()` passes the case assumptions and complete history to the checker, then appends its report to `history`.
- `_fact_check_search_record()` adds a program-written record of the checker's tool-call count, queries, and parsed, deduplicated source URLs to that report.
- If the checker makes no tool call, the program appends: `Search record written by the program: no web search was performed. Treat this report as unverified.` If calls occur but no usable URLs can be parsed, it also marks the report unverified.

Search records are saved for the checker reports. Advocate tool activity is printed to the console. This version does not keep separate `sources_a` and `sources_b` fields.

### Graph and moderator

After BUILD and BUY speak, the graph always runs `fact_checker` before calling `continue_debate()`. That routing function sends execution back to BUILD if another round remains, or to the moderator after the final check.

`moderator()` reads the complete history, including reports from every round. Its system prompt tells it to consider those findings, distinguish case assumptions from externally checked claims, acknowledge unresolved findings, and avoid treating repeated claims or returned URLs alone as proof.

The main print block prints the complete history. Diagram generation stays commented out so a missing renderer cannot stop the run.

### Observation only

`_append_trace()` and `_trace_messages()` write prompts, tool calls, and outputs to `DEBATE_TRACE_PATH` when that variable is set. They do not change prompts, insert corrections, or add sources. With the variable unset, they do nothing.

## 2. Why we chose this change

The instructor baseline retained only the latest argument from each side. The preserved file is `baseline/instructor_debateMAS.py`, byte-identical to `/Users/jensroylance/Desktop/School Stuff/Fall 2026/IS590R Agentic AI/Labs/lab5 - MAS/debateMAS.py` (SHA-256 `574feba4d9a873c7b343536d8dcdfaf45d46dc0bc640ee5117f96343315c06d3`). Its moderator prompt contains `position_a` and `position_b` only.

A full transcript would preserve earlier claims and still would not investigate them. A checker whose report never reaches the advocates would give the moderator findings the advocates cannot answer. The merged version puts the report in the same history the next advocates, the next checker, and the moderator already read.

Alternatives considered:

- **Keep only the transcript:** preserves earlier arguments without a way to check their factual claims.
- **Keep only the original checker:** adds checking but leaves corrections outside the advocates' prompts.
- **Add more debate rounds:** increases work without fixing missing memory or correction feedback.
- **Check every claim in every round:** expands search and prompt size. The implemented prompt instead prioritizes selected claims. Live compliance with that instruction has not been measured.
- **Advocate search records alone:** an earlier variant stored advocate queries and URLs. That shows whether an advocate searched. It does not check a claim and feed the finding back into the next round. This extension does not restore that variant.

## 3. Benefits and supporting evidence

The earlier writeup described seven checks from a ChatGPT harness that used synthetic messages, stub agents, and a simulated graph runner. That harness is not in this project and was not rerun. Its numbers are not reused as results from this session.

On 5 November 2026, `offline_checks.py` was executed with `Agentic_AI_Labs/.venv` (Python 3.14.7) against the current `debateMAS.py`. It passed 7 of 7 checks. The log is `evidence/offline/offline_checks.txt`. The checks call the compiled LangGraph. The advocates, checker, and moderator are stubs. Ollama, Tavily, and LangSmith are not called.

| Check | Observed result |
| --- | --- |
| T1 — retention and order | Two rounds preserved four arguments and two checker reports, labeled Round 1 BUILD, BUY, FACT CHECK, then Round 2 BUILD, BUY, FACT CHECK. The original history list stayed empty. |
| T2 — checker context | The second checker input contained both rounds' arguments and the first report, including the fixture URL. |
| T3 — feedback to advocates | Both second-round advocate inputs contained the first report, the words `contradicted`, the fixture URL `https://example.com/crm-pricing`, and `Search record written by the program`. |
| T4 — moderator context | The moderator input contained all six entries and both search records. The stub resolution began `BUY.` |
| T5 — skipped search | The second checker response had no tool call. The saved report contains `Search record written by the program: no web search was performed. Treat this report as unverified.` |
| T6 — malformed results | Bad JSON and unusable URL fields did not crash `_fact_check_search_record()`. A usable URL was kept. A call with no usable URL was flagged unverified. |
| T7 — routing | The compiled graph's update order was `agent_position_a`, `agent_position_b`, `fact_checker`, `agent_position_a`, `agent_position_b`, `fact_checker`, `moderator`. |

The fixture URL is not a retrieved page. The planted sentence "contradicted" shows that the text reached later prompts. It does not show that a live model found or accepted an error.

Relative to the instructor source, a two-round moderator input in this offline run contained six history entries instead of the two final arguments the instructor function formats. That is an information-delivery difference. It is not, by itself, a gain in accuracy.

### Live comparison

Six final runs used the same `CASE`, `qwen2.5:7b`, and `max_rounds=2`. Three used the runnable instructor baseline. Three used the merged script. An earlier merged attempt was killed before it finished and is not counted. Details are in `evidence/comparison.md` and `evidence/runs/`.

| Run | Elapsed | History entries | Reported input / output tokens | Public trace |
| --- | --- | --- | --- | --- |
| baseline-01 | 126.9 s | 0 stored; moderator saw the two final arguments | 16,959 / 1,336 | https://smith.langchain.com/public/8d26bcc4-0752-4679-83ac-fc07f9e0287d/r |
| baseline-02 | 144.2 s | 0 stored | 17,114 / 1,588 | https://smith.langchain.com/public/bd9f8567-685a-4435-ae69-ebf4f8a05669/r |
| baseline-03 | 127.9 s | 0 stored | 17,239 / 1,352 | https://smith.langchain.com/public/a984d17b-1dda-492d-a2ef-ce7bcf957b8a/r |
| merged-01 | 286.5 s | 6 | 29,847 / 3,560 | https://smith.langchain.com/public/afac2519-ade3-4e2b-9fc6-3086508f5d2b/r |
| merged-02 | 339.1 s | 6 | 31,079 / 4,199 | https://smith.langchain.com/public/a44bfa64-6620-4290-b960-0a4d8f3e4f9c/r |
| merged-03 | 351.0 s | 6 | 19,815 / 5,461 | https://smith.langchain.com/public/f0ae147e-8b30-473d-93ee-c51ad9700d7d/r |

Mean elapsed time was 133.0 seconds for the baseline and 325.5 seconds for the merged version. Token counts come from `usage_metadata` on the model responses. No dollar cost was reported. The runtime recorded 7 model-usage records on every baseline run and 11, 12, and 9 records on merged-01, merged-02, and merged-03. Those are underlying model calls, not the seven top-level turns.

In every merged run, both second-round advocate inputs and the second checker input contained `Round 1 - FACT CHECK`, and the moderator input contained all six history labels. The baseline moderator input contained `BUILD side's final argument` and did not contain `Round 1`.

That delivery did not produce a cleaner decision. None of the six moderators put BUILD or BUY in the first sentence. All three merged checkers searched a broad cost comparison rather than one quoted external claim, and the saved snippets were not checked against the full source pages. In merged-01 and merged-03 the second checker made no search; the program marked those reports unverified, and merged-01 still called two claims supported while attaching an earlier URL to a different claim. No run produced a clear "contradicted" verdict that a later advocate then withdrew. Merged-03's BUY advocate argued for building. These are failures of use, not evidence that the extra history improved the recommendation.

## 4. Drawbacks and tradeoffs

- **Extra work:** one checker-agent invocation is added per round. On these six runs the merged version averaged 325.5 seconds and the baseline averaged 133.0 seconds.
- **Growing prompts:** every later participant receives the accumulated history. A larger round limit increases input size.
- **Checker errors can spread:** a mistaken verdict or a misread snippet is stored and can influence later arguments and the moderator.
- **Selected coverage:** the checker is instructed to prioritize selected claims, not to verify every assertion.
- **Prompt-based behavior:** the code passes findings to participants. It does not force them to accept a correction, to search, or to use the requested verdict labels.
- **Source records are not proof:** a tool call and a URL do not establish that a snippet, or the full source page, supports a claim. The saved record contains queries and URLs.
- **Sequential state updates:** nodes return the full updated `history` list. Parallel writers would need another merge strategy.
- **Memory lasts for one run:** history stays in graph state for that invocation. The script does not configure a checkpointer.
- **Tracing is optional and local:** when enabled, the JSONL file can contain search text. It is not written during a normal run unless `DEBATE_TRACE_PATH` is set.
- **Diagram call removed from the run path:** the merged script and the runnable baseline leave `draw_mermaid_png` commented out. The exact instructor file still contains the call and is not the file the runner executes.

## 5. AI use disclosure

ChatGPT earlier merged the full-history and fact-checker implementations, drafted the first version of this note and the memo, and reported seven local checks. That harness is not in the project and was not executed again.

Cursor, in this session, copied and hashed the instructor baseline, created the project virtual environment, installed `requirements-lab5.txt`, started Ollama, downloaded `qwen2.5:7b`, added the optional tracer, wrote `offline_checks.py` and `eval_runner.py`, and ran the offline checks and the six live runs. The first baseline attempts failed because the tracer tried to replace `ChatOllama.invoke`. That wrapper was changed to recompile the same edges around the original moderator function. The debate prompts were not changed. The three baseline runs above are the runs after that fix. Cursor also opened the public merged-01 trace and saved the screenshots in `evidence/screenshots/`. No teammate review is claimed.

The supplied `CASE`, `ollama:qwen2.5:7b` model, and two-round limit remain in the merged script. No pipeline or supervisor change is part of this extension.
