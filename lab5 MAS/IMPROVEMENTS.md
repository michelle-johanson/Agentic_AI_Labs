# Lab 5 improvement

## One change we extended

We added a **shared fact-checked debate history** to `debateMAS.py`. After each round, a fact-checker reads the accumulated arguments and earlier findings, investigates selected claims, and appends its report to the same history. Both advocates receive those findings before their next turn, and the moderator receives the full record before recommending BUILD or BUY.

This combines the full-transcript and fact-checker versions into one feedback process: preserve the debate, check claims in context, and make the findings available to every participant.

## 1. What we changed in the code

### Shared state

- `DebateState.history: list[str]` stores chronological entries labeled `Round N - BUILD`, `Round N - BUY`, and `Round N - FACT CHECK`.
- `format_history(state)` joins those entries for participant prompts.
- `agent_position_a()` and `agent_position_b()` read the entire history and append their new arguments. Their system prompts tell them to correct or qualify claims challenged by evidence.
- `position_a` and `position_b` still hold the latest arguments for the existing final-output print statements.
- Initial state includes `history: []`. There is no separate `fact_check_notes` list; reports live with the arguments they concern.

### Checking and saving findings

- A new `fact_checker` agent has the existing `web_search` tool.
- Its prompt prioritizes one specific, decision-relevant claim per side from the latest round. It uses earlier history to identify unresolved or repeated claims and is instructed to avoid redundant checks unless a claim is contested or new evidence appears.
- The checker is instructed to identify each selected claim, label it supported, contradicted, or unverifiable, and explain the finding with a returned source URL. These instructions are prompt-based; the code does not enforce a claim count or verdict format.
- `run_fact_checker()` passes the case assumptions and complete history to the checker, then appends its report to `history`.
- `_fact_check_search_record()` adds a program-written record of the checker's tool-call count, queries, and parsed, deduplicated source URLs to that report.
- If the checker makes no tool call, the program appends: `Search record written by the program: no web search was performed. Treat this report as unverified.` If calls occur but no usable URLs can be parsed, it also marks the report unverified.

Search records are saved for the **checker reports**. Advocate tool activity is still printed to the console; this version does not keep separate `sources_a` and `sources_b` fields.

### Graph and moderator

After BUILD and BUY speak, the graph always runs `fact_checker` before calling `continue_debate()`. That routing function sends execution back to BUILD if another round remains, or to the moderator after the final check.

`moderator()` reads the complete history, including reports from every round. Its system prompt tells it to consider those findings, distinguish case assumptions from externally checked claims, acknowledge unresolved findings, and avoid treating repeated claims or returned URLs alone as proof.

The main print block now prints the complete history, including the fact-check reports. Diagram generation is commented out as an optional visualization.

## 2. Why we chose this change

The documented baseline retained only the latest argument from each side. A full transcript fixes lost context, but does not investigate the claims it preserves. The separate fact-checker version produces findings for the moderator, but does not give those findings to the advocates, so they cannot respond to corrections in later rounds.

The merged version connects checking to the shared memory. A first-round correction can now reach both second-round advocates, remain available to the next check, and reach the final moderator.

Alternatives considered:

- **Keep only the transcript:** preserves earlier arguments without adding a way to check their factual claims.
- **Keep only the original checker:** adds checking but leaves corrections outside the advocates' prompts.
- **Add more debate rounds:** increases work without fixing missing memory or correction feedback.
- **Check every claim in every round:** expands search and prompt costs. The implemented prompt instead prioritizes selected claims, although actual model compliance still requires live evaluation.

## 3. Benefits and supporting evidence

Seven local checks passed against the merged code. The harness executed its actual functions and graph declarations with synthetic messages, stub agents, and a simulated graph runner. It did not initialize or run the real LangGraph runtime, Ollama, Tavily, or LangSmith.

| Check | Observed result |
| --- | --- |
| T1 — retention and order | Two rounds preserved four arguments and two checker reports in chronological order. The original input history was not mutated. |
| T2 — checker context | The second checker received both rounds' arguments and the complete first checker report. |
| T3 — feedback to advocates | Both second-round advocates received the first correction and its recorded source URL. |
| T4 — moderator context | The moderator received all six history entries and both checker source records. |
| T5 — skipped search | A checker response without tool calls received the program-written unverified warning. |
| T6 — malformed results | Malformed JSON and unusable URL fields did not crash source-record creation; missing usable URLs were flagged. |
| T7 — routing | Simulation of the declared graph ran BUILD, BUY, and the checker twice, then the moderator. |

The synthetic scenario used a predetermined false claim that a fictional vendor lacked exports, a predetermined checker correction, and a predetermined later withdrawal. This demonstrated that the correction reached later prompts; it did **not** demonstrate that a live model independently found or corrected an error. The example.com URLs were fixture data, not retrieved evidence.

For two rounds, the updated moderator receives four arguments and two reports instead of the documented baseline's final two arguments. Seven top-level advocate/checker/moderator invocations occur instead of five. Tool-using agents may perform additional internal model calls and searches.

No live recommendation-quality, factual-accuracy, latency, or cost improvement has been measured. The previous search-record version's five tests and 152-to-483-character prompt comparison concern a different implementation and are not evidence for this merged version.

## 4. Drawbacks and tradeoffs

- **Extra work:** one checker-agent invocation is added per round, with potential search calls and additional inference latency.
- **Growing prompts:** every participant repeatedly receives the accumulated history. Larger round limits increase input size and may eventually exceed the model's context window.
- **Checker errors can spread:** a mistaken verdict or misread source is preserved and can influence later arguments and the moderator.
- **Selected coverage:** the checker is instructed to prioritize selected claims, not verify every assertion in the history.
- **Prompt-based behavior:** the code passes findings to participants, but does not force them to accept corrections, search, use the requested verdict format, or avoid redundant checks.
- **Source records are not proof:** a tool call and returned URL do not establish that a claim is supported. The saved record contains queries and URLs, not complete source snippets or a programmatic claim-to-source comparison.
- **Sequential state updates:** nodes return the full updated `history` list. Parallel writers would require an appropriate reducer or another merge strategy to avoid conflicting updates.
- **Memory lasts for one run:** history is retained in graph state and printed at the end. This script does not configure a checkpointer or automatically save a transcript file for use across separate runs.

## 5. AI use disclosure

ChatGPT merged the full-history and fact-checker implementations at the user's request, updated participant prompts and graph routing, added checker source records, and drafted this explanation and the matching memo. It also created and ran the seven local checks described above. The checks support claims about state retention, routing, and delivery of findings, not live factual accuracy or improved decision quality.

## Live evaluation still needed

1. Run the same `CASE` three times on the baseline and three times on the merged version.
2. Inspect whether the checker actually searches, whether its verdicts match the retrieved evidence, and whether corrected errors recur in later arguments or the final recommendation.
3. Compare total run time and record actual model/search usage where available.
4. Add public LangSmith run links and project-log screenshots to the memo, including a first-round checker report, both second-round advocate inputs, and the moderator's full-history input.

The existing `ollama:qwen2.5:7b` model, Tavily configuration, case assumptions, and two-round limit remain in the merged script. No pipeline or supervisor changes are part of this extension.
