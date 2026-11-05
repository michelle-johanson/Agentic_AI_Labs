# Lab 5 improvements

Compared this folder with the instructor copy at
`/Users/jensroylance/Desktop/School Stuff/Fall 2026/IS590R Agentic AI/Labs/lab5 - MAS`.
`debateMAS.md`, `pipelineMAS.md`, and `supervisorMAS.md` are identical.
The three Python files differ. These scripts were not run for this writeup,
so there are no before/after output numbers.

# Improvement: fact-checker between rounds

## 1. What we extended

In `debateMAS.py`:

- New `fact_checker` agent from `create_agent`, with `tools=[web_search]`.
  Its system prompt tells it to pick the single most specific verifiable
  claim from each side, check both with `web_search`, and return a 2-3
  sentence summary (supported, contradicted, or unverifiable).
- `DebateState` gained `fact_check_notes: list[str]`. The initial state in
  `debate_graph.invoke(...)` starts it as `[]`.
- New node `run_fact_checker`. It builds a prompt from the current
  `position_a` and `position_b`, calls `fact_checker`, and returns
  `state["fact_check_notes"] + [note]`.
- The graph used to go `agent_position_b` straight into `continue_debate`.
  It now goes `agent_position_b` → `fact_checker` → `continue_debate`
  (back to `agent_position_a`, or on to `moderator`).
- `moderator` now joins `fact_check_notes` into the user message, and its
  system prompt tells it to read those findings before recommending
  BUILD or BUY.
- The `__main__` print block prints every note under
  "Fact-check notes (all rounds)".

## 2. Why this and not the alternatives

The moderator is a plain `model.invoke()` with no tools, so it cannot
search on its own. A check has to be a separate agent, and the result
has to live on `DebateState` or the moderator never sees it.

The diff does not record which alternatives were rejected. Two choices
are visible in the code:

- Notes are stored for the moderator only. `agent_position_a` and
  `agent_position_b` still prompt from the other side's latest argument
  and never read `fact_check_notes`.
- The check runs after every round, not once at the end. `continue_debate`
  sits after `fact_checker`. With `max_rounds = 2` that is two checks.

## 3. Benefits

Not measured. No eval was run, and the scripts were not executed for this
comparison.

The behavior change is that the moderator receives one fact-check note
per round plus the final `position_a` and `position_b`. On a 2-round run
that is two notes. The advocates still only see the other side's latest
argument.

## 4. Drawbacks and tradeoffs

- Round 2 cannot answer a correction. The advocates never read
  `fact_check_notes`, so a claim marked contradicted in round 1 can be
  repeated in round 2.
- Each round adds one fact-checker call, and that agent may call Tavily.
  At `max_rounds = 2` that is two extra agent turns on top of the four
  advocate turns.
- The prompt asks for only the single most specific claim per side, so
  the other claims in that round are unchecked.
- `fact_check_notes` is a plain list with no LangGraph reducer.
  `run_fact_checker` returns the full list. A later node that returned
  a partial list would replace the notes instead of appending.
- The moderator still sees only the last argument on each side, same as
  the instructor graph. Earlier rounds survive only inside the notes.

# Other diffs (not scored as improvements)

These are setup changes, not extensions of the lab pattern. They were not
measured.

- `pipelineMAS.py`: `init_chat_model` changed from `ollama:mistral` to
  `ollama:qwen2.5:7b`. The `draw_mermaid_png` line stays commented.
- `supervisorMAS.py`: `supervisor_graph.get_graph().draw_mermaid_png(...)`
  was uncommented. `supervisor_workflow.png` is not in this folder; it is
  written when the script runs. The model line is still `ollama:qwen2.5:7b`.
