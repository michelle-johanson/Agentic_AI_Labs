# Lab 5 improvements

Compared this folder with the instructor copy at
`/Users/jensroylance/Desktop/School Stuff/Fall 2026/IS590R Agentic AI/Labs/lab5 - MAS`.
`debateMAS.md`, `pipelineMAS.md`, and `supervisorMAS.md` are identical.
The three Python files differ. These scripts were not run for this writeup,
so there are no before/after output numbers.

# Improvement: full debate transcript

## 1. What we extended

In `debateMAS.py`:

- `DebateState` gained `history: list[str]`. The initial state in
  `debate_graph.invoke(...)` starts it as `[]`.
- New `format_history(state)` joins that list for the prompts.
- `agent_position_a` and `agent_position_b` now prompt from
  `format_history(state)` instead of only the other side's latest
  `position_a` / `position_b`. Each appends its own argument as
  `Round N - BUILD` or `Round N - BUY`.
- `moderator` now reads `format_history(state)` instead of the two final
  arguments. Its system prompt tells it to weigh evidence and unanswered
  rebuttals from every round, and not to treat a repeated claim as extra
  evidence.

## 2. Why this and not the alternatives

The instructor version only stored the latest argument on each side.
With `max_rounds` at 2, round 2 and the moderator never saw round 1.

Rejected:

- A message log kept only inside each advocate. The moderator is a plain
  `model.invoke()` and can only see `DebateState`, so the transcript has
  to live on the graph state.
- Raising `max_rounds`. Extra rounds still drop earlier turns if only the
  latest `position_a` and `position_b` are kept.

## 3. Benefits

Not measured. No eval was run, and the scripts were not executed for this
comparison.

The behavior change is that round 2 and the moderator receive every prior
argument, not just the previous turn. On a 2-round run that is four
arguments (BUILD, BUY, BUILD, BUY) instead of the last two.

## 4. Drawbacks and tradeoffs

- Each advocate's prompt grows by one full argument every turn. At
  `max_rounds = 2` that is small. A higher cap would send the whole
  transcript twice per round.
- `position_a` and `position_b` still hold only the latest argument, so
  they repeat the last two history entries. The printed "Final BUILD /
  BUY argument" output still uses those fields.
- `history` is a plain list with no LangGraph reducer. Each node returns
  the full list (`state["history"] + [...]`). A later node that returned
  a partial list would replace the transcript instead of appending.

# Other diffs (not scored as improvements)

These are setup changes, not extensions of the lab pattern. They were not
measured.

- `pipelineMAS.py`: `init_chat_model` changed from `ollama:mistral` to
  `ollama:qwen2.5:3b`. `pipeline_graph.get_graph().draw_mermaid_png(...)`
  was uncommented. That write produced `pipeline_workflow.png`, which is
  only in this copy.
- `supervisorMAS.py`: `init_chat_model` changed from `ollama:qwen2.5:7b`
  to `ollama:qwen2.5:3b`. `supervisor_graph.get_graph().draw_mermaid_png(...)`
  was uncommented, but `supervisor_workflow.png` is not in this folder.

`debateMAS.py` already called `draw_mermaid_png` in the instructor copy,
so that line is not a change here. `debate_workflow.png` is not in this
folder either.
