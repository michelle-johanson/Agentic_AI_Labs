# Improvement 1: Keep each side's search record

## 1. What we extended

`DebateState` gained `sources_a` and `sources_b`. Those strings are the search record for the speech currently stored in `position_a` and `position_b`.

- `_collect_search_record()` walks one advocate's messages and returns `tool_calls`, `queries`, and `urls`. URL parsing is still `_extract_urls()`.
- `_format_search_record()` turns that dict into the text the other side and the moderator read. When `tool_calls` is 0, the text is the fixed line `No web search on this turn.`, written in code.
- `_describe_tool_usage()` still prints the calls and URLs, and now returns the record from `_collect_search_record()`. A tool body that does not parse still prints a 200-character preview, and that preview is not stored.
- `agent_position_a()` and `agent_position_b()` save `_format_search_record(...)` on the state. From the second speech on, the user message is built by `_opponent_block()`, which pastes the opponent's argument and that search record.
- `moderator_user_content()` adds both search records next to the final arguments. `moderator()` tells the model that a price, vendor, or case-study claim from a side whose record is `No web search on this turn` is unverified.

The advocates' system prompts are unchanged.

## 2. Why this and not the alternatives

`_describe_tool_usage()` already proved whether a search happened, then threw the result away. `moderator()` only received the case and the two final paragraphs. A paragraph can name prices with no search behind it, and the judge had no way to tell.

Rejected:

- **A stronger sentence in the advocate system prompt** ("you must search"). The supervisor lab already replaced that kind of instruction with a check in code. This file's zero-search line is written by `_format_search_record()`, so it shows up even when the model does not admit that it skipped the tool.
- **A full transcript of every round.** The moderator would still be unable to tell a real citation from an invented one. Round 1 is still overwritten; that is a separate change.
- **Checking that every dollar amount in the argument appears in a snippet.** Stronger, and it needs this record stored first. Not built.
- **A larger model or a second search tool.** The missing piece was the Tavily result this graph already fetched.

## 3. Benefits

No live debate has been run. The numbers below use one hand-built message list shaped like an advocate invoke: two `web_search` calls, one Tavily-style JSON string (two usable URLs, plus a result with no `url` and a result with an empty `url`), one already-parsed dict with one URL, and one tool message that is not JSON. The other side is a single AI message with no tool call, citing `$400K`. The "before" column is the previous `moderator` user message, rebuilt from the three fields it interpolated (`case`, `position_a`, `position_b`), on that same text.

| What reached the moderator | Before | After |
|---|---:|---:|
| URL lines | 0 | 3 |
| Copies of `No web search on this turn.` | 0 | 1 |
| Characters in the user message | 183 | 563 |

The rebuttal prompt for the searched side also went from 0 URL lines to 3. The three URLs are the two from the JSON string and the one from the parsed dict. The empty and missing `url` fields were dropped, and the non-JSON tool message did not crash parsing. `tool_calls` on that side is 2.

This shows the judge can see the sources. It does not show that the recommendation gets better. `qwen2.5:7b` varies from run to run, so that takes the same `CASE` three times before this change and three times after.

## 4. Drawbacks and tradeoffs

- **Only the latest turn is kept.** `sources_a` and `sources_b` are replaced the same way `position_a` and `position_b` are. With `max_rounds` at 2, each side's opening search record is gone before `moderator()` runs.
- **A search does not make the paragraph true.** The moderator is told to distrust claims only when the record is the zero-search line. A side can search, receive these URLs, and still invent a price. Nothing compares the argument text to the snippets.
- **The unverified rule is still a prompt.** The code forces the record into the moderator's user message. It does not read `resolution` to check that the model obeyed.
- **An unparsed tool body stays on the console.** The 200-character preview from `_describe_tool_usage()` is not copied into `sources_a` or `sources_b`. If every result fails to parse, the stored line is `No URLs could be parsed from the tool results.`
- **The prompt gets longer.** On this fixture the moderator user message grew from 183 to 563 characters, almost all of it two queries and three URLs. A real turn can attach up to 5 Tavily results per call, and one turn can call the tool more than once. The wording of `NO_SEARCH_NOTE` and the moderator sentence was chosen by eye.
- **Earlier debate evals:** there were none. Pipeline and supervisor code were not changed.

## Backlog (not implemented)

- Keep every round's argument and search record, not only the last turn.
- Check cited URLs and dollar amounts against the tool snippets, and drop claims that do not appear there.
- Check in code that the moderator's first sentence says BUILD or BUY.
- Run the same `CASE` three times before and three times after, and tally zero-search turns, first-sentence format, and prices repeated from a side that did not search.
- `draw_mermaid_png` still runs when the file loads. The other two labs have that line commented out.
