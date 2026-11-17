# Source excerpts used for the baseline comparison

These are code excerpts, not live model outputs.

## Instructor moderator input

The preserved instructor file sends only the latest argument from each side:

```
Case:
{state['case']}

BUILD side's final argument:
{state['position_a']}

BUY side's final argument:
{state['position_b']}
```

There is no `history` field and no fact-checker node. After BUY, `continue_debate()` either returns to BUILD or goes to the moderator.

## Merged checker and history

`run_fact_checker()` sends the case assumptions and `format_history(state)`, then appends:

`Round {round} - FACT CHECK:` plus the model text plus `_fact_check_search_record()`.

If the checker makes no tool call, the program appends:

`Search record written by the program: no web search was performed. Treat this report as unverified.`

The offline log quotes that sentence from a stub run. No live search URL was retrieved.
