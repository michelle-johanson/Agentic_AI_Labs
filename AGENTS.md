# Working on lab extension memos

Each lab has a baseline system from the instructor, and my job is to find and build improvements to it, then write them up. `IMPROVEMENTS.md` is the working log; `MEMO.md` is the final submission. This file is about the working log part.

## How I want to work

- Explain the baseline code before changing it — what each file does, and how to read anything unusual. I have some experiance with Python but don't code with it regularly.
- Show me the problems in the *actual data* (run queries, print counts), not generic advice about the topic. Concrete failures are what give me ideas.
- Give me a chance to spot things myself before you list your own ideas. Ask me which problem bugs me most and what I'd try. Tell me if my idea is a good direction and what it'd cost.
- One improvement at a time. Build it, measure it, write it up, then go looking for the next thing.
- Keep a short backlog of ideas we noticed but haven't built.
- Don't run the venv in a shell. For long commands and processes, ask me to run it.

## Building an improvement

- **Measure, don't assert.** Every improvement needs a before/after number on real queries — build a small eval and show the table. If the gain is small, say so.
- **Re-run the earlier evals after every change.** Features interact; twice this has silently broken something that used to work.
- **Try to break it.** Run adversarial/edge-case inputs and put the real failures in the writeup. An honest drawbacks section is worth more than a clean one.
- Prefer improvements that get more out of the model/tools we were given.

## Writing it up in IMPROVEMENTS.md

Four questions per improvement, in this order:

1. What we extended (name the actual function/parameter/field)
2. Why this and not the alternatives (list what we rejected and why)
3. Benefits (the before/after numbers)
4. Drawbacks and tradeoffs (the real ones, including what we tuned by eye)

Write it as we go, not at the end.
