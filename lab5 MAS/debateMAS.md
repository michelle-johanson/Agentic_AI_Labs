# Lab: Debate Pattern — Build vs. Buy (with live web search)

IS 590R — AI and Agentic Systems

## Scenario

Your company needs a new CRM. Finance has quoted a build-vs-buy decision:

- **BUILD:** ~$400K/year for 3 years (two engineers + infrastructure) for
  a custom CRM that exactly fits your sales process.
- **BUY:** ~$150K/year for a SaaS CRM that covers ~80% of what you need
  out of the box, with the remaining 20% requiring workaround processes.

Two agents argue fixed, opposing positions across a capped number of
rounds — each round responding directly to the other side's prior
argument — and a moderator agent synthesizes a final recommendation once
the rounds are up. This is the **Debate** pattern from Module 9.

Both debate agents can now search the web (via [Tavily](https://tavily.com))
for current pricing, vendor comparisons, or case studies instead of
arguing purely from the model's static training data. The moderator stays
a plain model call — it's judging arguments already made, not researching
new ones.

## Setup

1. Sign up for a free [Tavily](https://tavily.com) API key.
2. Add it to your `.env` file alongside your other provider keys:
   ```
   TAVILY_API_KEY=tvly-...
   ```
3. Install the package:
   ```
   pip install -U langchain-tavily
   ```

## What you're practicing

Unlike the Supervisor lab, the looping mechanic here is **not**
`Command`-based — it's a conditional edge (`add_conditional_edges`) that
keeps sending control back to `agent_position_a` until a round cap is hit,
then releases it to the moderator.

Note the shape of `build_advocate` and `buy_advocate`: each is a full 
`create_agent(...)` tool-calling agent, not a single
plain model call. That's deliberate — a node in a graph can be as simple
as one `model.invoke()` (like `moderator` still is) or as sophisticated as
a whole agent with its own tool-use loop. The Debate *pattern* doesn't
care which; the pattern is about the control flow between nodes, not what
happens inside any one of them.