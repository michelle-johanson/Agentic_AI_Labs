# Lab: Pipeline Pattern — Resume Screening

IS 590R — AI and Agentic Systems

## Scenario

An HR team wants every incoming application run through the same three
fixed steps, in the same order, every time:

    extract_skills  ->  score_candidate  ->  draft_response

No step reconsiders an earlier step's output. Once skills are extracted,
that's final input to scoring; once scored, that's final input to the
email draft. There's no branching and no loop — that's exactly what makes
this the **Pipeline** pattern from Module 9, and not a Handoff Chain
(which has a decision point and can loop) or a Supervisor (which routes
dynamically).

## What you're practicing

This is the first lab of the three, so the wiring itself is the new
material — you're writing the `add_node` / `add_edge` / `compile` calls
yourself, not just the node bodies. The Supervisor and Debate labs build on
different mechanics (`Command`-based routing, and a conditional-edge loop)
and assume you've got the basic `StateGraph` shape down from this one.