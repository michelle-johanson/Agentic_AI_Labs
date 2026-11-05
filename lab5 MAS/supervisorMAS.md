# Lab: Supervisor Pattern — IT Helpdesk Triage

IS 590R — AI and Agentic Systems

## Scenario

An internal IT helpdesk receives free-text tickets. A **supervisor** agent
reads the ticket (and whatever specialists have already tried) and
decides, each time control returns to it, which specialist should look at
it next:

    Password/Access  |  Hardware  |  Software/License  |  Done

This is the **Supervisor** pattern from Module 9: one node makes a fresh
routing decision every time, using `Command(goto=..., update=...)`.
There's no fixed sequence of workers the way there was in the Pipeline
lab — the ticket about "can't log in" might turn out to actually be a
hardware token problem, and the supervisor needs to be able to re-route
based on what comes back.

Three sample tickets are provided and will each run through the graph.