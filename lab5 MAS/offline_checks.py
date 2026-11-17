"""Offline checks for the shared fact-checked debate history.

These checks import the merged debateMAS module and drive its compiled
LangGraph with stub advocates, a stub checker, and a stub moderator.
They do not call Ollama, Tavily, or LangSmith.

Run from this directory:

    python offline_checks.py
"""

from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("TAVILY_API_KEY", "tvly-offline-placeholder")
os.environ.pop("DEBATE_TRACE_PATH", None)

import debateMAS as debate


class Msg:
    def __init__(self, text="", tool_calls=None, msg_type="ai", content=None):
        self.text = text
        self.tool_calls = tool_calls or []
        self.type = msg_type
        self.content = text if content is None else content


class SequencedAgent:
    def __init__(self, responses):
        self.responses = responses
        self.calls = []

    def invoke(self, payload):
        self.calls.append(payload["messages"][0]["content"])
        messages = self.responses[len(self.calls) - 1]
        return {"messages": messages}


class ModelStub:
    def __init__(self, text):
        self.text = text
        self.calls = []

    def invoke(self, messages):
        self.calls.append(messages)
        return Msg(self.text)


def _check(name, ok, detail):
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}")
    return ok


def check_malformed_source_records():
    skipped = debate._fact_check_search_record([Msg("No tools used.")])
    malformed = [
        Msg("", tool_calls=[{"name": "web_search", "args": {"query": "crm cost"}}]),
        Msg("", msg_type="tool", content="{not json"),
        Msg(
            "",
            msg_type="tool",
            content=json.dumps({"results": [{"url": ""}, {"title": "missing"}, "bad"]}),
        ),
        Msg("", msg_type="tool", content=json.dumps([{"url": "https://example.com/ok"}])),
    ]
    parsed = debate._fact_check_search_record(malformed)
    no_urls = debate._fact_check_search_record([
        Msg("", tool_calls=[{"name": "web_search", "args": {"query": "x"}}]),
        Msg("", msg_type="tool", content="not-json-and-not-a-url-list"),
    ])
    already_parsed = debate._extract_urls({"results": [{"url": "https://example.com/dict"}]})
    ok = (
        "no web search was performed" in skipped
        and "Treat this report as unverified." in skipped
        and "https://example.com/ok" in parsed
        and "No source URLs could be parsed" not in parsed
        and "No source URLs could be parsed" in no_urls
        and "unverified" in no_urls
        and already_parsed == ["https://example.com/dict"]
    )
    return _check(
        "T6 malformed results",
        ok,
        "skipped search warns; bad JSON does not crash; a usable URL is kept",
    )


def check_compiled_graph():
    tool_payload = json.dumps({
        "results": [
            {"url": "https://example.com/crm-pricing"},
            {"url": "https://example.com/crm-pricing"},
        ]
    })
    build = SequencedAgent([
        [Msg("BUILD-R1 custom fit beats a partial SaaS product.")],
        [Msg("BUILD-R2 withdraws the contradicted vendor claim and keeps the fit argument.")],
    ])
    buy = SequencedAgent([
        [Msg("BUY-R1 a SaaS CRM is cheaper and already covers most requirements.")],
        [Msg("BUY-R2 accepts the checker correction and argues the remaining workaround cost.")],
    ])
    checker = SequencedAgent([
        [
            Msg("", tool_calls=[{"name": "web_search", "args": {"query": "SaaS CRM pricing"}}]),
            Msg("", msg_type="tool", content=tool_payload),
            Msg(
                "BUILD claim that the named vendor has no export API is contradicted. "
                "Source URL: https://example.com/crm-pricing"
            ),
        ],
        [Msg("Round 2 claims add no new external fact. No search was run.")],
    ])
    moderator = ModelStub(
        "BUY. The central tradeoff is exact process fit against lower annual cost."
    )
    debate.build_advocate = build
    debate.buy_advocate = buy
    debate.fact_checker = checker
    debate.model = moderator

    original_history = []
    node_order = []
    history = []
    resolution = ""
    for update in debate.debate_graph.stream(
        {
            "case": debate.CASE,
            "position_a": "",
            "position_b": "",
            "history": original_history,
            "round": 0,
            "max_rounds": 2,
            "resolution": "",
        },
        stream_mode="updates",
    ):
        node, payload = next(iter(update.items()))
        node_order.append(node)
        history = payload.get("history", history)
        resolution = payload.get("resolution", resolution)

    labels = [entry.split("\n", 1)[0] for entry in history]
    expected_labels = [
        "Round 1 - BUILD:",
        "Round 1 - BUY:",
        "Round 1 - FACT CHECK:",
        "Round 2 - BUILD:",
        "Round 2 - BUY:",
        "Round 2 - FACT CHECK:",
    ]
    results = []
    results.append(_check(
        "T1 retention and order",
        labels == expected_labels and original_history == [],
        f"entries={labels}; original history untouched={original_history == []}",
    ))
    second_checker_input = checker.calls[1] if len(checker.calls) > 1 else ""
    results.append(_check(
        "T2 checker context",
        all(piece in second_checker_input for piece in (
            "BUILD-R1",
            "BUY-R1",
            "BUILD-R2",
            "BUY-R2",
            "https://example.com/crm-pricing",
            "contradicted",
        )),
        "second checker input contains both rounds and the first report",
    ))
    second_build = build.calls[1] if len(build.calls) > 1 else ""
    second_buy = buy.calls[1] if len(buy.calls) > 1 else ""
    results.append(_check(
        "T3 feedback to advocates",
        "https://example.com/crm-pricing" in second_build
        and "contradicted" in second_build
        and "https://example.com/crm-pricing" in second_buy
        and "contradicted" in second_buy
        and "Search record written by the program" in second_build
        and "Search record written by the program" in second_buy,
        "both second-round advocate prompts include the first report and source record",
    ))
    moderator_user = ""
    if moderator.calls:
        moderator_user = moderator.calls[0][1]["content"]
    results.append(_check(
        "T4 moderator context",
        all(label.rstrip(":") in moderator_user or label in moderator_user for label in expected_labels)
        and moderator_user.count("Search record written by the program") == 2
        and "https://example.com/crm-pricing" in moderator_user
        and resolution.startswith("BUY."),
        f"moderator history entries present; resolution starts: {resolution[:60]!r}",
    ))
    results.append(_check(
        "T5 skipped search",
        "no web search was performed" in history[-1]
        and "Treat this report as unverified." in history[-1],
        "second checker made no tool call and the program wrote the warning",
    ))
    results.append(_check(
        "T7 routing",
        node_order == [
            "agent_position_a",
            "agent_position_b",
            "fact_checker",
            "agent_position_a",
            "agent_position_b",
            "fact_checker",
            "moderator",
        ],
        f"compiled graph node order={node_order}",
    ))
    return results


def main():
    print("Offline checks use stub agents and the compiled merged graph.")
    print("They are not live Ollama, Tavily, or LangSmith results.")
    results = [check_malformed_source_records(), *check_compiled_graph()]
    passed = sum(bool(item) for item in results)
    print(f"\n{passed}/{len(results)} checks passed")
    return 0 if passed == len(results) else 1


if __name__ == "__main__":
    sys.exit(main())
