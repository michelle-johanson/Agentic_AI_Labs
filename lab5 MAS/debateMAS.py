"""
Lab: Debate Pattern -- Build vs. Buy with shared fact-checked history
IS 590R -- AI and Agentic Systems
"""

import json
import os
from datetime import datetime, timezone

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

load_dotenv()


def _append_trace(record: dict) -> None:
    """Append one observation when DEBATE_TRACE_PATH is set.

    This records prompts, tool activity, and outputs. It does not alter
    the debate, add evidence, or change a participant's response.
    """
    path = os.environ.get("DEBATE_TRACE_PATH")
    if not path:
        return
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    record = {"ts": datetime.now(timezone.utc).isoformat(), **record}
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")


def _trace_messages(messages) -> list[dict]:
    traced = []
    for message in messages:
        content = getattr(message, "content", None)
        if isinstance(content, str) and len(content) > 50000:
            content = content[:50000] + "\n...[truncated by tracer]"
        traced.append({
            "type": getattr(message, "type", None),
            "text": getattr(message, "text", None),
            "content": content,
            "tool_calls": getattr(message, "tool_calls", None) or [],
            "usage_metadata": getattr(message, "usage_metadata", None),
            "response_metadata": getattr(message, "response_metadata", None),
            "id": getattr(message, "id", None),
        })
    return traced

model = init_chat_model("ollama:qwen2.5:7b")
web_search = TavilySearch(max_results=5, topic="general")

CASE = """
Decision: build a custom CRM in-house, or buy/license a SaaS CRM.
BUILD: ~$400K/year for 3 years (two engineers + infrastructure); exact fit
       to our sales process.
BUY:   ~$150K/year for a SaaS CRM; covers ~80% of requirements out of the
       box; the remaining 20% needs manual workaround processes.
"""

build_advocate = create_agent(
    model,
    tools=[web_search],
    system_prompt=(
        "You are an advocate arguing FOR building a custom CRM in-house. "
        "Use the web_search tool when it would strengthen your case with "
        "current pricing, vendor comparisons, or case studies -- don't "
        "just guess at numbers. Make the strongest honest case you can in "
        "3-4 sentences, citing what you found. Read the shared history, including "
        "fact-check findings. Correct or qualify claims challenged by evidence; "
        "do not repeat a contradicted claim without new supporting evidence."
    ),
)

buy_advocate = create_agent(
    model,
    tools=[web_search],
    system_prompt=(
        "You are an advocate arguing FOR buying a SaaS CRM instead of "
        "building in-house. Use the web_search tool when it would "
        "strengthen your case with current SaaS CRM pricing or feature "
        "comparisons -- don't just guess at numbers. Make the strongest "
        "honest case you can in 3-4 sentences, directly rebutting the "
        "BUILD side's latest argument, citing what you found. Read the shared "
        "history, including fact-check findings. Correct or qualify claims "
        "challenged by evidence; do not repeat a contradicted claim without "
        "new supporting evidence."
    ),
)


# One extension: a shared history of arguments and fact-check findings.
# The checker reads that history after every completed BUILD/BUY round.
fact_checker = create_agent(
    model,
    tools=[web_search],
    system_prompt=(
        "You are a neutral fact-checker reviewing a cumulative debate history. "
        "Treat CASE numbers as supplied assumptions, not external facts to verify. "
        "Prioritize one specific, decision-relevant factual claim from each side "
        "in the latest round. Use earlier arguments and fact-check findings to "
        "identify unresolved or repeated claims; do not needlessly recheck a "
        "settled claim unless it is contested or new evidence appears. "
        "Use web_search to investigate selected claims. For each, identify the "
        "side, quote the claim, and label it supported, contradicted, or "
        "unverifiable, with a short reason and a source URL returned by the tool. "
        "If a side has no new checkable claim, say so. If search is skipped or "
        "the results do not support a conclusion, say unverifiable. "
        "Do not invent sources or treat earlier checker notes as proof. "
        "If revising an earlier finding, explicitly identify the revision. "
        "Keep the complete report to 4-6 sentences."
    ),
)


def _extract_urls(raw_content) -> list[str]:
    """
    Best-effort extraction of result URLs from a Tavily tool result.
    Tavily's output shape can be a JSON string or an already-parsed
    dict/list depending on version -- deliberately defensive.
    """
    data = raw_content
    if isinstance(data, str):
        try:
            data = json.loads(data)
        except (ValueError, TypeError):
            return []

    results = data.get("results") if isinstance(data, dict) else data
    if not isinstance(results, list):
        return []

    return [r.get("url") for r in results if isinstance(r, dict) and r.get("url")]


def _describe_tool_usage(label: str, messages) -> None:
    """
    Prints every tool call's name + arguments (the actual search query
    sent) and, when parseable, every URL the tool returned. This is the
    only reliable way to confirm a tool fired -- the agent's final
    argument text alone can look authoritative with zero search behind
    it.
    """
    saw_any_call = False
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            saw_any_call = True
            print(f"[{label}] called {call.get('name')} with args: {call.get('args')}")

        if getattr(message, "type", None) == "tool":
            urls = _extract_urls(message.content)
            if urls:
                print(f"[{label}] tool returned {len(urls)} result(s):")
                for url in urls:
                    print(f"    - {url}")
            else:
                preview = str(message.content)[:200]
                print(f"[{label}] tool returned (no URLs parsed out; raw preview): {preview}")

    if not saw_any_call:
        print(f"[{label}] made NO tool calls -- this argument is from the model's own knowledge only.")


def _fact_check_search_record(messages) -> str:
    """Save actual checker tool activity alongside the model's findings.

    Only web_search is registered on this agent. This record documents
    tool use and returned URLs; it does not prove a claim is supported.
    """
    calls = []
    urls = []
    for message in messages:
        calls.extend(getattr(message, "tool_calls", None) or [])
        if getattr(message, "type", None) == "tool":
            urls.extend(url for url in _extract_urls(message.content) if isinstance(url, str))

    if not calls:
        return (
            "Search record written by the program: no web search was performed. "
            "Treat this report as unverified."
        )
    lines = [f"Search record written by the program: {len(calls)} tool call(s)."]
    for call in calls:
        lines.append(f"Query: {call.get('name')} {call.get('args')}")
    if urls:
        lines.append("URLs returned by the tool:")
        lines.extend(f"- {url}" for url in dict.fromkeys(urls))
    else:
        lines.append("No source URLs could be parsed; treat this report as unverified.")
    return "\n".join(lines)


class DebateState(TypedDict):
    case: str
    position_a: str
    position_b: str
    history: list[str]  # Chronological BUILD, BUY, and FACT CHECK entries.
    round: int
    max_rounds: int
    resolution: str


# All participants read this same history, including saved checker reports.
def format_history(state: DebateState) -> str:
    return "\n\n".join(state["history"]) or "No arguments yet."


def agent_position_a(state: DebateState) -> dict:
    # Extension: give BUILD all prior arguments, with a separate opening instruction.
    instruction = (
        "Make your opening argument."
        if state["round"] == 0
        else "Respond to the BUY side's latest argument, consider earlier "
             "points, and strengthen your case without unnecessary repetition."
    )
    user_content = (
        f"Case:\n{state['case']}\n\n"
        f"Debate so far:\n{format_history(state)}\n\n"
        f"{instruction}"
    )

    result = build_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    _describe_tool_usage("build_advocate", result["messages"])
    _append_trace({
        "event": "participant",
        "participant": "build_advocate",
        "round_before": state["round"],
        "input": user_content,
        "messages": _trace_messages(result["messages"]),
    })
    # Extension: preserve BUILD's argument while keeping the latest-position output.
    argument = result["messages"][-1].text
    current_round = state["round"] + 1
    return {
        "position_a": argument,
        "round": current_round,
        "history": state["history"] + [
            f"Round {current_round} - BUILD:\n{argument}"
        ],
    }


def agent_position_b(state: DebateState) -> dict:
    # Extension: give BUY the full transcript, including BUILD's newest argument.
    user_content = (
        f"Case:\n{state['case']}\n\n"
        f"Debate so far:\n{format_history(state)}\n\n"
        "Respond to the BUILD side's latest argument, consider earlier "
        "points, and strengthen your case without unnecessary repetition."
    )

    result = buy_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    _describe_tool_usage("buy_advocate", result["messages"])
    _append_trace({
        "event": "participant",
        "participant": "buy_advocate",
        "round_before": state["round"],
        "input": user_content,
        "messages": _trace_messages(result["messages"]),
    })
    # Extension: append BUY's argument using the round already incremented by BUILD.
    argument = result["messages"][-1].text
    return {
        "position_b": argument,
        "history": state["history"] + [
            f"Round {state['round']} - BUY:\n{argument}"
        ],
    }


def run_fact_checker(state: DebateState) -> dict:
    """Review the full history and append one sourced report for this round.

    Nodes execute sequentially, so each returns the entire updated list.
    The next advocates and the moderator see this report through history.
    """
    user_content = (
        f"Case assumptions:\n{state['case']}\n\n"
        f"Complete debate history:\n{format_history(state)}\n\n"
        f"Review round {state['round']}. Check the latest claims against "
        "external sources, taking earlier arguments and findings into account."
    )
    result = fact_checker.invoke({"messages": [{
        "role": "user",
        "content": user_content,
    }]})
    messages = result["messages"]
    _describe_tool_usage("fact_checker", messages)
    _append_trace({
        "event": "participant",
        "participant": "fact_checker",
        "round": state["round"],
        "input": user_content,
        "messages": _trace_messages(messages),
    })
    report = (
        f"Round {state['round']} - FACT CHECK:\n{messages[-1].text}\n\n"
        f"{_fact_check_search_record(messages)}"
    )
    print(f"\n--- Saved fact-check report ---\n{report}\n")
    return {"history": state["history"] + [report]}


def moderator(state: DebateState) -> dict:
    system_content = (
        "You are a neutral moderator. Review the complete debate "
        "history and recommend BUILD or BUY. Consider evidence and "
        "rebuttals from every round, including points left unanswered. "
        "Consider the saved fact-check findings and their program-recorded "
        "search activity. A report flagged unverified is not verification. "
        "A returned URL alone does not establish support for a claim. "
        "Distinguish CASE assumptions from externally checked claims, "
        "and acknowledge unresolved or conflicting findings. "
        "Do not treat repeated claims as additional evidence. "
        "Name the single most important trade-off explicitly, and "
        "state the recommendation in the first sentence."
    )
    user_content = (
        f"Case:\n{state['case']}\n\n"
        f"Complete debate history:\n{format_history(state)}"
    )
    response = model.invoke([
        # Extension: judge evidence and unanswered rebuttals across every round.
        {"role": "system", "content": system_content},
        # Extension: supply the complete transcript instead of only final arguments.
        {"role": "user", "content": user_content},
    ])
    _append_trace({
        "event": "participant",
        "participant": "moderator",
        "input": user_content,
        "messages": _trace_messages([response]),
    })
    return {"resolution": response.text}


def continue_debate(state: DebateState) -> str:
    if state["round"] < state["max_rounds"]:
        return "agent_position_a"
    return "moderator"


graph = StateGraph(DebateState)
graph.add_node("agent_position_a", agent_position_a)
graph.add_node("agent_position_b", agent_position_b)
graph.add_node("fact_checker", run_fact_checker)
graph.add_node("moderator", moderator)
graph.add_edge(START, "agent_position_a")
graph.add_edge("agent_position_a", "agent_position_b")
graph.add_edge("agent_position_b", "fact_checker")
graph.add_conditional_edges(
    "fact_checker", continue_debate, ["agent_position_a", "moderator"]
)
graph.add_edge("moderator", END)
debate_graph = graph.compile()
# Optional visualization (may require network access):
# debate_graph.get_graph().draw_mermaid_png(output_file_path="debate_workflow.png")


if __name__ == "__main__":
    run_id = os.environ.get("DEBATE_RUN_ID", "")
    if run_id:
        print(f"[run] {run_id}")
    started = datetime.now(timezone.utc)
    invoke_config = None
    if run_id:
        invoke_config = {
            "run_name": run_id,
            "tags": ["lab5", "merged", run_id],
            "metadata": {"variant": "merged", "run_id": run_id},
        }
    result = debate_graph.invoke(
        {
            "case": CASE,
            "position_a": "",
            "position_b": "",
            "history": [],  # Extension: start with an empty debate transcript.
            "round": 0,
            "max_rounds": 2,
            "resolution": "",
        },
        config=invoke_config,
    )
    elapsed = (datetime.now(timezone.utc) - started).total_seconds()

    print("\n--- Final BUILD argument ---")
    print(result["position_a"])
    print("\n--- Final BUY argument ---")
    print(result["position_b"])
    print("\n--- Complete debate history including fact checks ---")
    print(format_history(result))
    print("\n--- Moderator's resolution ---")
    print(result["resolution"])
    print(f"\n[elapsed_seconds] {elapsed:.1f}")

    state_path = os.environ.get("DEBATE_STATE_PATH")
    if state_path:
        parent = os.path.dirname(state_path)
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(state_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_id": run_id,
                    "elapsed_seconds": elapsed,
                    "round": result["round"],
                    "position_a": result["position_a"],
                    "position_b": result["position_b"],
                    "history": result["history"],
                    "resolution": result["resolution"],
                },
                handle,
                indent=2,
            )
    _append_trace({
        "event": "final",
        "run_id": run_id,
        "elapsed_seconds": elapsed,
        "round": result["round"],
        "history": result["history"],
        "resolution": result["resolution"],
    })
