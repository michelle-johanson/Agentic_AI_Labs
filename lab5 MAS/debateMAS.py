"""
Lab: Debate Pattern -- Build vs. Buy (with live web search) -- SOLUTION
IS 590R -- AI and Agentic Systems
"""

import json
from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langchain.agents import create_agent
from langchain.chat_models import init_chat_model
from langchain_tavily import TavilySearch

load_dotenv()

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
        "3-4 sentences, citing what you found."
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
        "BUILD side's latest argument, citing what you found."
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


NO_SEARCH_NOTE = "No web search on this turn."


def _collect_search_record(messages) -> dict:
    """
    Pull the search facts out of one advocate invoke: how many tool calls
    fired, the arguments sent, and every URL that parsed out of a tool
    result. The argument text is not a reliable record of this -- a
    paragraph can cite prices with no search behind it.
    """
    tool_calls = 0
    queries: list[str] = []
    urls: list[str] = []
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
            tool_calls += 1
            queries.append(f"{call.get('name')} {call.get('args')}")
        if getattr(message, "type", None) == "tool":
            urls.extend(_extract_urls(message.content))
    return {"tool_calls": tool_calls, "queries": queries, "urls": urls}


def _format_search_record(record: dict) -> str:
    """
    Text the other advocate and the moderator actually read. The zero-search
    line is written here, in code, so it does not depend on the model
    admitting that it skipped the tool.
    """
    if record["tool_calls"] == 0:
        return NO_SEARCH_NOTE

    lines = [f"Web searches this turn: {record['tool_calls']}"]
    if record["queries"]:
        lines.append("Queries:")
        lines.extend(f"- {query}" for query in record["queries"])
    if record["urls"]:
        lines.append("URLs returned:")
        lines.extend(f"- {url}" for url in record["urls"])
    else:
        lines.append("No URLs could be parsed from the tool results.")
    return "\n".join(lines)


def _describe_tool_usage(label: str, messages) -> dict:
    """
    Print every tool call's name + arguments (the actual search query
    sent) and, when parseable, every URL the tool returned. Returns the
    count, queries, and parsed URLs so the debate state can keep them.
    A tool result that does not parse still prints a 200-character
    preview here; that preview is not copied into the state.
    """
    record = _collect_search_record(messages)
    for message in messages:
        for call in getattr(message, "tool_calls", None) or []:
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

    if record["tool_calls"] == 0:
        print(f"[{label}] made NO tool calls -- this argument is from the model's own knowledge only.")
    return record


def _opponent_block(side: str, argument: str, sources: str) -> str:
    return (
        f"The {side} side just argued:\n{argument}\n\n"
        "Search record for that argument (written by the program, not by the advocate):\n"
        f"{sources}\n\n"
        "Rebut their argument and strengthen your case."
    )


class DebateState(TypedDict):
    case: str
    position_a: str
    position_b: str
    sources_a: str
    sources_b: str
    round: int
    max_rounds: int
    resolution: str


def agent_position_a(state: DebateState) -> dict:
    if state["round"] == 0:
        user_content = f"Case:\n{state['case']}\n\nMake your opening argument."
    else:
        user_content = (
            f"Case:\n{state['case']}\n\n"
            + _opponent_block("BUY", state["position_b"], state["sources_b"])
        )

    result = build_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    record = _describe_tool_usage("build_advocate", result["messages"])
    return {
        "position_a": result["messages"][-1].text,
        "sources_a": _format_search_record(record),
        "round": state["round"] + 1,
    }


def agent_position_b(state: DebateState) -> dict:
    user_content = (
        f"Case:\n{state['case']}\n\n"
        + _opponent_block("BUILD", state["position_a"], state["sources_a"])
    )

    result = buy_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    record = _describe_tool_usage("buy_advocate", result["messages"])
    return {
        "position_b": result["messages"][-1].text,
        "sources_b": _format_search_record(record),
    }


def moderator_user_content(state: DebateState) -> str:
    return (
        f"Case:\n{state['case']}\n\n"
        f"BUILD side's final argument:\n{state['position_a']}\n\n"
        f"BUILD search record (written by the program):\n{state['sources_a']}\n\n"
        f"BUY side's final argument:\n{state['position_b']}\n\n"
        f"BUY search record (written by the program):\n{state['sources_b']}"
    )


def moderator(state: DebateState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "You are a neutral moderator. Read both sides' final "
            "arguments and recommend BUILD or BUY. Name the single most "
            "important trade-off explicitly, and state the recommendation "
            "in the first sentence. Each side has a search record written "
            "by the program. A price, vendor, or case-study claim from a "
            "side whose search record is 'No web search on this turn' is "
            "unverified. Do not treat that claim as a fact."
        )},
        {"role": "user", "content": moderator_user_content(state)},
    ])
    return {"resolution": response.text}


def continue_debate(state: DebateState) -> str:
    if state["round"] < state["max_rounds"]:
        return "agent_position_a"
    return "moderator"


graph = StateGraph(DebateState)
graph.add_node("agent_position_a", agent_position_a)
graph.add_node("agent_position_b", agent_position_b)
graph.add_node("moderator", moderator)
graph.add_edge(START, "agent_position_a")
graph.add_edge("agent_position_a", "agent_position_b")
graph.add_conditional_edges(
    "agent_position_b", continue_debate, ["agent_position_a", "moderator"]
)
graph.add_edge("moderator", END)
debate_graph = graph.compile()
debate_graph.get_graph().draw_mermaid_png(output_file_path="debate_workflow.png")


if __name__ == "__main__":
    result = debate_graph.invoke({
        "case": CASE,
        "position_a": "",
        "position_b": "",
        "sources_a": "",
        "sources_b": "",
        "round": 0,
        "max_rounds": 2,
        "resolution": "",
    })

    print("\n--- Final BUILD argument ---")
    print(result["position_a"])
    print("\n--- BUILD search record ---")
    print(result["sources_a"])
    print("\n--- Final BUY argument ---")
    print(result["position_b"])
    print("\n--- BUY search record ---")
    print(result["sources_b"])
    print("\n--- Moderator's resolution ---")
    print(result["resolution"])