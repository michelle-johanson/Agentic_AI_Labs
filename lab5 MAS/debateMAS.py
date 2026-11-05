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

# Neutral third-party agent that verifies the most specific factual claim
# from each side after every round. It has web_search access so it can
# actually check the claims rather than just re-state them.
fact_checker = create_agent(
    model,
    tools=[web_search],
    system_prompt=(
        "You are a neutral fact-checker in a debate. "
        "Pick the single most specific, verifiable factual claim from each side's argument. "
        "Use web_search to check each claim. "
        "Return a 2-3 sentence summary: state whether each claim is supported, "
        "contradicted, or unverifiable based on what you found. Be concise and neutral."
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


class DebateState(TypedDict):
    case: str
    position_a: str
    position_b: str
    round: int
    max_rounds: int
    resolution: str
    fact_check_notes: list[str]  # one entry per round, appended by fact_checker


def agent_position_a(state: DebateState) -> dict:
    if state["round"] == 0:
        user_content = f"Case:\n{state['case']}\n\nMake your opening argument."
    else:
        user_content = (
            f"Case:\n{state['case']}\n\n"
            f"The BUY side just argued:\n{state['position_b']}\n\n"
            "Rebut their argument and strengthen your case."
        )

    result = build_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    _describe_tool_usage("build_advocate", result["messages"])
    return {"position_a": result["messages"][-1].text, "round": state["round"] + 1}


def agent_position_b(state: DebateState) -> dict:
    user_content = (
        f"Case:\n{state['case']}\n\n"
        f"The BUILD side just argued:\n{state['position_a']}\n\n"
        "Rebut their argument and strengthen your case."
    )

    result = buy_advocate.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    _describe_tool_usage("buy_advocate", result["messages"])
    return {"position_b": result["messages"][-1].text}


def run_fact_checker(state: DebateState) -> dict:
    """
    Runs after each buy_advocate turn. Picks the most specific verifiable
    claim from each side's round argument, searches the web to check it,
    and appends a plain-English finding to fact_check_notes in the state.
    The moderator receives all notes so its final recommendation is grounded
    in what was actually verified, not just what was asserted.
    """
    user_content = (
        f"Round {state['round']} arguments:\n\n"
        f"BUILD side argued:\n{state['position_a']}\n\n"
        f"BUY side argued:\n{state['position_b']}\n\n"
        "Fact-check the most specific verifiable claim from each side."
    )
    result = fact_checker.invoke(
        {"messages": [{"role": "user", "content": user_content}]}
    )
    _describe_tool_usage("fact_checker", result["messages"])
    note = f"[Round {state['round']} fact-check] {result['messages'][-1].text}"
    print(f"\n--- Fact-check note (Round {state['round']}) ---\n{note}\n")
    return {"fact_check_notes": state["fact_check_notes"] + [note]}


def moderator(state: DebateState) -> dict:
    # Include fact-check notes so the moderator's recommendation is grounded
    # in what was independently verified, not just what was asserted.
    notes_text = (
        "\n\nFact-check findings from each round:\n" +
        "\n".join(state["fact_check_notes"])
        if state["fact_check_notes"] else ""
    )
    response = model.invoke([
        {"role": "system", "content": (
            "You are a neutral moderator. Read both sides' final "
            "arguments and the fact-check findings, then recommend BUILD or BUY. "
            "Name the single most important trade-off explicitly, and state "
            "the recommendation in the first sentence."
        )},
        {"role": "user", "content": (
            f"Case:\n{state['case']}\n\n"
            f"BUILD side's final argument:\n{state['position_a']}\n\n"
            f"BUY side's final argument:\n{state['position_b']}"
            f"{notes_text}"
        )},
    ])
    return {"resolution": response.text}


def continue_debate(state: DebateState) -> str:
    if state["round"] < state["max_rounds"]:
        return "agent_position_a"
    return "moderator"


graph = StateGraph(DebateState)
graph.add_node("agent_position_a", agent_position_a)
graph.add_node("agent_position_b", agent_position_b)
graph.add_node("fact_checker", run_fact_checker)  # new node between rounds
graph.add_node("moderator", moderator)
graph.add_edge(START, "agent_position_a")
graph.add_edge("agent_position_a", "agent_position_b")
graph.add_edge("agent_position_b", "fact_checker")  # always fact-check after each round
graph.add_conditional_edges(
    "fact_checker", continue_debate, ["agent_position_a", "moderator"]  # loop or finish
)
graph.add_edge("moderator", END)
debate_graph = graph.compile()
debate_graph.get_graph().draw_mermaid_png(output_file_path="debate_workflow.png")


if __name__ == "__main__":
    result = debate_graph.invoke({
        "case": CASE,
        "position_a": "",
        "position_b": "",
        "round": 0,
        "max_rounds": 2,
        "resolution": "",
        "fact_check_notes": [],  # starts empty; fact_checker appends one per round
    })

    print("\n--- Final BUILD argument ---")
    print(result["position_a"])
    print("\n--- Final BUY argument ---")
    print(result["position_b"])
    print("\n--- Fact-check notes (all rounds) ---")
    for note in result["fact_check_notes"]:
        print(note)
    print("\n--- Moderator's resolution ---")
    print(result["resolution"])