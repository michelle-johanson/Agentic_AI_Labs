"""
Lab: Supervisor Pattern -- IT Helpdesk Triage
IS 590R -- AI and Agentic Systems

Three rounds of fixes documented there. Each round replaced a soft instruction
the model wasn't reliably following with a hard, code-level check:

  Round 1: "is this done?" (free-text guess) -> an explicit STATUS line,
           checked in code.
  Round 2: "avoid repeating a specialist" (free-text instruction) -> the
           supervisor hard-excludes already-tried categories in code.
  Round 3: "STATUS: RESOLVED" taken at face value -> the code now also
           requires a real diagnosis BEFORE that line, or it downgrades
           the claim to NOT RESOLVED rather than trusting a bare,
           content-free claim of success.
"""

import operator
from typing import Literal, Optional
from typing_extensions import Annotated, TypedDict
from langgraph.graph import START, StateGraph
from langgraph.types import Command
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("ollama:qwen2.5:7b") #openai:gpt-5-mini

MAX_ATTEMPTS = 4  # backstop only -- see supervisor's logic below.

ALL_CATEGORIES = ["PASSWORD", "HARDWARE", "SOFTWARE"]
CATEGORY_TO_NODE = {
    "PASSWORD": "password_worker",
    "HARDWARE": "hardware_worker",
    "SOFTWARE": "software_worker",
}

MIN_DIAGNOSIS_WORDS = 5  # a bare "STATUS: RESOLVED" with nothing else is
                          # not an acceptable diagnosis -- see
                          # _parse_specialist_reply below.


class TicketState(TypedDict):
    ticket_text: str
    history: Annotated[list[str], operator.add]
    attempts: int
    resolved: bool
    tried: Annotated[list[str], operator.add]


def _parse_specialist_reply(raw_text: str) -> tuple[bool, str]:
    """
    Shared parsing logic for all three specialists.

    Returns (resolved, entry_text). `resolved` is downgraded to False --
    even if the model wrote "STATUS: RESOLVED" -- when there's no real
    diagnosis before that line. This is a hard guardrail against a
    specialist that skips straight to the status line to satisfy the
    format requirement without doing the actual work: don't just check
    that the claim was MADE, check that something supports it.
    """
    claimed_resolved = "STATUS: RESOLVED" in raw_text.upper()
    body = raw_text.upper().split("STATUS:")[0].strip()

    if claimed_resolved and len(body.split()) < MIN_DIAGNOSIS_WORDS:
        entry = (
            f"{raw_text} "
            "[flagged: claimed RESOLVED with no supporting diagnosis -- "
            "treated as NOT RESOLVED]"
        )
        return False, entry

    return claimed_resolved, raw_text


def password_worker(state: TicketState) -> Command[Literal["supervisor"]]:
    response = model.invoke([
        {"role": "system", "content": (
            "You are the Password/Access specialist on an IT helpdesk. "
            "You handle login, password reset, VPN access, and "
            "account-permission issues ONLY.\n\n"
            "Respond in exactly two parts, in this order:\n"
            "1. One to two sentences: if the ticket IS about "
            "login/password/VPN/permissions, name the likely cause and "
            "the standard fix you're applying; if it is NOT, say so "
            "briefly and do not attempt to solve it yourself. Never skip "
            "this part -- a bare status line with no diagnosis is not an "
            "acceptable response.\n"
            "2. A final line, on its own, with no other text on that "
            "line: 'STATUS: RESOLVED' if you applied a fix that should "
            "work, or 'STATUS: NOT RESOLVED' if you declined or couldn't "
            "resolve it."
        )},
        {"role": "user", "content": (
            f"Ticket: {state['ticket_text']}\n\n"
            f"Already tried: {state['history']}"
        )},
    ])
    resolved, body_text = _parse_specialist_reply(response.text)
    entry = f"[Password/Access] {body_text}"
    return Command(
        goto="supervisor",
        update={
            "history": [entry],
            "attempts": state["attempts"] + 1,
            "resolved": resolved,
            "tried": ["PASSWORD"],
        },
    )


def hardware_worker(state: TicketState) -> Command[Literal["supervisor"]]:
    response = model.invoke([
        {"role": "system", "content": (
            "You are the Hardware specialist on an IT helpdesk. You "
            "handle physical-device issues ONLY (laptops, monitors, "
            "docks, peripherals, power/charging problems).\n\n"
            "Respond in exactly two parts, in this order:\n"
            "1. One to two sentences: if the ticket IS a hardware issue, "
            "name the likely cause and the standard fix you're applying "
            "(assume it works unless the ticket describes a clear "
            "hardware failure a simple fix wouldn't address, such as a "
            "unit needing physical replacement); if it is NOT a hardware "
            "issue, say so briefly and do not attempt to solve it "
            "yourself. Never skip this part -- a bare status line with "
            "no diagnosis is not an acceptable response.\n"
            "2. A final line, on its own, with no other text on that "
            "line: 'STATUS: RESOLVED' or 'STATUS: NOT RESOLVED'."
        )},
        {"role": "user", "content": (
            f"Ticket: {state['ticket_text']}\n\n"
            f"Already tried: {state['history']}"
        )},
    ])
    resolved, body_text = _parse_specialist_reply(response.text)
    entry = f"[Hardware] {body_text}"
    return Command(
        goto="supervisor",
        update={
            "history": [entry],
            "attempts": state["attempts"] + 1,
            "resolved": resolved,
            "tried": ["HARDWARE"],
        },
    )


def software_worker(state: TicketState) -> Command[Literal["supervisor"]]:
    response = model.invoke([
        {"role": "system", "content": (
            "You are the Software/License specialist on an IT helpdesk. "
            "You handle installs, licenses, and application errors "
            "ONLY.\n\n"
            "Respond in exactly two parts, in this order:\n"
            "1. One to two sentences: if the ticket IS a "
            "software/install/license issue, name the likely cause and "
            "the standard fix you're applying (e.g. repair the install, "
            "clear a corrupt cache/profile, disable a conflicting "
            "add-in, reapply a license); if it is NOT, say so briefly "
            "and do not attempt to solve it yourself. Never skip this "
            "part -- a bare status line with no diagnosis is not an "
            "acceptable response.\n"
            "2. A final line, on its own, with no other text on that "
            "line: 'STATUS: RESOLVED' or 'STATUS: NOT RESOLVED'."
        )},
        {"role": "user", "content": (
            f"Ticket: {state['ticket_text']}\n\n"
            f"Already tried: {state['history']}"
        )},
    ])
    resolved, body_text = _parse_specialist_reply(response.text)
    entry = f"[Software/License] {body_text}"
    return Command(
        goto="supervisor",
        update={
            "history": [entry],
            "attempts": state["attempts"] + 1,
            "resolved": resolved,
            "tried": ["SOFTWARE"],
        },
    )


def supervisor(
    state: TicketState,
) -> Command[Literal["password_worker", "hardware_worker", "software_worker", "__end__"]]:
    if state.get("resolved"):
        entry = f"[Supervisor] Ticket resolved after {state['attempts']} attempt(s)."
        return Command(goto="__end__", update={"history": [entry]})

    tried = state.get("tried", [])
    remaining = [c for c in ALL_CATEGORIES if c not in tried]

    if not remaining:
        entry = (
            f"[Supervisor] Escalating to a human after all specialists "
            f"({', '.join(tried)}) were tried without resolution."
        )
        return Command(goto="__end__", update={"history": [entry]})

    if state["attempts"] >= MAX_ATTEMPTS:
        entry = (
            f"[Supervisor] Escalating to a human after {state['attempts']} "
            "specialist attempts without resolution."
        )
        return Command(goto="__end__", update={"history": [entry]})

    if len(remaining) == 1:
        return Command(goto=CATEGORY_TO_NODE[remaining[0]])

    response = model.invoke([
        {"role": "system", "content": (
            "You triage IT helpdesk tickets. Based on the ticket and what "
            "specialists have already tried, respond with EXACTLY ONE "
            f"WORD from this list: {', '.join(remaining)}. Choose "
            "whichever of these is the best remaining match for the "
            "ticket. No punctuation, no explanation."
        )},
        {"role": "user", "content": (
            f"Ticket: {state['ticket_text']}\n\n"
            f"History so far: {state['history']}\n\n"
            f"Remaining untried specialists: {remaining}"
        )},
    ])

    decision = response.text.strip().upper()
    choice: Optional[str] = next((c for c in remaining if c in decision), None)
    if choice is None:
        choice = remaining[0]

    return Command(goto=CATEGORY_TO_NODE[choice])


graph = StateGraph(TicketState)
graph.add_node("supervisor", supervisor)
graph.add_node("password_worker", password_worker)
graph.add_node("hardware_worker", hardware_worker)
graph.add_node("software_worker", software_worker)
graph.add_edge(START, "supervisor")
supervisor_graph = graph.compile()
#supervisor_graph.get_graph().draw_mermaid_png(output_file_path="supervisor_workflow.png")


SAMPLE_TICKETS = [
    "I can't log into the VPN since this morning, it just says 'access denied'.",
    "My laptop won't turn on at all, the power light doesn't even come on.",
    "Excel keeps crashing every time I open a file with more than 2 tabs.",
]

if __name__ == "__main__":
    for ticket in SAMPLE_TICKETS:
        print(f"\n=== Ticket: {ticket} ===")
        result = supervisor_graph.invoke({
            "ticket_text": ticket,
            "history": [],
            "attempts": 0,
            "resolved": False,
            "tried": [],
        })
        for entry in result["history"]:
            print(entry)