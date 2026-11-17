import re
from decimal import Decimal

from dotenv import load_dotenv
from typing_extensions import TypedDict
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langchain.agents import create_agent
from langchain.agents.middleware import PIIMiddleware, HumanInTheLoopMiddleware
from langchain.chat_models import init_chat_model
from langchain_core.tools import tool

load_dotenv()

model = init_chat_model("ollama:qwen3") #anthropic:claude-haiku-4-5

# US-style phone numbers: 555-201-9988, 555.201.9988, 5552019988, etc.
PHONE_PATTERN = r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b"
# Same shape as the email detector on sanitize_intake. Local, so the refund
# fields do not depend on the model copying them out of the summary.
EMAIL_PATTERN = r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"
# Dollar amounts only. A bare number would also match pieces of a card number.
AMOUNT_PATTERN = r"\$\d+(?:,\d{3})*(?:\.\d{2})?"


def detect_customer_email(text: str) -> str:
    """Return the email addresses in `text`, in order, without duplicates.

    One address is returned as itself. Several are joined with ", " so a
    second address is not dropped. No address returns an empty string.
    """
    found = list(dict.fromkeys(re.findall(EMAIL_PATTERN, text)))
    if not found:
        return ""
    if len(found) == 1:
        return found[0]
    return ", ".join(found)


def detect_claimed_amount(text: str) -> str:
    """Return the refund amount the message is asking for.

    One dollar amount is returned as written. When the message says
    "difference" and quotes at least two prices, the amount is the gap
    between the last two. Otherwise every detected amount is listed, and
    the model is not handed a single guess. "Twice" does not multiply the
    price: a duplicate charge of $49.99 is a $49.99 refund.
    """
    found = re.findall(AMOUNT_PATTERN, text)
    if not found:
        return ""
    if len(found) == 1:
        return found[0]
    if re.search(r"\bdifference\b", text, re.IGNORECASE):
        values = [
            Decimal(amount.replace("$", "").replace(",", "")) for amount in found
        ]
        gap = abs(values[-1] - values[-2])
        return f"${gap:.2f}"
    return ", ".join(found)


sanitize_intake = create_agent(
    model,
    tools=[],
    system_prompt=(
        "Summarize this support message in 2-3 sentences: what the "
        "customer needs, and how urgent it sounds. Do not repeat any "
        "contact details back in your summary."
    ),
    middleware=[
        PIIMiddleware("email", strategy="redact", apply_to_input=True),
        PIIMiddleware("credit_card", strategy="mask", apply_to_input=True),
        PIIMiddleware(
            "phone",
            detector=PHONE_PATTERN,
            strategy="redact",
            apply_to_input=True,
        ),
    ],
)


@tool
def issue_refund(customer_email: str, amount: str, reason: str) -> str:
    """Issue a refund to a customer. This actually issues it -- treat every call as real."""
    return f"Refund of {amount} issued to {customer_email} ({reason})."


# No checkpointer on this inner agent -- only the outer graph is
# checkpointed, so it's the one that catches the interrupt. See
# middleware_example_hitl.py's docstring for why that matters.
draft_refund_action = create_agent(
    model,
    tools=[issue_refund],
    system_prompt=(
        "You help resolve billing complaints. The user message gives you a "
        "redacted summary, the customer email, and the claimed amount. "
        "Those were extracted for you. If a refund is clearly warranted, "
        "call issue_refund with that email, that claimed amount, and a "
        "one-line reason. Do not invent an email or an amount, and do not "
        "ask for the original message. If a refund isn't clearly warranted, "
        "explain why instead of calling the tool."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
            },
            description_prefix="This refund is about to be issued for real:",
        ),
    ],
)


class SupportState(TypedDict):
    raw_message: str
    redacted_summary: str
    category: str
    refund_outcome: str
    email: str
    claimed_amount: str


def sanitize_intake_node(state: SupportState) -> dict:
    result = sanitize_intake.invoke(
        {"messages": [{"role": "user", "content": state["raw_message"]}]}
    )
    return {"redacted_summary": result["messages"][-1].text}


def triage_node(state: SupportState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "Classify this support summary into exactly one category: "
            "BILLING, TECHNICAL, or ACCOUNT. Reply with just the word."
        )},
        {"role": "user", "content": state["redacted_summary"]},
    ])
    return {"category": response.text.strip().upper()}


def extract_refund_fields_node(state: SupportState) -> dict:
    """Fill email and claimed_amount from the raw message before the refund agent runs."""
    return {
        "email": detect_customer_email(state["raw_message"]),
        "claimed_amount": detect_claimed_amount(state["raw_message"]),
    }


def refund_agent_input(state: SupportState) -> str:
    """What draft_refund_action sees. The raw message is not included."""
    email = state["email"] or "(none detected)"
    amount = state["claimed_amount"] or "(none detected)"
    return (
        f"Redacted summary: {state['redacted_summary']}\n"
        f"Customer email: {email}\n"
        f"Claimed amount: {amount}"
    )


def draft_refund_action_node(state: SupportState) -> dict:
    result = draft_refund_action.invoke(
        {"messages": [{"role": "user", "content": refund_agent_input(state)}]}
    )
    return {"refund_outcome": result["messages"][-1].text}


def route_after_triage(state: SupportState) -> str:
    # Only a billing ticket has any business reaching an agent whose
    # entire system prompt is about resolving billing complaints.
    if state["category"] == "BILLING":
        return "extract_refund_fields"
    return END


graph = StateGraph(SupportState)
graph.add_node("sanitize_intake", sanitize_intake_node)
graph.add_node("triage", triage_node)
graph.add_node("extract_refund_fields", extract_refund_fields_node)
graph.add_node("draft_refund_action", draft_refund_action_node)
graph.add_edge(START, "sanitize_intake")
graph.add_edge("sanitize_intake", "triage")
graph.add_conditional_edges(
    "triage", route_after_triage, ["extract_refund_fields", END]
)
graph.add_edge("extract_refund_fields", "draft_refund_action")
graph.add_edge("draft_refund_action", END)
support_graph = graph.compile(checkpointer=InMemorySaver())
#support_graph.get_graph().draw_mermaid_png(output_file_path="support_workflow.png")


def run_with_approval(initial_state: dict, thread_id: str) -> dict:
    """
    Invoke, check for a pause, collect a human decision, resume, 
    repeat until the graph actually finishes.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = support_graph.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        interrupt = result["__interrupt__"][0]
        actions = interrupt.value.get("action_requests", interrupt.value)
        print("\n--- Needs human approval ---")
        print(actions)
        decision_type = input("approve / edit / reject? ").strip().lower()

        if decision_type == "approve":
            decision = {"type": "approve"}
        elif decision_type == "reject":
            reason = input("Reason: ")
            decision = {"type": "reject", "message": reason}
        else:
            new_amount = input("Corrected amount: ")
            decision = {
                "type": "edit",
                "edited_action": {
                    "name": "issue_refund",
                    "args": {"amount": new_amount},
                },
            }

        result = support_graph.invoke(Command(resume={"decisions": [decision]}), config=config)

    return result


SAMPLE_MESSAGES = {
    "billing_refund_warranted": (
        "Hi, this is Dana Price. My card ending in 4319 2100 2222 3344 was "
        "charged twice this month -- $49.99 twice on the same day. Call me "
        "back at 555-201-9988 or email dana.price@example.com if you need "
        "anything else -- this is urgent, I need a refund before Friday."
    ),
    "billing_refund_not_warranted": (
        "This is Marcus Yi. I just noticed my plan went from $19.99 to "
        "$24.99 this month and I want a refund for the difference. I know "
        "you sent an email about the price change back in September but I "
        "don't think that's fair. You can reach me at "
        "marcus.yi88@example.com."
    ),
    "technical_password_reset": (
        "Hey, I'm locked out of my account -- the reset link you emailed "
        "isn't working and I've tried three times. My number is "
        "555-884-2201 if someone can just call me back today."
    ),
    "account_update_request": (
        "I need to update the mailing address on file for my account and "
        "add my spouse as an authorized user. The new address isn't "
        "saving in the portal. You can email me at r.chen.acct@example.com "
        "to confirm once it's done."
    ),
    "misplaced_off_topic": (
        "Hi there! I saw your company is hiring for a customer success "
        "role on LinkedIn -- is this still open, and who should I send my "
        "resume to? Thanks!"
    ),
}

if __name__ == "__main__":
    i = 1
    for label, raw_message in SAMPLE_MESSAGES.items():
        print(f"\n{'=' * 70}\nRunning message {i}: {label}\n{'=' * 70}")
        final = run_with_approval(
            {
                "raw_message": raw_message,
                "redacted_summary": "",
                "category": "",
                "refund_outcome": "",
                "email": "",
                "claimed_amount": "",
            },
            thread_id=label,
        )
        print("\n--- Redacted summary (this is all `triage` ever saw) ---")
        print(final["redacted_summary"])
        print("\n--- Category ---")
        print(final["category"])
        print("\n--- Email / claimed amount (filled only on the billing path) ---")
        print(final["email"] or "(none detected)")
        print(final["claimed_amount"] or "(none detected)")
        print("\n--- Refund outcome ---")
        print(
            final["refund_outcome"]
            or "(skipped -- not a BILLING ticket, or no refund action taken)"
        )
        i += 1