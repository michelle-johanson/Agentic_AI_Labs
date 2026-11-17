# ===== NEW (Hardening 1-3: standard-library imports) =====
import json
import re
from datetime import datetime
from pathlib import Path
# ===== END NEW =====

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


# ===== NEW (Option 13: approval-gated account credit) =====
@tool
def apply_account_credit(customer_email: str, amount: str, reason: str) -> str:
    """Apply a one-time goodwill credit to a customer's account. This actually applies it -- treat every call as real."""
    return f"Account credit of {amount} applied for {customer_email} ({reason})."
# ===== END NEW =====


# No checkpointer on this inner agent -- only the outer graph is
# checkpointed, so it's the one that catches the interrupt. See
# middleware_example_hitl.py's docstring for why that matters.
# ===== CHANGED (Option 13: second tool, new instructions, per-tool approval rules) =====
draft_refund_action = create_agent(
    model,
    tools=[issue_refund, apply_account_credit],
    system_prompt=(
        "You help resolve billing complaints. Read the customer's raw "
        "message and choose at most one action:\n"
        "- If the company clearly made a billing error (for example, a "
        "duplicate charge), call issue_refund.\n"
        "- If there was no billing error but the complaint is reasonable "
        "and a small goodwill gesture would help (for example, a price "
        "change the customer was notified about), call "
        "apply_account_credit.\n"
        "- Otherwise, explain why no action is warranted instead of "
        "calling a tool.\n"
        "Whichever tool you call, use the customer's actual email, a "
        "specific amount, and a one-line reason."
    ),
    middleware=[
        HumanInTheLoopMiddleware(
            interrupt_on={
                "issue_refund": {"allowed_decisions": ["approve", "edit", "reject"]},
                "apply_account_credit": {"allowed_decisions": ["approve", "reject"]},
            },
            description_prefix="This billing action is about to happen for real:",
        ),
    ],
)
# ===== END CHANGED =====


class SupportState(TypedDict):
    raw_message: str
    redacted_summary: str
    category: str
    refund_outcome: str


def sanitize_intake_node(state: SupportState) -> dict:
    result = sanitize_intake.invoke(
        {"messages": [{"role": "user", "content": state["raw_message"]}]}
    )
    return {"redacted_summary": result["messages"][-1].text}


# ===== NEW (Hardening 2: robust triage parsing) =====
ALLOWED_CATEGORIES = ["BILLING", "TECHNICAL", "ACCOUNT"]
THINK_BLOCK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)


def _parse_category(reply: str) -> str:
    """
    Picks the earliest allowed category word in the reply, so "BILLING.",
    "Category: BILLING", or a qwen3 <think> block can't misroute a ticket.
    Falls back to UNKNOWN rather than passing arbitrary text to the router.
    """
    text = THINK_BLOCK.sub("", reply).upper()
    matches = [
        (match.start(), category)
        for category in ALLOWED_CATEGORIES
        if (match := re.search(rf"\b{category}\b", text))
    ]
    if matches:
        return min(matches)[1]
    print(f"[triage] WARNING: no allowed category in reply {text.strip()!r} -- using UNKNOWN")
    return "UNKNOWN"
# ===== END NEW =====


def triage_node(state: SupportState) -> dict:
    response = model.invoke([
        {"role": "system", "content": (
            "Classify this support summary into exactly one category: "
            "BILLING, TECHNICAL, or ACCOUNT. Reply with just the word."
        )},
        {"role": "user", "content": state["redacted_summary"]},
    ])
    # ===== CHANGED (Hardening 2) =====
    return {"category": _parse_category(response.text)}
    # ===== END CHANGED =====


def draft_refund_action_node(state: SupportState) -> dict:
    # Deliberately reads raw_message, not redacted_summary -- see docstring.
    result = draft_refund_action.invoke(
        {"messages": [{"role": "user", "content": state["raw_message"]}]}
    )
    return {"refund_outcome": result["messages"][-1].text}


def route_after_triage(state: SupportState) -> str:
    # Only a billing ticket has any business reaching an agent whose
    # entire system prompt is about resolving billing complaints.
    if state["category"] == "BILLING":
        return "draft_refund_action"
    return END


graph = StateGraph(SupportState)
graph.add_node("sanitize_intake", sanitize_intake_node)
graph.add_node("triage", triage_node)
graph.add_node("draft_refund_action", draft_refund_action_node)
graph.add_edge(START, "sanitize_intake")
graph.add_edge("sanitize_intake", "triage")
graph.add_conditional_edges(
    "triage", route_after_triage, ["draft_refund_action", END]
)
graph.add_edge("draft_refund_action", END)
support_graph = graph.compile(checkpointer=InMemorySaver())
#support_graph.get_graph().draw_mermaid_png(output_file_path="support_workflow.png")


# ===== NEW (Hardening 1: reliable approval prompt) =====
AMOUNT_PATTERN = re.compile(r"^\$?\d+(\.\d{1,2})?$")


def _ask_choice(allowed: list[str]) -> str:
    """Re-prompts until the answer is one of the decisions this action allows."""
    prompt = " / ".join(allowed) + "? "
    while True:
        choice = input(prompt).strip().lower()
        if choice in allowed:
            return choice
        print(f"Please type one of: {', '.join(allowed)}")


def _ask_amount() -> str:
    while True:
        raw = input("Corrected amount (e.g. 49.99): ").strip()
        if AMOUNT_PATTERN.match(raw):
            return f"${float(raw.lstrip('$')):.2f}"
        print("Enter a dollar amount like 49.99 or $49.99.")


def _review_action(action: dict, allowed: list[str]) -> dict:
    """
    Builds one decision for one paused tool call. An edit replaces the tool
    call's args wholesale, so the original args are copied and only the
    amount is changed -- otherwise customer_email and reason would be lost.
    """
    print(f"\n--- Needs human approval: {action['name']} ---")
    if action.get("description"):
        print(action["description"])
    print(f"Arguments: {action['args']}")

    choice = _ask_choice(allowed)
    if choice == "approve":
        return {"type": "approve"}
    if choice == "reject":
        return {"type": "reject", "message": input("Reason: ").strip()}
    edited_args = {**action["args"], "amount": _ask_amount()}
    return {"type": "edit", "edited_action": {"name": action["name"], "args": edited_args}}
# ===== END NEW =====


# ===== NEW (Hardening 3: audit log entry) =====
AUDIT_LOG_FILE = Path(__file__).parent / "refund_audit_log.json"


def _audit_entry(thread_id: str, action: dict, decision: dict) -> dict:
    if decision["type"] == "approve":
        final_args = action["args"]
    elif decision["type"] == "edit":
        final_args = decision["edited_action"]["args"]
    else:
        final_args = None
    return {
        "thread_id": thread_id,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "tool": action["name"],
        "original_args": action["args"],
        "final_args": final_args,
        "decision": decision["type"],
        "reject_reason": decision.get("message"),
    }
# ===== END NEW =====


# ===== CHANGED (Hardening 1 + 3: one decision per action, each one audited) =====
def run_with_approval(initial_state: dict, thread_id: str, audit_log: list | None = None) -> dict:
    """
    Invoke, check for a pause, collect a human decision, resume, 
    repeat until the graph actually finishes.

    The middleware requires exactly one decision per paused tool call, in
    the same order as action_requests, so every action gets its own review.
    """
    config = {"configurable": {"thread_id": thread_id}}
    result = support_graph.invoke(initial_state, config=config)

    while "__interrupt__" in result:
        request = result["__interrupt__"][0].value
        decisions = []
        for action, review in zip(request["action_requests"], request["review_configs"]):
            decision = _review_action(action, review["allowed_decisions"])
            decisions.append(decision)
            if audit_log is not None:
                audit_log.append(_audit_entry(thread_id, action, decision))

        result = support_graph.invoke(Command(resume={"decisions": decisions}), config=config)

    return result
# ===== END CHANGED =====


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
    # ===== NEW (Hardening 3) =====
    audit_log: list[dict] = []
    # ===== END NEW =====
    i = 1
    for label, raw_message in SAMPLE_MESSAGES.items():
        print(f"\n{'=' * 70}\nRunning message {i}: {label}\n{'=' * 70}")
        final = run_with_approval(
            {
                "raw_message": raw_message,
                "redacted_summary": "",
                "category": "",
                "refund_outcome": "",
            },
            thread_id=label,
            audit_log=audit_log,  # NEW (Hardening 3)
        )
        print("\n--- Redacted summary (this is all `triage` ever saw) ---")
        print(final["redacted_summary"])
        print("\n--- Category ---")
        print(final["category"])
        # ===== CHANGED (Option 13: covers refunds and account credits) =====
        print("\n--- Billing action outcome ---")
        print(
            final["refund_outcome"]
            or "(skipped -- not a BILLING ticket, or no billing action taken)"
        )
        # ===== END CHANGED =====
        i += 1

    # ===== NEW (Hardening 3: write the audit log) =====
    AUDIT_LOG_FILE.write_text(json.dumps(audit_log, indent=2))
    print(f"\nWrote {len(audit_log)} decision(s) to {AUDIT_LOG_FILE}")
    # ===== END NEW =====