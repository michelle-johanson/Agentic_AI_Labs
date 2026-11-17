# Improvement 1: Stop handing the refund agent the raw message

## 1. What we extended

`SupportState` gained `email` and `claimed_amount`.

`detect_customer_email()` and `detect_claimed_amount()` read those values out of `raw_message`. `extract_refund_fields_node()` runs them. `route_after_triage()` now sends a `BILLING` ticket to that node instead of straight to `draft_refund_action_node()`. The refund node calls `refund_agent_input()` and passes that string to the agent.

`draft_refund_action`'s system prompt no longer tells the model to read the customer's raw message. It tells the model to use the email and claimed amount it was given.

## 2. Why this and not the alternatives

`sanitize_intake` already redacts email, masks the card, and redacts the phone, and its prompt says not to repeat contact details. `triage` only sees `redacted_summary`. `draft_refund_action_node()` did not. It sent `state["raw_message"]` because `issue_refund` needs a real email and a specific amount, and the summary is told to drop contact details.

On `billing_refund_warranted` that raw message contains the card `4319 2100 2222 3344`, the phone `555-201-9988`, and `dana.price@example.com`. The refund model is the one about to call the tool. It does not need the card or the phone.

Rejected:

- **Pass only `redacted_summary`.** The tool call would have no email and no amount unless the summary model disobeyed its own prompt and copied them.
- **Keep passing `raw_message` and add card/phone middleware on the refund agent.** The model would still see the whole complaint, and middleware would also be in a position to redact the email the tool needs.
- **Let the model extract the email and amount from the raw message.** That is the old path. The card and phone ride along.
- **Hide the email inside a Python closure and take it off the tool schema.** The model then cannot put the email in the call, which is safer, but this extension was specifically to hand the agent the two extracted fields plus the summary.
- **Always take the first dollar amount.** On `billing_refund_not_warranted` that is `$19.99`, the old plan price. The message asks for the difference, which is `$5.00`.

## 3. Benefits

No Ollama calls. The before column is the string `draft_refund_action_node()` used to send, which was `raw_message`. The after column is `refund_agent_input()` with the extracted fields and a stand-in summary that itself contains no card, phone, or email: "Customer reports a billing problem and wants it resolved before Friday." Card and phone counts use the same detectors as intake (`detect_credit_card`, `PHONE_PATTERN`). Only the two billing messages are routed to this node.

| Message | Cards before | Cards after | Phones before | Phones after | Emails before | Emails after | Extracted amount |
|---|---|---|---|---|---|---|---|
| billing_refund_warranted | 1 | 0 | 1 | 0 | 1 | 1 | $49.99 |
| billing_refund_not_warranted | 0 | 0 | 0 | 0 | 1 | 1 | $5.00 |
| **Total** | **1** | **0** | **1** | **0** | **2** | **2** | |

The email count stays at 1 on purpose. `issue_refund` still needs it, so `refund_agent_input()` puts `dana.price@example.com` and `marcus.yi88@example.com` in the prompt. The card number and both sample phone numbers are absent, and the raw message is not a substring of the new prompt.

Detector output on every sample, including messages that never reach the refund node:

| Message | Email | Claimed amount |
|---|---|---|
| billing_refund_warranted | dana.price@example.com | $49.99 |
| billing_refund_not_warranted | marcus.yi88@example.com | $5.00 |
| technical_password_reset | (none) | (none) |
| account_update_request | r.chen.acct@example.com | (none) |
| misplaced_off_topic | (none) | (none) |

`technical_password_reset` contains `555-884-2201`. The graph does not route it to the refund node. Building the refund prompt for it anyway still leaves the phone out.

There is no earlier Lab 6 eval to re-run. `run_with_approval()` was not changed.

## 4. Drawbacks and tradeoffs

- **The summary is copied through.** `refund_agent_input()` does not scan `redacted_summary`. Feeding it a summary that still says `4319 2100 2222 3344` puts that card in the refund prompt (1 card detected). Intake masks a card to the last four digits before the summary model sees it, but whatever that model writes is what the refund model reads. This check used a clean stand-in summary, not a live `qwen3` summary.
- **The amount rule is tuned to these messages.** One `$` amount is kept as written, so "$49.99 twice" stays `$49.99` and is not doubled. The word "difference" with two or more prices uses the gap between the last two only. `$10.00`, `$12.00`, `$15.00` and the word "difference" comes out `$3.00`, not `$5.00`. Two prices and no "difference" are listed rather than guessed: `$10.00, $3.00`. The word "different" does not trigger the gap.
- **A price with no dollar sign is missed.** `refund 49.99 to dana.price@example.com` extracts the email and an empty amount. Matching bare numbers would also match pieces of the card.
- **Several emails are joined, not chosen.** `a@ex.com` and `b@ex.com` become `a@ex.com, b@ex.com`. The tool wants one address. The model still has to pick.
- **An empty detection becomes the text `(none detected)`.** The prompt tells the model not to invent an email or an amount. Nothing in code stops it from calling `issue_refund` with something else anyway.
- **`$1,250.00` is kept with the comma.** The tool argument is that string, not `1250.00`.
