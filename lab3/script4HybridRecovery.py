"""
Lab 4 Extension — Script 4: Is keyword matching still worth keeping?

Script 3 produced a "wasted_by_keywords" set: chunks the keyword filter
flagged as relevant where script 2's structured extraction came back
empty. That emptiness is ambiguous — it could mean the keyword was a true
false positive (e.g. "green" inside "evergreen contract"), or it could
mean the primary LLM pass under-extracted on a chunk that actually did
have something in it.

This script re-examines exactly that set with a second, more pointed
pass — telling the model which keyword(s) triggered the flag and asking
it to look again — and tallies how often that recovers real content
versus confirms the chunk was genuinely empty. That recovery rate is the
answer to "does keeping the keyword list around still earn its keep, now
that we have a validated LLM extractor doing the real work?" A high
recovery rate says yes (hybrid: keyword matching as a safety net on top
of the LLM pass); a low one says the keyword list is mostly noise at this
point and could be retired or shrunk.

Run script3_keyword_gap.py first — this script reads its output.
"""
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from docAnalyzerCommon import (
    ClimateDisclosure, EVIDENCE_INSTRUCTIONS, LAB_DIR, matched_keywords, validate_evidence,
)

MODEL_NAME = "qwen2.5:7b"
GAP_REPORT_FILE = LAB_DIR / "files" / "script3_gap_report.json"
OUTPUT_FILE = LAB_DIR / "files" / "script4_recovery_report.json"

llm = ChatOllama(model=MODEL_NAME, temperature=0.0)
structured_llm = llm.with_structured_output(ClimateDisclosure)

RECHECK_PROMPT = PromptTemplate.from_template("""
A keyword-based filter flagged the text below as possibly containing climate-related
disclosures (it matched on: {matched_keywords}), but a first extraction pass found
nothing. Re-read it carefully for anything relevant, including indirect or partial
statements. If there is genuinely nothing relevant here, leave every category empty —
do not invent content just to justify the flag.

Text:
{text}
""" + EVIDENCE_INSTRUCTIONS)

recheck_chain = RECHECK_PROMPT | structured_llm


def run():
    if not GAP_REPORT_FILE.exists():
        raise SystemExit(f"{GAP_REPORT_FILE} not found — run script3_keyword_gap.py first.")

    gap_report = json.loads(GAP_REPORT_FILE.read_text())
    candidates = gap_report["wasted_by_keywords"]

    recovered, confirmed_empty, failed = [], [], []

    for row in candidates:
        chunk = row["chunk_text"]
        matched = matched_keywords(chunk)
        try:
            disclosure = recheck_chain.invoke({
                "text": chunk,
                "matched_keywords": ", ".join(matched),
            })
            validate_evidence(disclosure, chunk)
        except Exception as e:
            failed.append({**row, "error": str(e)})
            continue

        if disclosure.is_empty():
            confirmed_empty.append(row)
        else:
            recovered.append({**row, "recovered_extraction": disclosure.model_dump()})

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps({
        "recovered": recovered,
        "confirmed_empty": confirmed_empty,
        "failed": failed,
    }, indent=2))

    total = len(candidates)
    print(f"Keyword-flagged / first-pass-empty chunks re-checked: {total}")
    print(f"Recovered real content on the second pass: {len(recovered)} "
          f"({len(recovered) / max(total, 1):.1%})  <-- keyword matching earned its "
          f"keep here")
    print(f"Confirmed genuinely empty: {len(confirmed_empty)} "
          f"({len(confirmed_empty) / max(total, 1):.1%})  <-- keyword matching was "
          f"just noise here")
    if failed:
        print(f"Failed to re-check (schema/model errors): {len(failed)}")
    print(f"Full report written to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
