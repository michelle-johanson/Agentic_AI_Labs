"""
Lab 4 Extension — Script 2: Structured output with a Pydantic schema

Same pipeline as script 1 — same PDFs, same chunking, same keyword
prefilter — but the model call itself changes: instead of trusting the
prompt alone to produce correct JSON, we bind a Pydantic schema
(ClimateDisclosure, in docAnalyzerCommon.py) with .with_structured_output(),
and wrap it in a small bounded retry loop. Every finding also needs an exact
supporting quotation, checked against the source chunk before it is accepted.

Compare this script's failure counts against script 1's parse/shape
failure counts
"""
import json

from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate

from docAnalyzerCommon import (
    ClimateDisclosure, EVIDENCE_INSTRUCTIONS, EXTRACTION_INSTRUCTIONS, LAB_DIR, PDF_DIR,
    chunk_text, extract_text_from_pdf, keyword_hit, list_pdfs, validate_evidence,
)

MODEL_NAME = "qwen2.5:7b"
MAX_RETRIES = 3
OUTPUT_FILE = LAB_DIR / "files" / "script2_structured_results.json"

llm = ChatOllama(model=MODEL_NAME, temperature=0.3)
structured_llm = llm.with_structured_output(ClimateDisclosure)

PROMPT = PromptTemplate.from_template(
    EXTRACTION_INSTRUCTIONS + EVIDENCE_INSTRUCTIONS + "\n{retry_feedback}\n"
)
chain = PROMPT | structured_llm


def extract_with_retry(chunk: str, max_retries: int = MAX_RETRIES):
    """Returns (disclosure_or_None, attempts_used, last_error_or_None).
    A bounded retry, not an unbounded one — see today's retry-bound slide
    for why that cap matters."""
    if max_retries < 1:
        raise ValueError("max_retries must be at least 1")
    last_error = None
    for attempt in range(1, max_retries + 1):
        try:
            disclosure = chain.invoke({
                "text": chunk,
                "retry_feedback": (
                    "The previous response failed validation. Return the required objects "
                    "and use only exact quotes from the source text; omit unsupported findings."
                    if last_error else ""
                ),
            })
            return validate_evidence(disclosure, chunk), attempt, None
        except Exception as e:  # includes schema, provider, and evidence-validation errors
            last_error = str(e)
    return None, max_retries, last_error


def run():
    results = []
    total_sent = 0
    hard_failures = 0
    needed_a_retry = 0

    for file in list_pdfs(PDF_DIR):
        print(f"Extracting from: {file}")
        text = extract_text_from_pdf(file)
        for i, chunk in enumerate(chunk_text(text)):
            if not keyword_hit(chunk):
                continue
            total_sent += 1
            disclosure, attempts, error = extract_with_retry(chunk)

            if disclosure is None:
                hard_failures += 1
                results.append({
                    "file": file, "chunk_index": i, "status": "failed",
                    "attempts": attempts, "error": error,
                })
                continue

            if attempts > 1:
                needed_a_retry += 1

            results.append({
                "file": file, "chunk_index": i, "status": "ok", "attempts": attempts,
                "extraction": disclosure.model_dump(), "is_empty": disclosure.is_empty(),
            })

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))

    validated_first_try = total_sent - hard_failures - needed_a_retry
    print(f"\nChunks sent to the LLM: {total_sent}")
    print(f"Validated on the first try: {validated_first_try}")
    print(f"Needed a retry but eventually validated: {needed_a_retry}")
    print(f"Still invalid after {MAX_RETRIES} attempts: {hard_failures} "
          f"({hard_failures / max(total_sent, 1):.1%})")
    print(f"Wrote {len(results)} chunk-level records to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()
