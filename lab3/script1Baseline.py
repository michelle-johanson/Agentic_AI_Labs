"""
Script 1: Baseline (prompt engineering only)

This is the starting point for the whole extension sequence: the *only*
defense against bad output is the wording of the prompt itself. We ask the
model, in plain English, to return a JSON object with six specific keys —
and then we just measure how often that actually happens, instead of
assuming it did.

Instead of writing the raw model output straight to a file, we try to json.loads() it and
check its shape, and tally the results. That tally is the number this
whole extension sequence is trying to improve on.
"""
import json
from pathlib import Path

from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate

from docAnalyzerCommon import (
    EXPECTED_KEYS, EXTRACTION_INSTRUCTIONS, PDF_DIR,
    chunk_text, extract_text_from_pdf, keyword_hit, list_pdfs,
)

MODEL_NAME = "qwen2.5:3b"
OUTPUT_FILE = Path("lab3/files/script1_baseline_results.json")

llm = OllamaLLM(model=MODEL_NAME, temperature=0.3)

PROMPT = PromptTemplate.from_template(
    EXTRACTION_INSTRUCTIONS
    + "\nOutput ONLY a single valid JSON object with exactly these six keys, each a "
      "list of strings: climate_risks, emissions, targets_and_goals, governance, "
      "financials, mitigation. Do not include markdown code fences, comments, or any "
      "text other than the JSON object itself.\n"
)

chain = PROMPT | llm


def naive_shape_ok(obj) -> bool:
    """Checks the parsed JSON has exactly the six expected keys, each a list.
    No type-checking of the list *contents* — that's intentionally cruder
    than what Pydantic will do in script 2."""
    return isinstance(obj, dict) and all(
        key in obj and isinstance(obj[key], list) for key in EXPECTED_KEYS
    )


def run():
    results = []
    total_sent = 0
    parse_failures = 0
    shape_failures = 0

    for file in list_pdfs(PDF_DIR):
        print(f"Extracting from: {file}")
        text = extract_text_from_pdf(file)
        for i, chunk in enumerate(chunk_text(text)):
            if not keyword_hit(chunk):
                continue
            total_sent += 1
            raw = chain.invoke({"text": chunk})
            print(f"Chunk {i} sent to LLM for assessment")
            record = {"file": file, "chunk_index": i, "raw_output": raw, "chunk_text": chunk}

            try:
                parsed = json.loads(raw)
                record["parsed_ok"] = True
                record["shape_ok"] = naive_shape_ok(parsed)
                if not record["shape_ok"]:
                    shape_failures += 1
            except json.JSONDecodeError:
                record["parsed_ok"] = False
                record["shape_ok"] = False
                parse_failures += 1

            results.append(record)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps(results, indent=2))

    print(f"\nChunks sent to the LLM: {total_sent}")
    print(f"Failed to parse as JSON at all: {parse_failures} "
          f"({parse_failures / max(total_sent, 1):.1%})")
    print(f"Parsed but wrong shape (missing/extra keys, non-list values): "
          f"{shape_failures} ({shape_failures / max(total_sent, 1):.1%})")
    print(f"Wrote {len(results)} chunk-level records to {OUTPUT_FILE}")


if __name__ == "__main__":
    run()