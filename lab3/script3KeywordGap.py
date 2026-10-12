"""
Lab 4 Extension — Script 3: What does keyword-only filtering miss?

Script 1 and 2 both only ever looked at chunks the keyword list flagged.
This script removes that prefilter and runs script 2's validated extractor
over EVERY chunk, then compares that against what the keyword filter would
have kept. That turns the keyword filter into something we can score like
a classifier, against the LLM's own judgment as a (imperfect, but
independent) reference:

  - keyword_hit=False, llm_found_something=True
      -> the keyword filter's recall gap: real disclosure content it
         would have silently dropped.
  - keyword_hit=True,  llm_found_something=False
      -> wasted LLM calls (or: cases script 4 re-examines, since an empty
         result here might be the LLM's miss, not the keyword's false
         positive — that's exactly what script 4 checks).

This is more expensive than scripts 1-2 (every chunk gets an LLM call, not
just the keyword-flagged ones) — that cost is the price of having a
ground truth to measure the shortcut against.
"""
import json

from docAnalyzerCommon import LAB_DIR, PDF_DIR, chunk_text, extract_text_from_pdf, keyword_hit, list_pdfs
from script2PydanticStructured import extract_with_retry

OUTPUT_FILE = LAB_DIR / "files" / "script3_gap_report.json"


def run():
    rows = []

    for file in list_pdfs(PDF_DIR):
        print(f"Extracting from: {file}")
        text = extract_text_from_pdf(file)
        for i, chunk in enumerate(chunk_text(text)):
            print(f"Processing chunk {i}")
            disclosure, attempts, error = extract_with_retry(chunk)
            llm_found_something = (disclosure is not None) and not disclosure.is_empty()
            if llm_found_something:
                print(f"LLM found a disclosure, keyword hit = {keyword_hit(chunk)}")
            rows.append({
                "file": file,
                "chunk_index": i,
                "status": "failed" if disclosure is None else "ok",
                "attempts": attempts,
                "error": error,
                "keyword_hit": keyword_hit(chunk),
                "llm_found_something": llm_found_something,
                "item_count": disclosure.total_items() if disclosure else 0,
                "extraction": disclosure.model_dump() if disclosure else None,
                "chunk_text": chunk,
                "chunk_preview": chunk[:200],
            })

    missed_by_keywords = [r for r in rows if not r["keyword_hit"] and r["llm_found_something"]]
    # A failed validation is unknown, not a confirmed empty extraction.
    wasted_by_keywords = [
        r for r in rows
        if r["status"] == "ok" and r["keyword_hit"] and not r["llm_found_something"]
    ]
    failed = [r for r in rows if r["status"] == "failed"]
    kept_count = sum(r["keyword_hit"] for r in rows)

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.write_text(json.dumps({
        "all_rows": rows,
        "missed_by_keywords": missed_by_keywords,
        "wasted_by_keywords": wasted_by_keywords,
        "failed": failed,
    }, indent=2))

    total = len(rows)
    print(f"\nTotal chunks evaluated: {total}")
    print(f"Failed extraction/validation (not counted as empty): {len(failed)}")
    print(f"Keyword filter would have kept: {kept_count} ({kept_count / max(total, 1):.1%})")
    print(f"Dropped by the keyword filter but the LLM still found real content: "
          f"{len(missed_by_keywords)}  <-- the keyword filter's recall gap")
    print(f"Kept by the keyword filter but the LLM found nothing: "
          f"{len(wasted_by_keywords)}  <-- script 4 checks whether these are true "
          f"negatives or LLM misses")
    print(f"Full report written to {OUTPUT_FILE} — see missed_by_keywords for concrete "
          f"examples worth discussing in class.")


if __name__ == "__main__":
    run()
