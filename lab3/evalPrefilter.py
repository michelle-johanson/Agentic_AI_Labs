"""Compare the original keyword prefilter against the current one on the real 10-K chunks.

Run from the repository root: python lab3/evalPrefilter.py
No Ollama calls are made. Two measurements:
  1. Gold passages: hand-picked passages that meet the SEC climate rule's categories.
     A passage counts as caught if any chunk containing it passes the prefilter.
  2. LLM reference (only if files/script3_gap_report.json exists): of the chunks where
     script 3's extractor found something, how many each prefilter would have kept.
"""
import json
import re
from collections import Counter

from docAnalyzerCommon import (
    LAB_DIR, PDF_DIR, chunk_text, extract_text_from_pdf, keyword_hit, list_pdfs,
    matched_keywords,
)

# Reproduce the original substring-match prefilter for the comparison.
BASELINE_KEYWORDS = [
    "climate", "emission", "greenhouse", "ghg", "scope 1", "scope 2", "scope 3",
    "weather", "natural disaster", "act of god", "acts of god", "tornado",
    "hurricane", "earthquake", "wildfire", "drought", "flood", "temperature",
    "sea level", "sea-level", "carbon", "warming", "net zero", "net-zero",
    "methane", "precipitation", "co2", "clean", "green", "earth",
]


def baseline_keyword_hit(chunk: str) -> bool:
    return any(kw.casefold() in chunk.casefold() for kw in BASELINE_KEYWORDS)


# Anchors are short verbatim excerpts. Matching ignores all whitespace, because PDF
# extraction inserts stray spaces and line breaks inside words and table rows.
GOLD_PASSAGES = {
    "costco10K.pdf": [
        ("transition risk", "concerns and regulation about climate change"),
        ("physical risk", "Natural disasters and extreme weather conditions, including those impacted by climate change"),
        ("transition risk", "Government regulations limiting carbon dioxide and other greenhouse gas emissions"),
        ("transition risk", "More stringent fuel economy standards"),
        ("targets", "Costco has set public targets and disclosed intentions for future action regarding sustainability"),
        ("targets", "the ability to procure sufficient clean energy at competitive market rates"),
        ("physical risk", "financial liquidity, climate change, natural disasters, extreme weather conditions"),
        ("financials", "additional heating, cooling, and refrigeration costs"),
    ],
    "eliLilly10K.pdf": [
        ("physical risk", "events that may be due to climate change"),
        ("transition risk", "regulating greenhouse gas emissions, carbon taxes, emissions trading schemes"),
        ("transition risk", "stakeholders on climate, social, and other sustainability matters"),
    ],
    "tesla10K.pdf": [
        ("financials", "as well as sales of automotive regulatory credits"),
        ("financials", "We earn tradable credits in the operation of our business under various regulations related to zero-emission vehicles"),
        ("financials", "Automotive regulatory credits 2,763 1,790 1,776"),
        ("financials", "Automotive regulatory credits revenue increased $973 million"),
        ("financials", "Our revenue from automotive regulatory credits is directly related to our new vehicle production"),
        ("physical risk", "the long-term effects of climate change"),
        ("physical risk", "located in a flood-prone area"),
        ("transition risk", "New, or changes in, environmental and climate change laws"),
        ("mitigation", "We believe the world cannot reduce carbon emissions"),
        ("financials", "The IRA includes multiple incentives to promote clean energy"),
        ("financials", "the impact from our IRA incentive was primarily a reduction of our costs of revenue"),
        ("transition risk", "facing increasing scrutiny related to their environmental, social and governance"),
        ("transition risk", "other renewable energy sources could make our products less attractive"),
        ("transition risk", "increasingly conducive to the adoption of renewable energy systems"),
        ("transition risk", "incentives that support the development and adoption of electric vehicles in the U.S. and abroad"),
        ("financials", "increases in IRA manufacturing credits recognized"),
    ],
}


def squash(text: str) -> str:
    return re.sub(r"\s+", "", text)


def run():
    file_rows, gold_rows = [], []
    new_only_keywords = Counter()
    new_only_examples, dropped_examples = [], []

    for file in list_pdfs(PDF_DIR):
        chunks = chunk_text(extract_text_from_pdf(file))
        before = [baseline_keyword_hit(c) for c in chunks]
        after = [keyword_hit(c) for c in chunks]
        file_rows.append({
            "file": file, "chunks": len(chunks),
            "sent_before": sum(before), "sent_after": sum(after),
        })

        for i, chunk in enumerate(chunks):
            if after[i] and not before[i]:
                matched = matched_keywords(chunk)
                new_only_keywords.update(matched)
                new_only_examples.append({"file": file, "chunk_index": i, "matched": matched,
                                          "preview": " ".join(chunk.split())[:200]})
            elif before[i] and not after[i]:
                dropped_examples.append({"file": file, "chunk_index": i,
                                         "preview": " ".join(chunk.split())[:200]})

        squashed = [squash(c) for c in chunks]
        for category, anchor in GOLD_PASSAGES.get(file, []):
            containing = [i for i, s in enumerate(squashed) if squash(anchor) in s]
            if not containing:
                raise AssertionError(f"Gold anchor not found in {file}: {anchor!r}")
            gold_rows.append({
                "file": file, "category": category, "anchor": anchor,
                "caught_before": any(before[i] for i in containing),
                "caught_after": any(after[i] for i in containing),
            })

    print("\nChunks sent to the LLM (lower = cheaper):")
    print(f"{'File':<18}{'Chunks':>8}{'Before':>8}{'After':>8}")
    for row in file_rows:
        print(f"{row['file']:<18}{row['chunks']:>8}{row['sent_before']:>8}{row['sent_after']:>8}")
    total_chunks = sum(r["chunks"] for r in file_rows)
    total_before = sum(r["sent_before"] for r in file_rows)
    total_after = sum(r["sent_after"] for r in file_rows)
    print(f"{'Total':<18}{total_chunks:>8}{total_before:>8}{total_after:>8}")

    print("\nGold passages reaching the LLM (higher = better recall):")
    print(f"{'File':<18}{'Passages':>9}{'Before':>8}{'After':>8}")
    for file in GOLD_PASSAGES:
        rows = [r for r in gold_rows if r["file"] == file]
        print(f"{file:<18}{len(rows):>9}{sum(r['caught_before'] for r in rows):>8}"
              f"{sum(r['caught_after'] for r in rows):>8}")
    print(f"{'Total':<18}{len(gold_rows):>9}{sum(r['caught_before'] for r in gold_rows):>8}"
          f"{sum(r['caught_after'] for r in gold_rows):>8}")
    for row in gold_rows:
        if not row["caught_after"]:
            print(f"  still missed ({row['file']}): {row['anchor']}")
        elif not row["caught_before"]:
            print(f"  newly caught ({row['file']}): {row['anchor']}")

    print(f"\nChunks newly sent: {len(new_only_examples)}; keywords responsible: "
          f"{dict(new_only_keywords.most_common())}")
    print(f"Chunks no longer sent: {len(dropped_examples)}")
    for row in dropped_examples:
        print(f"  dropped ({row['file']} #{row['chunk_index']}): {row['preview'][:120]}")

    llm_reference = None
    gap_report_file = LAB_DIR / "files" / "script3_gap_report.json"
    if gap_report_file.exists():
        rows = json.loads(gap_report_file.read_text())["all_rows"]
        positives = [r for r in rows if r["status"] == "ok" and r["llm_found_something"]]
        llm_reference = {
            "llm_positive_chunks": len(positives),
            "kept_before": sum(baseline_keyword_hit(r["chunk_text"]) for r in positives),
            "kept_after": sum(keyword_hit(r["chunk_text"]) for r in positives),
        }
        print(f"\nScript 3 LLM reference: {llm_reference['llm_positive_chunks']} chunks with findings; "
              f"prefilter kept {llm_reference['kept_before']} before, {llm_reference['kept_after']} after")
    else:
        print(f"\n{gap_report_file.name} not found; run script3KeywordGap.py for the LLM-reference recall.")

    output = LAB_DIR / "files" / "prefilter_eval_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({
        "files": file_rows, "gold": gold_rows, "llm_reference": llm_reference,
        "new_only": new_only_examples, "dropped": dropped_examples,
    }, indent=2))
    print(f"Results: {output}")


if __name__ == "__main__":
    run()
