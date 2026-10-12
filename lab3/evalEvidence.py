"""Check evidence validation on real PDF text and controlled bad responses.

Run from the repository root: python lab3/evalEvidence.py
No Ollama calls are made. These are validation tests, not model-accuracy scores.
"""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import Mock, patch

from pydantic import Field, ValidationError, create_model

from docAnalyzerCommon import (
    ClimateDisclosure, EvidenceFinding, EXPECTED_KEYS, LAB_DIR, PDF_DIR,
    chunk_text, extract_text_from_pdf, keyword_hit, list_pdfs, validate_evidence,
)
import script2PydanticStructured as structured
import script3KeywordGap as gap
import script4HybridRecovery as recovery


# Reproduce the original six List[str] fields for the validation comparison.
LegacyClimateDisclosure = create_model(
    "LegacyClimateDisclosure",
    **{category: (list[str], Field(default_factory=list)) for category in EXPECTED_KEYS},
)


def make_disclosure(quote: str, finding: str = "Climate-related source excerpt"):
    return ClimateDisclosure(climate_risks=[EvidenceFinding(
        finding=finding, evidence_quote=quote,
    )])


def accepts(disclosure, chunk):
    try:
        validate_evidence(disclosure, chunk)
    except ValueError:
        return False
    return True


def run():
    rows = []
    samples = []
    for file in list_pdfs(PDF_DIR):
        chunks = chunk_text(extract_text_from_pdf(file))
        index, chunk = next((i, c) for i, c in enumerate(chunks) if keyword_hit(c))
        samples.append((file, index, chunk))
        # This tests quote provenance, not whether the excerpt is a material disclosure.
        quote = " ".join(chunk.split())[:160].rsplit(" ", 1)[0]
        fabricated = "The company emitted 987654321 tonnes of CO2e in 2099."
        assert fabricated not in chunk
        cases = [
            ("verbatim", quote, True),
            ("whitespace_only", "\n\t".join(quote.split()), True),
            ("invented_quote", fabricated, False),
            ("altered_punctuation", quote + " !!!", False),
            ("altered_case", quote.swapcase(), False),
            ("stitched_passages", quote[:40] + " ... " + quote[-40:], False),
        ]
        for name, evidence, expected in cases:
            actual = accepts(make_disclosure(evidence), chunk)
            assert actual == expected, (file, name, actual)
            rows.append({
                "file": file, "chunk_index": index, "case": name,
                # Legacy List[str] accepts all of these strings without a source check.
                "before_source_check": bool(LegacyClimateDisclosure.model_validate(
                    {"climate_risks": [evidence]}
                ).climate_risks),
                "after_source_check": actual,
                "expected": expected,
            })

    # Validate that normalization does not hide changed numbers or units.
    source = "Scope 1 emissions were 12.5 tonnes of CO2e."
    assert accepts(make_disclosure(source), source)
    assert not accepts(make_disclosure(source.replace("12.5", "125")), source)
    assert not accepts(make_disclosure(source.replace("tonnes", "kg")), source)
    assert accepts(ClimateDisclosure(), "")
    for blank in ("", " \n\t"):
        try:
            make_disclosure(blank)
        except ValidationError:
            pass
        else:
            raise AssertionError("Blank evidence was accepted")
    try:
        ClimateDisclosure(climate_risks=["Legacy string without evidence"])
    except ValidationError:
        pass
    else:
        raise AssertionError("Legacy string was accepted as an evidence object")
    for category in EXPECTED_KEYS:
        item = EvidenceFinding(finding="A finding", evidence_quote=source)
        disclosure = ClimateDisclosure(**{category: [item]})
        assert accepts(disclosure, source)
        assert not accepts(disclosure, "Different source")
        assert disclosure.total_items() == 1
    parrot = ClimateDisclosure(climate_risks=[EvidenceFinding(
        finding=ClimateDisclosure.model_fields["climate_risks"].description,
        evidence_quote="Not real source evidence",
    )])
    assert parrot.is_empty()

    _, _, chunk = samples[0]
    quote = " ".join(chunk.split())[:100].rsplit(" ", 1)[0]
    valid = make_disclosure(quote)
    invalid = make_disclosure("Invented quote absent from this chunk.")
    assert not accepts(invalid, chunk)
    with patch.object(structured, "chain", Mock()) as mock_chain:
        invoke = mock_chain.invoke
        invoke.side_effect = [invalid, valid]
        result, attempts, error = structured.extract_with_retry(chunk)
        assert result == valid and attempts == 2 and error is None
        assert invoke.call_args_list[1].args[0]["retry_feedback"]
    with patch.object(structured, "chain", Mock()) as mock_chain:
        invoke = mock_chain.invoke
        invoke.return_value = invalid
        result, attempts, error = structured.extract_with_retry(chunk)
        assert result is None and attempts == structured.MAX_RETRIES and error
        assert invoke.call_count == structured.MAX_RETRIES
    with patch.object(structured, "chain", Mock()) as mock_chain:
        mock_chain.invoke.return_value = ClimateDisclosure()
        result, attempts, error = structured.extract_with_retry(chunk)
        assert result.is_empty() and attempts == 1 and error is None
    with patch.object(structured, "chain", Mock()) as mock_chain:
        mock_chain.invoke.side_effect = RuntimeError("Provider error")
        result, attempts, error = structured.extract_with_retry(chunk, max_retries=2)
        assert result is None and attempts == 2 and error == "Provider error"
    try:
        structured.extract_with_retry(chunk, max_retries=0)
    except ValueError:
        pass
    else:
        raise AssertionError("An empty retry budget was accepted")

    with TemporaryDirectory(prefix="lab3_evidence_") as temp_dir:
        # Script 3 must not label failed validation as a successful empty result.
        with (
            patch.object(gap, "list_pdfs", return_value=["sample.pdf"]),
            patch.object(gap, "extract_text_from_pdf", return_value=chunk),
            patch.object(gap, "chunk_text", return_value=[chunk]),
            patch.object(gap, "extract_with_retry", return_value=(None, 3, "Bad evidence")),
            patch.object(gap, "OUTPUT_FILE", Path(temp_dir) / "gap.json"),
        ):
            gap.run()
            report = json.loads(gap.OUTPUT_FILE.read_text())
            assert len(report["failed"]) == 1 and not report["wasted_by_keywords"]

        # Script 4 must reject invalid quotes instead of counting them as recovered.
        check_input = Path(temp_dir) / "input.json"
        check_input.write_text(json.dumps({"wasted_by_keywords": [
            {"file": "sample.pdf", "chunk_index": 0, "chunk_text": chunk},
            {"file": "sample.pdf", "chunk_index": 1, "chunk_text": chunk},
        ]}))
        with (
            patch.object(recovery, "GAP_REPORT_FILE", check_input),
            patch.object(recovery, "OUTPUT_FILE", Path(temp_dir) / "recovery.json"),
            patch.object(recovery, "recheck_chain", Mock(
                invoke=Mock(side_effect=[invalid, valid])
            )),
        ):
            recovery.run()
            report = json.loads(recovery.OUTPUT_FILE.read_text())
            assert len(report["failed"]) == 1 and len(report["recovered"]) == 1

    totals = []
    for name in sorted({row["case"] for row in rows}):
        cases = [row for row in rows if row["case"] == name]
        totals.append({
            "case": name, "tested": len(cases),
            "before_accepted": sum(row["before_source_check"] for row in cases),
            "after_accepted": sum(row["after_source_check"] for row in cases),
        })
    print("\nControlled responses against actual PDF text (not LLM accuracy):")
    print("Case                     Tested  Before accepted  After accepted")
    for total in totals:
        print(f"{total['case']:<24} {total['tested']:>6} "
              f"{total['before_accepted']:>16} {total['after_accepted']:>15}")
    output = LAB_DIR / "files" / "evidence_eval_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps({"totals": totals, "rows": rows}, indent=2))
    print("Schema, all categories, blank evidence, numbers/units, retry bounds, and scripts 3-4: passed")
    print(f"Results: {output}")


if __name__ == "__main__":
    run()
