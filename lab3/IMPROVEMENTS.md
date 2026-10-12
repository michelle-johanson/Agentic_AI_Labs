# Source evidence validation

## 1. What we extended

Added `EvidenceFinding(finding, evidence_quote)` in `docAnalyzerCommon.py` and changed
all six `ClimateDisclosure` fields from lists of strings to lists of these objects.
`validate_evidence()` checks every quotation against its input chunk, normalizing
whitespace only. `extract_with_retry()` in `script2PydanticStructured.py` now validates
quotes before returning a result and gives corrective instructions on retries, within
the existing three-attempt limit. Script 4 applies the same check to recovered findings.
Script 3 records failed extractions separately instead of treating them as empty results.

The original string-output prompt in script 1 remains the baseline. The new evidence
instructions are appended only in the structured and recovery paths. The model,
temperature, keyword list, and chunk sizes remain as supplied.

PDF and structured-report paths now resolve relative to the lab directory, so scripts
2-4 read and write the same `lab3/files` folder when launched from the repository root.
Reports from the old schema must be regenerated before comparing runs.

## 2. Why this change

Valid JSON and a valid Pydantic schema can still contain invented findings. The existing
`drop_parroted_description()` validator removes echoed schema descriptions, but cannot
establish that other outputs came from the filing. Quotes provide a source trail and
allow an inexpensive check before saving results.

We chose this change over increasing model size or running another LLM review on every
chunk: neither alternative directly requires source evidence, and both add inference
cost. Simply lowering temperature reduces randomness but does not verify provenance.

## 3. Benefits and evaluation

Run the short, offline evaluation from the repository root:

```powershell
python lab3/evalEvidence.py
```

It uses one keyword-selected chunk from each of `costco10K.pdf`, `eliLilly10K.pdf`, and
`tesla10K.pdf`. It compares the original list-of-strings schema against quote validation
using exact quotes, whitespace variants, invented quotes, and controlled alterations.
Results are printed as a before/after table and saved in
`lab3/files/evidence_eval_results.json`. Model calls are mocked for the retry and script
3-4 integration checks; no Ollama inference is performed.

Evaluation status: Python syntax compilation passed. Running `evalEvidence.py` with
the installed system Python stopped at `ModuleNotFoundError: No module named
'pydantic'`, before any evaluation cases executed. Run the command above in the lab's
configured environment; the repository's `AGENTS.md` says not to run the venv from
the shell. No before/after rejection counts or model-accuracy improvements have been
measured yet. A full extraction comparison on the three filings is still needed to
measure accuracy, recall, latency, and retry cost. No earlier lab3 evaluation script
was present to rerun.

## 4. Drawbacks and tradeoffs

- A quote can appear in the source while the finding misinterprets it or uses the wrong
  category. This check establishes provenance, not semantic correctness or materiality.
- Matching preserves case, punctuation, numbers, and units. Legitimate quotes with
  altered punctuation, PDF hyphenation, or combined passages are rejected; whitespace
  variations alone are accepted.
- Supporting quotes increase output length. Invalid quotes trigger retries, which can
  increase inference cost; after three failed attempts the chunk is recorded as failed.
- Rejecting the entire response for one invalid quote can lose other valid findings
  when retries fail. This conservative choice is explicit rather than silently saving
  partially validated findings.
- Old output consumers expecting strings must adapt to the new objects. Empty category
  lists remain valid, and item counts still count findings rather than object fields.

## Backlog (not implemented)

- Preserve PDF page numbers for each quote to make source review easier.
- Measure whether quote validation reduces hallucinations without losing disclosures
  on a manually labeled sample of the three filings.

## Lab report wording

If this were put into production, the one change we would make is to require source
evidence for every extracted finding. We added `EvidenceFinding` with `finding` and
`evidence_quote` fields and changed the six `ClimateDisclosure` categories to return
these objects. We updated `extract_with_retry()` to verify that each quote occurs in
its input chunk after normalizing whitespace, retrying and recording a failure when
validation does not pass. This adds an auditable source trail and a low-cost provenance
check, although a matching quote alone does not prove that the finding is correct.
