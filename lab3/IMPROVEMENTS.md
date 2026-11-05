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

The committed `files/evidence_eval_results.json` contains 18 controlled cases:
all 12 invalid quote variants were accepted by the old string schema and rejected
by source checking; all 6 verbatim/whitespace-only variants were accepted by both.
These are validator results, not measured LLM accuracy. The earlier system-Python
attempt lacked `pydantic`; the presence of saved results supersedes the earlier
note that no rejection counts were available.

For a fresh run and LangSmith records, `runEvidenceStudy.py` reruns the complete
existing evaluation, records the individual cases and aggregate results, and can
also compare live extraction with the same model and temperature on both sides.
Script 1's 3B and script 2's 7B results should not be used to isolate the benefit
of quote validation because their models differ.

```powershell
python lab3/runEvidenceStudy.py
python lab3/runEvidenceStudy.py --trace
python lab3/runEvidenceStudy.py --trace --live --limit 6
```

Tracing requires a locally configured `LANGSMITH_API_KEY`. Use
`LANGSMITH_PROJECT=lab3-camden-evidence`. The optional `--share` flag creates
public links for the recorded filing excerpts and outputs; otherwise links are
private. Timestamped controlled/live results and trace links are saved under
`lab3/files`. Live outputs include blank human-review fields for unsupported
findings, omissions, and category errors. Model calls alternate variant order
and retain the same 0.3 temperature and three-attempt cap. The sample is small
and selected by the existing keyword filter, not a broad accuracy estimate.

Preparation status (October 13, 2026): Ollama is running with `qwen2.5:3b`,
but `qwen2.5:7b` needs to be downloaded for the default live comparison. No fresh
runtime/accuracy measurements or LangSmith links have been produced in this
session at preparation time. The user will run the configured environment and longer commands,
as required by `AGENTS.md`. The measurement runner's syntax and isolated
validation/retry behavior were checked separately before handing over commands.

Fresh user-run validation (October 13, 2026): the complete original evaluation
passed, reproducing 12/12 invalid variants accepted before and 0/12 after,
with all 6 valid quote/whitespace variants accepted. Results are saved in
`files/controlled_evidence_20261013_125828_ca5efb.json`. LangSmith created
15 public case links; the summary and three early cases initially returned
404 during sharing. Credentials worked for later cases, consistent with delayed
visibility rather than a blanket authentication failure.

The runner now uses the supported v2 sharing resource with explicit project
and trace coordinates, retries transient availability failures, preserves
existing public links, and offers `--repair-links` to retry the saved run IDs
without repeating PDF tests or model calls. Isolated tests covered delayed
visibility, coordinate arguments, preservation of working links, and immediate
reporting of permission errors. Live extraction evaluation remains pending.

The user subsequently ran the repair command successfully: all 19 links (18
individual controlled cases plus one summary) are available, with no remaining
link errors. The public aggregate evidence is:
https://smith.langchain.com/public/8c70dfd3-3533-448d-8dbd-59e389fa8aff/r
The summary records 12 invalid variants accepted before and zero after, and
6 valid variants accepted both before and after. These remain controlled
validation results, not measured live-model accuracy. Project and result
screenshots still need to be added to the final memo by the group.

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
