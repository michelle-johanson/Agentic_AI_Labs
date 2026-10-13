# Improvement 2: Keyword prefilter recall

## 1. What we extended

`KEYWORDS`, `keyword_hit()`, and `matched_keywords()` in `docAnalyzerCommon.py`. These
decide which chunks scripts 1, 2, and 4 send to the LLM, so anything they reject is never
extracted.

- **Matching rule.** Was a raw substring test (`kw in chunk`). Now a regex per keyword
  (`_keyword_pattern()`, compiled once into `KEYWORD_PATTERNS`): the keyword must start
  at a word boundary, and spaces/hyphens/line breaks inside a phrase are interchangeable.
  Acronyms in `WHOLE_WORD_KEYWORDS` (`esg`, `zev`) must also end at a word boundary.
- **Removed:** bare `clean` (its only solo hit outside climate text was Eli Lilly's
  Superfund "cleanup"), and the `sea-level` / `net-zero` duplicates the new rule makes
  redundant.
- **Added** transition-risk and climate-finance terms the SEC rule cares about:
  `regulatory credit`, `renewable energy`, `renewable power`, `clean energy`, `clean fuel`,
  `clean power`, `clean technolog`, `zero-emission`, `zev`, `fuel economy`, `sustainab`,
  `esg`, `environmental, social`, `energy transition`, `paris agreement`, `tcfd`, and
  physical-risk terms `storm`, `typhoon`, `extreme heat`, `heat wave`.

New file: `lab3/evalPrefilter.py` (offline, no Ollama calls).

## 2. Why this and not the alternatives

The prefilter sat in front of everything: about 26 of the ~850 chunks in the three 10-Ks
reached the model. Tesla mentions regulatory credits 20 times (a $2.76B revenue line that
exists only because of greenhouse-gas and ZEV regulation) and no keyword matched any of it.
Improving extraction quality downstream can't recover text the model never sees.

Rejected:
- **Drop the prefilter and send every chunk** (script 3's approach): ~850 LLM calls x up to
  3 attempts per run, about 14x the new cost, for the same three filings.
- **Embedding-similarity prefilter** (sentence-transformers, as in labs 1-2): likely better
  on paraphrases, but adds a model load, a threshold to tune, and is harder to explain
  why a chunk was kept. Worth trying next, measured against this eval.
- **Add product words like `solar`, `energy storage`, `electric vehicle`**: Tesla uses them
  268 / 67 / 37 times, mostly in product descriptions that aren't disclosures under the
  rule. Prototyping showed they flood the filter.
- **Add `IRA` / `inflation reduction act`**: tried and removed. It matched Eli Lilly's
  drug-pricing provisions of the same law (9 chunks) and a Tesla director named Ira.
- **Keep bare `renewable`**: tried and replaced with `renewable energy` / `renewable power`
  after it matched a Tesla loan "renewable until March 2026".

## 3. Benefits

`python lab3/evalPrefilter.py` builds the real chunks and checks 27 hand-picked gold
passages (verbatim anchors that meet the rule's risk / targets / financials / mitigation
categories) plus the number of chunks sent.

| File | Chunks | Sent before | Sent after | Gold before | Gold after |
|---|---|---|---|---|---|
| Costco | 179 | 9 | 13 | 8/8 | 8/8 |
| Eli Lilly | 346 | 5 | 5 | 3/3 | 3/3 |
| Tesla | 322 | 12 | 43 | 8/16 | 14/16 |
| **Total** | 847 | **26** | **61** | **19/27** | **25/27** |

Newly caught: all four Tesla regulatory-credit passages (segment description, the
$2,763M revenue table row, the $973M MD&A change, the revenue-driver explanation) and both
renewable-energy transition-risk passages. Nothing in the gold set was lost. The one chunk
that stopped being sent is the Eli Lilly Superfund/patent-litigation chunk.

**Numbers above were produced with macOS PDFKit text and a fixed 1,500/300 window**, because
the venv isn't run from the agent shell. `evalPrefilter.py` uses the real `PyPDF2` +
`RecursiveCharacterTextSplitter` path; run it in the venv and replace this table with its
output. If `files/script3_gap_report.json` exists, it also reports recall against the
LLM's own findings on every chunk, which is the less biased measure.

Re-run of earlier evals: `evalEvidence.py` picks its sample chunk with `keyword_hit()`, so
its chosen chunk may change. Its assertions don't depend on which chunk, but it has not
been re-run yet (needs the venv).

## 4. Drawbacks and tradeoffs

- **The gold set is mine and small.** I picked the 27 passages and the new keywords from
  the same reading of the filings, so 25/27 overstates how well this generalizes to a
  fourth 10-K. The script 3 comparison is the honest check.
- **2.3x more LLM calls** (26 -> 61), almost all on Tesla. Script 2 runtime grows roughly
  in proportion.
- **Real false positives among the 36 newly sent chunks:** "unable to sustainably meet our
  timelines" (supplier risk), "individuals passionate about sustainable energy" (intern
  hiring), NHTSA fuel-economy *labeling* rules, and Costco "stormwater" (matched `storm`).
  `sustainab` is the noisiest addition (11 chunks).
- **Still missed:** EV incentive-policy risk ("incentives that support... electric
  vehicles") and "IRA manufacturing credits". Catching them needs `IRA` or `electric
  vehicle`, which the bullets above show are too noisy as keywords.
- **Tuned by eye:** the keyword list, the decision to drop bare `clean`, and which acronyms
  need whole-word matching were all chosen by inspecting matches, not by a held-out test.
- **Prefix matching** means `storm` also hits "stormwater" and `flood` still hits
  "flooding" (intended). Within-word PDF artifacts like "regulat ory" would defeat any
  keyword; PDFKit showed these mostly around the letter "v", so it hasn't mattered for the
  current list, but `PyPDF2` may differ.

---

# Improvement 1: Source evidence validation

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
- Quote check is all-or-nothing and its retry message is generic: pass `last_error`
  (the exact failing quote) back to the model, and drop only the bad findings.
- Normalize PDF artifacts (line-break hyphenation, split words like "competitiv e")
  before quote matching; there are 11-14 hyphenated line breaks per filing.
- Prompt is ~1,300 tokens before the chunk and schema; confirm Ollama's default context
  isn't truncating it, or set `ChatOllama(num_ctx=...)`.
- Embedding-similarity prefilter as an alternative to keywords, scored with
  `evalPrefilter.py`.
- Company-level report: dedupe findings across overlapping chunks.
- `requirements.txt` is missing `PyPDF2`, which every lab3 script imports.
- Script 1 (3b model) vs script 2 (7b model) comparison mixes model size into the
  structured-output result.

## Lab report wording

If this were put into production, the one change we would make is to require source
evidence for every extracted finding. We added `EvidenceFinding` with `finding` and
`evidence_quote` fields and changed the six `ClimateDisclosure` categories to return
these objects. We updated `extract_with_retry()` to verify that each quote occurs in
its input chunk after normalizing whitespace, retrying and recording a failure when
validation does not pass. This adds an auditable source trail and a low-cost provenance
check, although a matching quote alone does not prove that the finding is correct.
