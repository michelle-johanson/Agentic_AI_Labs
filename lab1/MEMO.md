# Lab Extension Memo

## 1. What We Extended
*3–5 sentences. Be specific: name the exact node, function, parameter, dataset, or architecture change you made relative to the baseline lab — not "we improved it."*

We changed how each job description gets embedded: instead of one mean-pooled vector per posting, a `chunk_text()` step in `createChromadb.py` splits the text into ~120-token sentence-packed pieces (each prefixed with a title/skills header) before passing it through the sentence-transformer, so the Chroma collection holds one row per chunk rather than one per job. All-MiniLM-L6-v2 has a 256-token context window and silently drops anything past it, and even when text does fit, averaging a long description into a single 384-dim vector dilutes the signal, so chunking fixes both the truncation and the dilution problem at once. We also pulled location, remote status, and company out of the semantic search path entirely: those get normalized into structured metadata fields (`city`, `state`, `is_remote`, `company_norm`) at index time, and a `parse_query()` function in `jobSearch.py` converts phrases like "in Denver" or "at Figma" into exact Chroma `where` filters. Skills and job titles stay in the embedding but also get a soft re-ranking boost (+0.10 each, +0.15 for a company match) on top of the cosine score in `boosted_search()`, since filtering them outright would exclude too many jobs on minor wording differences.

## 2. Why We Made This Choice
*What alternative(s) did you consider and reject? Why this one instead of those?*

**Longer context window** (max_seq_length=512) — tested first for the truncation problem; only fixes truncation, not the pooling dilution that comes from averaging a long description into one vector, so it recovered less than chunking overall.
**Stronger embedding models** (bge-small-en-v1.5, then all-mpnet-base-v2) — both picked up real gains on visible text, and mpnet in particular edged out a couple extra hits on keyword queries. But neither closed the gap on truncated tails the way chunking did, and mpnet's cost — 9x slower embedding, its own re-tuned score thresholds — made it a heavier lift for a small win once chunking was already handling the bigger problem.
**Full hybrid BM25 + vector search** — the standard fix for proper-noun weakness (company names, exact locations), but effectively runs a second retrieval engine alongside the embedding one, so we used metadata filters and lightweight boosts instead.
**Hard similarity cutoff** — considered instead of a match-quality label, but a hard cutoff hides a thin result pool rather than explaining it, so we added the label as a minor add-on instead.

## 3. Benefits
*What does this improve, and how would you show it — a number, a case that now passes, a before/after comparison?*

Chunking raised recall on queries targeting truncated tail content from 28% to 42%, and it also bumped recall on ordinary head queries from 75% to 87%, since shorter, sharper chunks beat one diluted vector even when nothing was getting cut off. The filters and boosting combo had the bigger swing: queries that were basically unsolvable before, like "Anthropic" (0/3 correct) or "data analyst in Denver" (1/5), jumped to 3/3 and 5/5 once the pipeline could enforce location and company as hard constraints instead of hoping they'd surface in the embedding.

## 4. Drawbacks & Tradeoffs
*What does this cost or risk — complexity, latency, $, a new failure mode, less generalizable?*

The query parser is deterministic and vocabulary-bound in both directions: a city, company, or skill not in the index falls back to plain cosine similarity with no warning, and phrasing it doesn't model — negation ("not remote"), ambiguous names ("Washington") — can produce a confident, filtered result set that isn't what was asked. Chunking also triples the vector count (857 vs. 285 for 285 postings), which doesn't matter at this scale but would add real storage and query overhead in a larger deployment. Similarity scores are model-specific, so swapping to mpnet required its own label thresholds; any future model change means re-calibrating. It's also worth being upfront that our numbers come from a modest, self-built test set — a few dozen queries scored against metadata rules rather than human relevance judgments — so they're most trustworthy as before/after comparisons between versions of our own system, not as absolute measures of search quality.

