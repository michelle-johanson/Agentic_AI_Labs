# Lab Extension Memo

## Improvement 1: Embed the whole posting, not just the Description

1. What we extended (specifically):

   In `createChromadb.py`, the baseline set `texts = df["Description"].tolist()`, so
   only the Description column was embedded. Job title, Company, Location and Skills
   were stored as Chroma *metadata* — attached to each result for display, but never
   part of the vector that gets searched. We now build each document by concatenating
   text columns (newline-joined, after `fillna("")`/`astype(str)`) and pass that as
   `documents=` to `collection.add`. Metadata keeps the same fields (plus URL) but
   is coerced to strings so Chroma never sees a NaN. A `delete_collection("jobs")`
   step runs before `create_collection` so re-running the script replaces the old
   Description-only embeddings instead of erroring with "collection already exists".

   *Revision after Improvements 2-3:* we first embedded all five columns
   (`Job title`, `Company`, `Location`, `Skills`, `Description`) and measured.
   Company and Location contributed nothing reliable (see §3-4), cost ~7 tokens
   each against a 256-token budget, and are handled exactly by metadata
   filters/boosts in Improvements 2-3. The final index embeds **`Job title` +
   `Skills` + `Description`** only.

2. Why we made this choice (and not something else):

   The dataset has real information that lived only in metadata: 31 postings list
   Kubernetes in Skills, but 7 of them never say "Kubernetes" in the Description;
   2 of the 9 Denver jobs never mention Denver in their text. Those were
   unreachable by the baseline search.

   Alternatives we considered:
   - **Chroma metadata filters** (`collection.query(..., where={"Location": ...})`).
     Precise, but it requires the query to be parsed into structured fields (or a
     separate UI), and exact-match filters are brittle ("Denver" vs "Denver, CO",
     "Remote" vs "Remote, US"). Kept as a follow-up idea.
   - **One embedding per field** (separate vectors for title, skills, description
     and a weighted combination). More expressive but multiplies storage and
     needs custom scoring logic outside Chroma.
   - **Hybrid keyword + vector search** (BM25 on the Skills column). Good for exact
     terms, but a bigger change to the query path.

   Concatenation was chosen because it is a one-line change at index time, needs
   no change to `jobSearch.py`, and makes every column searchable with the same
   natural-language query.

3. Benefits (compare improvements before/after):

   We rebuilt the Description-only index in memory and ran the same 8 queries
   against both, counting how many of the top 5 results satisfy the query
   according to the metadata column (e.g. a "Terraform" hit must list Terraform
   in Skills):

   | query                          | before | after |
   |--------------------------------|-------:|------:|
   | Kubernetes                     | 5/5    | 5/5   |
   | Terraform infrastructure       | 4/5    | 5/5   |
   | Snowflake data engineering     | 4/5    | 5/5   |
   | business analyst               | 5/5    | 5/5   |
   | data analyst job in Denver     | 1/5    | 1/5   |
   | software engineer in Seattle   | 1/5    | 1/5   |
   | remote DevOps engineer         | 3/5    | 2/5   |
   | jobs at Anthropic              | 0/5    | 0/5   |
   | **total**                      | 23/40  | 24/40 |

   The gain is real but concentrated in **skills** queries: the Skills column is a
   dense list of exact technical terms, so adding it to the embedded text is pure
   signal. Title queries were already fine because titles are usually restated in
   the description.

4. Drawbacks and Tradeoffs (cost or risk):

   - **Longer documents → more truncation.** `all-MiniLM-L6-v2` reads at most 256
     tokens and silently drops the rest. Before this change 80/285 postings were
     over the limit; with all five columns **143/285** were; with the final
     Title+Skills+Description document it is **135/285**. The header fields go
     first so they survive, but the *end of the Description* (often the
     requirements section) is cut off for almost half the dataset. Resolved by
     chunking in Improvement 6.
   - **Location and Company barely improve.** One word out of ~200 cannot act as
     a hard constraint inside a single 384-dim vector, and embedding models don't
     understand proper nouns ("Anthropic" has no *meaning* to match on). With
     Company embedded, the one-word query "anthropic" ranked the three Anthropic
     jobs at **1, 6 and 158** — a signal exists, but it's a coin flip. This is
     why we ultimately dropped both columns from the embedding in favour of
     metadata filtering (Improvement 2) and boosting (Improvement 3).
   - **Must rebuild the index.** `delete_collection` is destructive; anyone with an
     existing `chroma/` folder has to re-run the script, and the old vectors are
     not recoverable.
   - Cosmetic: `jobSearch.py` previews `doc[:250]`, which now starts with the
     title/company/location header rather than the description text.


## Improvement 2: Natural-language metadata filters (location / remote / company)

1. What we extended (specifically):

   **Index side (`createChromadb.py`).** Chroma `where` filters are exact-match
   only (`$eq`, `$in`, `$and`, `$or`; `$contains` on metadata silently returns
   nothing), and the raw `Location` column has 113 distinct spellings —
   "remote" alone appears 13 ways ("Remote, US", "Remote (US)", "New York, NY
   (Remote eligible)", "Remote / Spokane, WA / Austin, TX / ..."). So at index
   time we derive clean, lowercase, filterable fields alongside the raw ones:
   `is_remote` (bool, regex `remote|virtual`), `city` and `state` (first
   `City, ST` pair via regex, excluding `US`), and `company_norm` (lowercased
   Company). Result: 70 remote postings, 221 with a city, 99 distinct cities.

   **Query side (`jobSearch.py`).** A `parse_query()` function splits the user's
   sentence into *semantic text* + *filters*:
   - `remote` / `work from home` / `wfh`  → `is_remote == True`
   - any city that exists in the index (a "gazetteer" built from the
     collection's metadata at startup, longest name first) → `city == ...`
   - `at|with|for <known company>` → `company_norm == ...`

   Several values in one field are collected into an `$in` — "at Anthropic or
   Figma", "in Denver or Seattle", "at Plaid, Ramp and Anthropic" — with
   `or` / `and` / `,` accepted as continuations once a first company has been
   found with a preposition. Matched phrases (and a leading "in/near/at") are
   stripped so only the job content gets embedded; 2+ fields are combined with
   `$and`. The CLI prints
   the detected filters so the user can see what was applied. Filtering happens
   *inside* `collection.query(where=...)`, i.e. before the nearest-neighbour
   search, so we always get the best 5 of the matching jobs instead of the
   1-2 survivors of post-filtering the top 5.

2. Why we made this choice (and not something else):

   Improvement 1 showed that hard constraints don't survive embedding: "Denver"
   is one word in ~200 and "Anthropic" has no meaning for the model to match on
   (1/5 and 0/5 respectively). Only a filter can enforce them.

   For *how the user expresses* the filter we considered:
   - **Separate CLI prompts** ("Location? Remote? Company?"). Most robust and
     trivial to build, but it breaks the "just type what you want" promise of
     semantic search and can't be reused by a chatbot/agent front-end.
   - **Inline syntax** (`data analyst loc:denver remote:yes`). Deterministic,
     but makes the user learn a mini-language.
   - **LLM extraction** (ask a model to return `{"city":..,"remote":..}` as
     JSON). Handles negation, states, synonyms and unknown companies gracefully,
     and is the natural agentic next step — but adds an API key, ~1s latency and
     per-query cost to what is currently a fully local tool.
   - **`where_document={"$contains": "Denver"}`** (substring on the embedded
     text). One-liner, but case-sensitive and matches "near our Denver office"
     in a remote posting's description.

   We chose regex + gazetteer parsing because it keeps the natural-language UX,
   stays fully local and deterministic, and its rules can be justified from the
   data: cities may match anywhere (none of the 99 are English words), but
   companies *require* a preposition because the company list contains `close`,
   `loop`, `branch`, `ramp`, `hive`, `fabric`, `raft`, `alt` — "close to Boston"
   or "feedback loop" would otherwise pick up a bogus company filter.

3. Benefits (compare improvements before/after):

   Same 8 queries and scoring as Improvement 1 (hits in top 5 judged by
   metadata). "Before" is the Improvement-1 index with no filters; "after" adds
   the parsed filters:

   | query                          | before | after | filter detected             |
   |--------------------------------|-------:|------:|-----------------------------|
   | Kubernetes                     | 5/5    | 5/5   | –                           |
   | Terraform infrastructure       | 5/5    | 5/5   | –                           |
   | Snowflake data engineering     | 5/5    | 5/5   | –                           |
   | business analyst               | 5/5    | 5/5   | –                           |
   | data analyst job in Denver     | 1/5    | 5/5   | city=denver                 |
   | software engineer in Seattle   | 1/5    | 5/5   | city=seattle                |
   | remote DevOps engineer         | 2/5    | 5/5   | is_remote=True              |
   | jobs at Anthropic              | 0/5    | 3/3   | company_norm=anthropic      |
   | **total**                      | 24/40  | 38/40 |                             |

   Queries with no recognisable constraint are untouched (`where=None`), so
   nothing regresses. Multi-constraint queries work: "data analyst in Denver at
   Ibotta" → `{city: denver, company_norm: ibotta}`, semantic text
   "data analyst".

4. Drawbacks and Tradeoffs (cost or risk):

   We ran a set of adversarial queries; these are the real failures:
   - **Negation is inverted.** "hybrid role, not remote" → `is_remote=True`.
     The parser sees the word, not the intent, and is confidently wrong.
   - **States are unsupported.** "jobs in Utah" → no filter. Two-letter codes
     can't be gazetteer-matched (`in`, `or`, `me`, `co` are English words) and
     we didn't add full state names; 64 postings have no `City, ST` at all
     ("Utah", "Arizona (Remote, US)").
   - **Unknown values fail silently.** "senior engineer at Google" (no Google in
     the data) → no filter, and the user gets generic engineer jobs with no
     warning. An LLM extractor or fuzzy matching would at least say "no such
     company".
   - **Filtering shrinks the pool, so junk fills the top 5.** Only 9 Denver jobs
     exist; "data analyst in Denver" now returns an *IT Support Specialist*
     (distance 0.695) in slot 5. Before, weak matches were out-competed;
     after, nothing is left to out-compete them. A distance cutoff is needed to
     make this safe (see Improvement 3 / backlog).
   - **Stripping can leave a degenerate semantic query.** "New York remote
     eligible" embeds just "eligible". "remote jobs" embeds "jobs".
   - **First city only.** "Bellevue, WA / Mountain View, CA" is filterable as
     Bellevue but not Mountain View, because Chroma metadata can't hold lists.
   - **Gazetteer is data-bound and loaded at startup** (`collection.get()` over
     all 285 metadatas — fine here, a cost at 285k). New cities/companies only
     become filterable after re-indexing.
   - **OR works within a field, not across fields.** "at Anthropic or Figma"
     and "in Denver or Seattle" work; "remote or in Boston" becomes
     remote AND Boston and returns nothing.
   - `is_remote` treats "Remote eligible" and "remote options in NC" the same
     as fully remote — a judgement call that could surprise a user.


## Improvement 3: Field-boosted re-ranking (exact-term evidence on top of the vector search)

1. What we extended (specifically):

   **Index side (`createChromadb.py`).** Two more normalized metadata fields,
   `title_norm` and `skills_norm` (lowercased; skills comma-split and
   re-joined), and the collection is now created with
   `metadata={"hnsw:space": "cosine"}`. Chroma's default is squared L2, which
   made the baseline's `1 - distance` "similarity" a meaningless number; in
   cosine space it is a real cosine similarity, which we need so the boosts
   below are on the same scale.

   **Query side (`jobSearch.py`).** The embedding remains the *only* retriever.
   `boosted_search()` asks Chroma for the top `N_CANDIDATES = 40` by cosine
   similarity (inside any Improvement-2 filters), then re-scores each candidate:

   ```
   score = cosine
         + 0.10 * (fraction of the query's known skills found in the job's Skills)
         + 0.10 if a known job-title phrase in the query appears in the job's title
         + 0.15 if a known company in the query is the job's company
   ```

   "Known" terms come from vocabularies built from the index at startup: the
   skills column split on commas (word-boundary matched, so `spark` hits
   `apache spark` but `sql` does not hit `mysql`), *core* job titles (seniority
   prefixes and team/level suffixes stripped — "Senior Data Scientist, Core Data
   - PhD" → "data scientist"; 2+ word phrases only), and company names. Because
   proper nouns embed badly, a named company's jobs are also pulled into the
   candidate pool via a `company_norm` filter so they can be boosted at all
   (they can otherwise sit at rank 150+). The CLI prints the boost terms it
   found and, per result, `score = cosine + boosts [...]` so the ranking is
   explainable.

2. Why we made this choice (and not something else):

   The class is about vector search, so the constraint was: keep the embedding
   as the engine, use exact terms only to *steer* it. Two things motivated it:
   (a) skills, titles and companies are exact tokens where embeddings are
   weakest — after Improvement 1 removed Company from the embedded text, a
   one-word query like "anthropic" or "plaid" returned 0 correct results; and
   (b) Improvement 2's parser fails silently on anything it doesn't recognise
   ("anthropic" without "at", a company that collides with an English word), so
   we wanted a *soft* second chance that doesn't exclude jobs when it misfires.

   Alternatives considered:
   - **Full hybrid search (BM25 + vector, reciprocal-rank fusion).** The
     standard industry answer and strictly more powerful (it can surface a job
     the embedding never retrieved). Rejected because it is a second search
     engine running beside the vector one — half the results would come from
     keyword search, which is exactly what we didn't want for this lab. Listed
     as the natural next step.
   - **Put more weight on the fields inside the embedding** (e.g. repeat the
     title/skills line). Fights the 256-token budget and gives no control over
     *how much* a field counts.
   - **Extend the filters to skills/titles** (Improvement 2 style). Filters are
     hard; "python" as a filter would exclude every job whose Skills column just
     forgot to list it. Boosts degrade gracefully.
   - **Cross-encoder re-ranker.** Better relevance, but a second model and
     ~50x slower per query; overkill at 285 documents.

3. Benefits (compare improvements before/after):

   Pure vector (with Improvement-2 filters) vs. the same plus boosting, top 5;
   denominator is min(5, jobs that actually satisfy the query) so a 1/1 is a
   perfect score:

   | query                                | vector | boosted | terms found                          |
   |--------------------------------------|-------:|--------:|--------------------------------------|
   | anthropic                            | 0/3    | 3/3     | company                              |
   | figma                                | 0/1    | 1/1     | company (+ skill "figma")            |
   | plaid                                | 0/1    | 1/1     | company                              |
   | data scientist                       | 3/5    | 5/5     | title                                |
   | data engineer                        | 5/5    | 5/5     | title                                |
   | business analyst                     | 5/5    | 5/5     | title                                |
   | sql                                  | 5/5    | 5/5     | skill                                |
   | python and sql                       | 4/5    | 4/5     | skills                               |
   | react typescript                     | 4/5    | 4/5     | skills                               |
   | data engineer with spark and kafka   | 2/3    | 2/3     | skills + title                       |
   | machine learning engineer in Seattle | 2/2    | 2/2     | skills + title (+ city filter)       |
   | **total**                            | 30/40  | 37/40   |                                      |

   The gains are exactly where embeddings are weakest: proper nouns (0 → 100%)
   and title precision ("data scientist" no longer mixes in data engineers).
   Skills queries were already strong after Improvement 1, so boosts mostly
   reorder ties there. Nothing regressed; queries with no recognised terms are
   returned in pure cosine order. Side benefit: every result now shows *why* it
   ranked where it did.

4. Drawbacks and Tradeoffs (cost or risk):

   - **Weights are hand-tuned guesses.** 0.10 / 0.10 / 0.15 were chosen by eye
     against a cosine spread of ~0.05-0.15 between ranks 1 and 5. There is no
     labelled relevance data to fit them; a different dataset may need different
     values, and a +0.15 company boost can lift a mediocre job above a better
     one from another company.
   - **Boosting cannot rescue what the vector search didn't retrieve.** Only
     the top 40 candidates are re-scored (plus the company recall guard). A
     perfect skills match at vector rank 60 is invisible. This is the price of
     not doing true hybrid retrieval.
   - **Vocabulary-bound, like Improvement 2.** Skills/titles/companies not in
     the index are ignored; synonyms aren't ("ML" ≠ "machine learning",
     "postgres" ≠ "postgresql"). Partial words inside multi-word skills ("ai"
     in "generative ai") were also missed — fixed in Improvement 5. Skills are
     only as good as the Skills column — a job that uses Python but didn't
     list it gets no boost.
   - **Title normalisation is heuristic.** Messy titles ("AI/NLP Engineer",
     "Senior/Staff ML Research Engineer, General Agents") reduce to junk cores;
     we drop single-word cores (11 titles) to avoid a "senior" boost term, at
     the cost of never boosting on those titles.
   - **Company boosting is bare-match**, so "close to Boston" does detect
     company `close`. Harmless here (a small nudge, and no Close job survives
     the Boston filter) but it is a latent false positive.
   - **Doesn't fix the junk-fills-top-5 problem** from Improvement 2: in the
     Denver example a DevOps job with cosine 0.25 still appears at #5 because
     only 9 Denver jobs exist. Addressed by the match labels in Improvement 7.
   - **Re-indexing required** (new metadata fields, cosine space) and ~40+
     candidate fetch per query instead of 5 — negligible at 285 docs, real at
     scale.
   - **Some scope leakage:** the `1 - distance` bug fix and the "explain the
     ranking" output rode along with this improvement rather than being their
     own items.


## Improvement 4: State-level location filter (unified location vocabulary)

1. What we extended (specifically):

   **Index side.** `location_fields()` in `createChromadb.py` now fills `state`
   for postings with no `City, ST` pair by matching a full state name ("Utah",
   "Arizona (Remote, US)") or, failing that, the first bare code in a list
   ("Hybrid (CT, VT, NH, MD, or MA)" → `ct`). The name→code table lives in a
   new shared module `us_states.py` so both scripts normalize identically.
   Also fixed a latent Improvement-2 bug found on the way: "CT, VT" was being
   parsed as city=`ct`, state=`vt`; a "city" that is itself a 2-letter code is
   now rejected. 223/285 postings have a state, 219 a city.

   **Query side.** `KNOWN_CITIES` (a list) became `KNOWN_LOCATIONS`, a dict of
   `term → (field, value)`: `"denver" → ("city","denver")`,
   `"colorado" → ("state","co")`, plus explicit aliases (`"washington state"`,
   `"new york state"`, `"nyc"`, `"washington dc"`). The parser collects every
   location mentioned into one `location` group and the `where` builder turns
   it into `city IN [...] OR state IN [...]` — so "Denver or Utah" is a single
   `$or` across two metadata fields, while location is still AND-ed with
   remote/company.

2. Why we made this choice (and not something else):

   64 postings have no city, and state-level queries ("jobs in Colorado") are a
   normal way to search; before, the only way into Colorado was to name Denver
   and get 9 of the 12 CO jobs.

   Design question: **one merged vocabulary or two separate lists?** Merged at
   the parser level, separate at the metadata level. A flat list would lose
   *which field* a term filters, and once one query can name a city and a state,
   the two fields must be OR-ed rather than AND-ed like every other field. The
   `term → (field, value)` dict keeps the parser a single loop and moves the
   field logic into the `where` builder. Alternatives:
   - **Two-letter codes as query terms** ("in CO"). Rejected: `in`, `or`,
     `me`, `co`, `hi`, `ok` are English words; matching them would break
     ordinary queries. Only full state names are recognised.
   - **Store a `states` list per posting** for the multi-state entries. Chroma
     metadata can't hold lists; would need one document per state (duplicates
     in results) or `where_document` tricks. Not worth it for 2 postings.
   - **Bare "New York" / "Washington" as state.** Both are city *and* state
     names. We let the city win (more specific) and added `"... state"`
     aliases. A superset rule (state) would also have been defensible.

3. Benefits (compare improvements before/after):

   Pure vector (no parser) vs. full pipeline, counting top-5 results whose
   `state` matches the query:

   | query                          | before | after |
   |--------------------------------|-------:|------:|
   | data engineer in Colorado      | 2/5    | 5/5   |
   | jobs in Utah                   | 1/5    | 5/5   |
   | software engineer in Texas     | 0/5    | 5/5   |
   | machine learning in California | 3/5    | 5/5   |
   | analyst in Virginia            | 1/5    | 5/5   |
   | devops in washington state     | 0/5    | 5/5   |
   | **total**                      | 7/30   | 30/30 |

   Pool sizes show the aggregation effect: "in Denver" → 9 jobs, "in Colorado"
   → 12, "in Denver or Utah" → 17, "new york state" → 23 vs. "New York" → 19.
   Improvements 2 and 3 re-ran unchanged (38/40 and 37/40).

4. Drawbacks and Tradeoffs (cost or risk):

   - **First state only.** "Hybrid (CT, VT, NH, MD, or MA)" is findable via
     Connecticut, not Vermont; "Remote (WA, OR, ID, AZ, TX, SC, GA)" only via
     Washington. Same list-in-scalar limitation as cities.
   - **"Washington" / "New York" ambiguity is resolved by fiat.** A Seattle
     job-seeker typing "in Washington" gets the 5 Washington, DC jobs; they
     must say "washington state". The CLI prints the detected filter, which is
     the only mitigation.
   - **No two-letter codes** ("in CO", "Denver, CO" works only via the city).
   - **Cross-field OR is still impossible**: "remote or in Texas" → remote AND
     Texas → 0 results. The `$or` we added is scoped to the location group.
   - **62 postings still have no state at all** ("Remote, US", "Remote, North
     America") — remote jobs can only be found via `remote`, never via a state,
     even when the poster would happily hire there.
   - **Vocabulary grew to ~150 location terms scanned per query** — still
     microseconds, but the regex-per-term loop is O(vocabulary), not
     O(query).


## Improvement 5: Partial skill keywords, weighted by rarity (IDF)

1. What we extended (specifically):

   The skills boost in Improvement 3 could only recognise *whole* Skills
   entries. No posting lists "ai" as a skill by itself — it only appears inside
   "generative AI", "Google Cloud Vertex AI", "medical AI", "LLM/AI-paired
   development" (12 postings) — so a query of "ai" never got a skills boost and
   Calendly's "generative AI" ML job ranked on cosine alone. Same for "llm"
   (jobs list "LLMs").

   In `jobSearch.py`, `find_terms()` now also scans the query for individual
   **tokens** that occur inside any posting's Skills (811 tokens, minus a
   14-word stopword list), and every skill term — phrase or token — carries an
   inverse-document-frequency weight `idf = log(N / doc_freq)`. The boost
   changed from *fraction of query skills matched* to *IDF-weighted fraction*:

   ```
   skills boost = W_SKILL * sum(idf of matched terms) / sum(idf of all query skill terms)
   ```

   Matching gained an optional trailing `s` so "llm" hits "llms". Whole-skill
   phrases are still matched first (and their words removed before the token
   scan), and title/company phrases are stripped before the skill scan so a
   query word counts as one thing only.

2. Why we made this choice (and not something else):

   The naive fix — add every word inside every skill to the vocabulary — makes
   filler words into boost terms: "management" (37 postings), "systems" (35),
   "requirements", "analysis", "design" (~20 each) would boost a third of the
   index on ordinary queries. Rarity is exactly the signal that separates "ai"
   (12 postings, idf 3.2) from "python" (103, idf 1.0), and IDF is the
   textbook way to encode it — one line, no tuning. Alternatives:
   - **Hand-written alias list** ("ai" → "generative ai", "llm" → "llms").
     Precise but never complete, and it has to be maintained per dataset.
   - **Character n-gram / fuzzy matching** on skills. Catches typos too, but
     "ai" is a 2-letter string that fuzzy-matches inside "email", "domain",
     "training" — far noisier than word-boundary tokens.
   - **Embed the skill list separately** and compare query vs. skills vectors.
     Elegant, but a second vector per job and the same proper-noun weakness.
   - **Stemming/lemmatising** (NLTK). Heavier dependency for what an optional
     "s" achieves on this data.

3. Benefits (compare improvements before/after):

   Before = Improvement 3 boosting (whole phrases only); after = with IDF
   tokens. Hit = top-5 result whose Skills contains the term
   (denominator = min(5, postings that do):

   | query             | before | after | detected            |
   |-------------------|-------:|------:|---------------------|
   | ai                | 3/5    | 5/5   | ai (idf 3.2)        |
   | llm               | 4/5    | 5/5   | llm (3.5)           |
   | llm engineer      | 4/5    | 5/5   | llm                 |
   | generative ai     | 1/2    | 2/2   | generative ai (5.7) |
   | spark             | 4/5    | 5/5   | spark (3.5)         |
   | react typescript  | 4/5    | 5/5   | typescript, react   |
   | python and sql    | 5/5    | 5/5   | python (1.1), sql (1.3) |
   | systems analyst   | 5/5    | 5/5   | title only          |
   | **total**         | 34/47  | 41/47 |                     |

   Calendly's "Senior Machine Learning Engineer" (Skills: generative AI) went
   from unboosted #4 to boosted #3 for "ai"; every top-5 result for "ai" now
   has an AI skill. Improvements 2-4 re-ran unchanged (38/40, 37/40, 30/30).

   The first attempt regressed "data scientist" from 5/5 to 4/5: the title
   words leaked into the token scan as the skill "data" (idf 2.2) and handed
   Data Engineers with a "data modeling" skill the same +0.10 the Data
   Scientists got from the title. Stripping title/company phrases before the
   skill scan fixed it — worth recording because it is the kind of interaction
   bug that only an eval catches.

4. Drawbacks and Tradeoffs (cost or risk):

   - **Title phrases now shadow skill tokens.** "ai engineer" matches the core
     title "ai engineer", so "ai" is *not* also a skill term; results are AI
     Engineer titles (correct), but a job whose title lacks "AI" and whose
     skills have it gets nothing. A word counts once — we chose title.
   - **Common tokens still boost, just less.** "data" (idf 2.2) or "aws"
     (1.7) alone still give a full W_SKILL to matching jobs because the weight
     is normalised within the query. An absolute IDF floor would fix it at the
     cost of another magic number.
   - **Half the token vocabulary is singletons** (404/811 appear in one
     posting): typos and one-off phrasings become boost terms with the highest
     possible weight (idf 5.65). Harmless unless a query happens to hit one.
   - **Optional-"s" is not stemming.** "analytics"/"analytic", "model"/
     "modeling", "ml"/"machine learning" still don't connect; ML → machine
     learning would need an alias table or an LLM parser (backlog).
   - **IDF is computed over the Skills column only**, so a term's weight
     reflects how often recruiters *listed* it, not how important it is. A
     skill everyone lists (python) is under-weighted even when the user cares
     about it most.
   - Slightly more startup work (two Counters over 285 postings) — negligible.


## Improvement 6: Chunking — one vector per passage instead of one per posting

1. What we extended (specifically):

   **Index side (`createChromadb.py`).** Each Description is split into
   sentence-packed chunks of at most `CHUNK_TOKENS = 120` tokens (one-sentence
   overlap between neighbours). Every chunk is prefixed with a
   `Job title\nSkills` header so it carries context on its own, and the whole
   chunk is guaranteed to fit the model's 256-token window. Each chunk is a
   Chroma row with the job's full metadata plus `job_id`, `chunk`, `n_chunks`.
   285 postings → 857 chunks (max 6 per job), 0 chunks over the limit.

   **Query side (`jobSearch.py`).** Retrieval fetches ~3× more rows (chunks),
   collapses them to the **best-matching chunk per job**, and only then applies
   the Improvement-3/5 boosts, which are per-job. Vocabularies and IDF counts
   de-duplicate by `job_id` so they still count postings, not chunks. The CLI's
   preview now shows the chunk that matched — i.e. *which passage* of the
   posting the query hit.

   Interaction fix found by the regression run: a query that is *only* a
   company name ("anthropic") is now a company filter rather than a boost.
   With small chunks a bare proper noun's cosines are ~0 against everything,
   so a +0.15 boost could no longer reliably beat noise (an unrelated Tendo
   chunk at 0.16 outranked Anthropic's third job at 0.00 + 0.15).

2. Why we made this choice (and not something else):

   The question we set out to answer was "is this just a matter of a better
   model?" — and the answer, measured, is no. Two different problems hide
   under "256 tokens":
   - **Truncation** — text the model never sees (9.3% of the corpus; 135
     postings; 48 of the lost tails contain requirements language).
   - **Dilution** — an embedding is a *pooled average* of its tokens, so one
     vector for a 400-token posting is a blurrier summary than one for a
     120-token paragraph, whatever the window size.

   A longer window fixes only the first. Chunking fixes both, is
   model-agnostic, and is the technique every RAG system uses regardless of
   embedding model — so it fits the assignment's goal of maximising the small
   model rather than swapping it. Alternatives, all measured in §3:
   - **`model.max_seq_length = 512`** — a one-line setting (the underlying BERT
     has 512 position slots) but the model was fine-tuned on ≤256; recovers
     less than chunking.
   - **Swap to `bge-small-en-v1.5`** (same size class, 512-token window).
     Better on visible text, *worse* on truncated tails — a "better model" did
     not solve the truncation problem.
   - **Long-context model** (`nomic-embed-text`, 8k tokens): ~6× the
     parameters, slower, still one diluted vector per posting. Not run.
   - **Trim boilerplate** (salary, benefits, EEO) before embedding: partial,
     brittle regexes; 43 of the 135 truncated postings lose ≤20 tokens of
     exactly this kind anyway.

3. Benefits (compare improvements before/after):

   **Test design.** For the 76 postings that lose ≥30 tokens we built two
   12-word queries each: a *tail* query from the text the baseline model never
   saw (preferring a sentence with requirements language) and a *head* query
   from the first 100 visible tokens as a control. Recall@5 = the posting is
   in the top 5 unique jobs; MRR = mean of 1/rank.

   | index                                  | tail R@5 | tail MRR | head R@5 | vectors |
   |----------------------------------------|---------:|---------:|---------:|--------:|
   | MiniLM, 256 tokens (baseline)          | 28%      | 0.14     | 75%      | 285     |
   | MiniLM, `max_seq_length = 512`         | 36%      | 0.23     | 75%      | 285     |
   | bge-small-en-v1.5, 512 tokens          | 21%      | 0.18     | 87%      | 285     |
   | **MiniLM, chunked (this improvement)** | **42%**  | **0.32** | **87%**  | 857     |

   Chunking gives the best recall *and* rank on hidden text, and — because of
   dilution — also lifts recall on *visible* text from 75% to 87%, matching
   the stronger bge-small model at zero model cost. The 512-token setting
   sees the same text as chunking yet recovers half as much: seeing ≠
   representing.

   **Chunk size sweep** (description-token budget → tail R@5 / head R@5 /
   vectors): 220 → 45/75/438; 160 → 45/83/617; **120 → 43/88/857**;
   80 → 51/87/1206; 50 → 49/88/1423. Head recall saturates at ~120; tail
   differences beyond that are a few queries out of 76. 120 (~one paragraph)
   is the knee.

   Improvements 2-5 re-ran unchanged (38/40, 37/40). Live example:
   "robotics perception and control" now surfaces Machina Labs at #2 — the
   matching sentence ("robotics experience in perception, control, or
   planning") was in the truncated tail.

4. Drawbacks and Tradeoffs (cost or risk):

   - **3× the vectors and index size** (857 vs 285), 3× the query fan-out
     (we fetch ~125 chunk rows to get 40 jobs). Trivial here; at 1M postings
     it's the dominant cost of the system.
   - **Tail recall is 42%, not 90%.** Many lost tails are short boilerplate
     ("and 401(k) match") or generic sentences shared by many postings; a
     12-word window from them isn't distinctive. The head-query ceiling for the
     same method is 87%, so chunking closes roughly a third of the gap.
   - **Verbatim test queries flatter every index.** Our tail/head queries are
     copied from the postings; real users paraphrase. The *relative* ordering
     of the four indexes is the finding, not the absolute numbers.
   - **Header repetition biases toward title/skills.** Each chunk carries the
     `Title\nSkills` header, so ~30 of every ~150 tokens are the same text;
     short chunks are proportionally more "header". Good for title queries,
     slightly worse for pure-description queries.
   - **Best-chunk scoring ignores breadth.** A posting that matches the query
     in three chunks scores the same as one that matches in one. Summing or
     averaging top-k chunks is an alternative we didn't explore.
   - **Sentence splitting is a regex** (`[.!?]` + whitespace): bullet lists,
     abbreviations ("Sr.", "e.g.") and missing punctuation produce odd
     chunks; a single sentence longer than the budget is still truncated.
   - **bge-small comparison is rough** — run without its recommended query
     instruction prefix, which likely understates it on short queries.
   - Rebuilding the index is required; the chroma folder grows.


## Improvement 7: Match-quality labels instead of a hard score cutoff

1. What we extended (specifically):

   Every result in `jobSearch.py` is now headed by a traffic-light label based
   on its final score (cosine + boosts): 🟢 **Strong match** (≥ 0.50),
   🟡 **Moderate match** (0.40-0.50), 🔴 **Weak match** (< 0.40). If even the
   top result is below the moderate line, a one-line note says nothing matched
   strongly and suggests rewording. The score line was also tidied: results
   with no boosts print `Score: 0.53 (cosine similarity)` instead of
   `+ boosts [none]`. Nothing is removed from the results list.

2. Why we made this choice (and not something else):

   Improvements 2 and 4 made this necessary: a location or company filter can
   shrink the candidate pool to a handful of postings, and the top 5 then fills
   with whatever is left — "data analyst in Denver" returned an IT Support
   Specialist at 0.27 that looked, on screen, like a peer of the 0.68 Sr. Data
   Analyst above it.

   We measured where the line should go rather than guess. Over the eval
   queries, correct results scored 0.43-0.92 (lower quartile 0.58); the four
   wrong-but-on-topic results scored 0.27-0.53; results for five deliberately
   off-topic queries ("forklift operator", "nurse practitioner", "marine
   biologist", ...) never exceeded 0.42. We then compared candidate bands on
   how they label those groups (correct → 🟢 / correct → 🔴 / off-topic → 🟡):
   0.60/0.45 gave 62% / 7% / 0%; **0.50/0.40 gave 92% / 0% / 3%**; loosening
   the weak line to 0.35 let 20% of off-topic results show as 🟡. The first
   version shipped at 0.60/0.45 and labelled every adjacent-role result inside
   a narrow filter ("software engineer in Utah" → an ML Engineer at 0.50) as
   merely moderate, which felt too strict in use; 0.50/0.40 is the loosest
   setting that still keeps off-topic results red.

   Alternatives considered:
   - **Hard cutoff** (drop results under 0.45). Rejected: after a narrow
     filter the user is often better served by seeing the weak results with a
     warning than by an empty screen, and a cutoff hides the system's
     behaviour instead of explaining it.
   - **Threshold on cosine only.** Rejected: boosts are legitimate evidence
     (a title match *is* a reason to trust a result); labelling on the score we
     actually rank by keeps the label consistent with the order.
   - **Relative labels** (e.g. within 80% of the top score). Rejected: the top
     result would always be "strong", including for "forklift operator".

3. Benefits (compare improvements before/after):

   Before: five results, five scores, no guidance. After, on the same queries:
   - "data engineer python sql" → 🟢🟢🟢🟢🟢 (0.77-0.83).
   - "data analyst in Denver" → 🟢 Sr. Data Analyst 0.68, 🟢 ML Engineer
     0.53, 🟡 ×2 (0.47-0.49), 🔴 IT Support Specialist 0.27 — the
     junk-fills-top-5 problem is now visible instead of silent.
   - "software engineer in Utah" → 🟢 Senior Software Engineer 0.60,
     🟢 Senior ML Engineer 0.50, 🟡 ML Engineer / SRE / IT Support
     (0.43-0.46) — only 8 Utah postings exist, and the labels say which are
     the real thing.
   - "forklift operator" → info note + 🔴 ×5 (0.28-0.29) — the system says
     "we don't have this" rather than confidently offering a Data Engineer.
   Ranking is unchanged, so no eval numbers moved (38/40, 37/40).

4. Drawbacks and Tradeoffs (cost or risk):

   - **Two magic numbers, tuned on ~100 labelled results from our own eval
     queries, then loosened once by feel.** They are specific to this model (cosine ranges differ between
     embedding models), to cosine space, and to the boost weights: change any
     of those and the bands must be re-measured.
   - **Score, not cosine, so boosts can promote a label.** A weak-cosine job
     with a title + skills + company match (+0.35) can reach 🟢 on exact-term
     evidence alone. Arguably right, but a 🟢 no longer means "semantically
     close".
   - **Three coarse bands.** 0.44 and 0.46 get different colours; 0.46 and
     0.59 get the same one. The numeric score is still printed for that
     reason.
   - The labels describe *retrieval confidence*, not job quality or fit; a
     user could over-read a 🟢.


## Improvement 8: Swappable embedding model — all-mpnet-base-v2 vs. all-MiniLM-L6-v2

1. What we extended (specifically):

   The embedding model is now a single setting read by both scripts:
   `EMBED_MODEL` (env var, default `all-MiniLM-L6-v2`). `createChromadb.py`
   also takes the chunk window from the model itself (`model.max_seq_length`:
   256 for MiniLM, 384 for mpnet) so chunking adapts automatically. Setting
   `EMBED_MODEL=all-mpnet-base-v2` and re-running `createChromadb.py` re-embeds
   the corpus with the larger model: 110M parameters (vs 22M), 768-dimensional
   vectors (vs 384), 384-token window (vs 256). The hypothesis was that the
   larger representation captures more detail from descriptions and queries
   and therefore returns more relevant jobs for the same test queries.

2. Why we made this choice (and not something else):

   `all-mpnet-base-v2` is the sentence-transformers authors' recommended
   general-purpose model and the natural "bigger sibling" of MiniLM: same
   training recipe and same pooling, so it isolates the effect of model
   capacity from everything else. We chose to test it *through the full
   pipeline* (filters, boosts, chunking) rather than in isolation, because a
   model swap only matters if it changes what the user sees. Alternatives:
   - **`bge-small-en-v1.5`** (33M, 512 tokens) — tested in Improvement 6:
     better on visible text, worse on truncated tails; roughly MiniLM's size.
   - **Long-context models** (`nomic-embed-text`, 8k tokens) — not run; the
     dilution result in Improvement 6 predicts limited benefit from window
     size alone.
   - **Fine-tuning MiniLM on job-search pairs** — the highest-ceiling option,
     but needs labelled query→job data we don't have.

3. Benefits (compare improvements before/after):

   Everything else held fixed (same chunking, filters, boosts, test queries):

   | measure                                   | MiniLM-L6-v2 | mpnet-base-v2 |
   |-------------------------------------------|-------------:|--------------:|
   | filter queries (eval 2), full pipeline    | 38/40        | 38/40         |
   | keyword/boost queries (eval 3), full pipeline | 37/40    | **39/40**     |
   | pure vector, no boosts (eval 3 "before")  | 35/40        | 35/40         |
   | head R@5, one vector per posting          | 76%          | **84%**       |
   | tail R@5, one vector per posting          | 28%          | 29%           |
   | head R@5, chunked                         | **87%**      | 84%           |
   | tail R@5, chunked                         | **42%**      | 37%           |
   | embed 285 postings                        | 0.9 s        | 8.4 s (9×)    |
   | full index build                          | 12 s         | 43 s          |
   | query latency, full pipeline              | 34 ms        | 45 ms         |
   | vector size / model download              | 384-d / ~90 MB | 768-d / ~420 MB |

   The larger model *is* better at representing a whole posting in one vector
   (head recall 76% → 84%) and picks up two extra hits on multi-skill queries
   ("python and sql", "react typescript" both 4/5 → 5/5). Its 384-token window
   also sees most of the text MiniLM truncates.

   But the two gains overlap almost entirely with what chunking already
   bought: once postings are chunked, MiniLM matches or beats mpnet on both
   visible text (87% vs 84%) and hidden text (42% vs 37%), and the end-to-end
   filter results are identical. A bigger window did not recover the tails
   (29% vs 28% single-vector) — the same dilution effect as Improvement 6.
   Net effect of the swap on the finished system: +2 results out of 40 on
   keyword queries, nothing elsewhere.

4. Drawbacks and Tradeoffs (cost or risk):

   - **9× slower embedding, 3.6× slower index build, 2× vector storage,
     ~4.5× download**, +30% query latency. Trivial at 285 postings; at real
     scale it is the difference between re-indexing nightly and hourly.
   - **The match-label thresholds don't transfer.** mpnet's cosines run
     higher *and* overlap more: off-topic results reach 0.46 (MiniLM: 0.42)
     while correct adjacent-role results go as low as 0.33 (MiniLM: 0.43).
     With MiniLM's bands, 26% of "forklift operator" results would show 🟡.
     We re-measured and made the bands per-model (`MATCH_BANDS` in
     `jobSearch.py`: mpnet 0.55/0.45 → 72% of correct results 🟢, 7% 🔴, 3%
     of off-topic 🟡 — a slightly worse separation than MiniLM's 92/0/3).
     Any further model needs its own row.
   - **Gains are small and partly within noise** — 2 queries out of 40, on a
     hand-built eval. A larger labelled set is needed before claiming the
     model is better *for this task*.
   - **It is the one improvement that doesn't teach anything about using the
     small model well** — which is the point of the exercise. We kept MiniLM
     as the default and left mpnet as a fully integrated alternative
     (`EMBED_MODEL=all-mpnet-base-v2`, own chunk window, own label bands) for
     anyone who values the extra +2 over the 9× cost.
   - The index is model-specific: switching `EMBED_MODEL` without
     re-running `createChromadb.py` yields silent garbage (a 384-d query
     against 768-d vectors, or vice versa, errors — but two different
     384-d models would not).


## Backlog: other ideas we noticed (not yet written up)

- **Chunk scoring variants** — sum/average of top-k chunk scores instead of
  best-chunk, to reward postings that match in several places.
- **Paraphrased tail-query eval** — our truncation eval uses verbatim
  snippets; a human-written (or LLM-written) paraphrase set would be fairer.
- **Full hybrid search (BM25 + vector, RRF)** — the step beyond Improvement 3;
  can surface jobs the embedding never retrieved.
- **LLM-based query understanding** — replaces `parse_query()` + `find_terms()`
  with a generative model returning JSON (semantic_text, cities, states,
  is_remote incl. negation, companies, skills, unknown_terms). The embedding
  model can't do this (it only maps text → vector); it needs a *generative*
  model — hosted (Claude API, ~0.5-1s, needs a key) or local (Ollama + a
  3B-7B model, no key, slower, weaker JSON compliance). Fixes every parser
  failure in Improvement 2 §4; costs non-determinism, latency, a dependency,
  and it is the idea least about vector search itself. Deliberately left for a
  teammate.
- **Learned boost weights** — collect a small set of query→relevant-job labels
  and grid-search the three weights instead of guessing.
