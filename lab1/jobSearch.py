import chromadb
from chromadb.config import Settings
import numpy as np
import re
import math
from collections import Counter
from pathlib import Path
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
from us_states import US_STATES
import os
load_dotenv()

EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")   # must match createChromadb.py

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(anonymized_telemetry=False)
)

sbert_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)
collection = client.get_collection(name="jobs", embedding_function=sbert_ef)

# The collection holds one row per description *chunk*, and every chunk carries
# its job's full metadata plus job_id. Vocabularies, IDF counts, and candidate
# pools all need to count jobs, not chunks, so we de-duplicate by job_id once
# at startup.
_chunk_meta = collection.get(include=["metadatas"])["metadatas"]
_all_meta = list({m["job_id"]: m for m in _chunk_meta}.values())
CHUNKS_PER_JOB = len(_chunk_meta) / len(_all_meta)

# --- Main improvement #2: metadata filters + term boosting -----------------
# Vocabularies built from the index, used below to recognize hard constraints
# (location/remote/company) and exact-term boosts (skills/title/company) in
# the query text.

# term -> (field, value): "denver" -> ("city", "denver"), "colorado" -> ("state", "co").
# A city and a state can appear in the same query ("Denver or Utah"), so both
# map into one "location" group and get OR-ed together at query time.
KNOWN_LOCATIONS = {}
for name, code in US_STATES.items():
    KNOWN_LOCATIONS[name] = ("state", code)
    KNOWN_LOCATIONS[f"{name} state"] = ("state", code)      # "washington state", "new york state"
for m in _all_meta:
    if m.get("city"):
        KNOWN_LOCATIONS[m["city"]] = ("city", m["city"])    # city wins for "new york" / "washington"
KNOWN_LOCATIONS["washington dc"] = KNOWN_LOCATIONS["washington, dc"] = ("city", "washington")
KNOWN_LOCATIONS["nyc"] = KNOWN_LOCATIONS["new york city"] = ("city", "new york")
# longest-first so "salt lake city" / "new york state" are tried before their prefixes
KNOWN_LOCATION_TERMS = sorted(KNOWN_LOCATIONS, key=len, reverse=True)
KNOWN_COMPANIES = sorted({m["company_norm"] for m in _all_meta if m.get("company_norm")}, key=len, reverse=True)
KNOWN_SKILLS = sorted({sk for m in _all_meta for sk in m.get("skills_norm", "").split(",") if sk}, key=len, reverse=True)

# Skills are matched as whole phrases first, but many terms ("ai", "llm") only
# ever appear *inside* a longer skill ("generative AI") and would otherwise
# never be recognized. So we also index individual tokens, weighted by rarity
# (idf = log(N / doc_freq)) so a rare term like "ai" counts for more than a
# filler word like "systems" or "management".
_STOPWORDS = {"and", "or", "of", "for", "the", "with", "in", "on", "to", "a", "an", "etc", "via", "using"}
_TOKEN_RE = re.compile(r"[a-z0-9+#.]+")
_N = len(_all_meta)
_phrase_df = Counter(sk for m in _all_meta for sk in set(m.get("skills_norm", "").split(",")) if sk)
_token_df = Counter(t for m in _all_meta
                    for t in set(_TOKEN_RE.findall(m.get("skills_norm", ""))) - _STOPWORDS if len(t) >= 2)
SKILL_IDF = {sk: math.log(_N / c) for sk, c in _phrase_df.items()}
SKILL_TOKEN_IDF = {t: math.log(_N / c) for t, c in _token_df.items()}

SENIORITY_RE = re.compile(r"^(?:(?:senior|sr\.?|junior|jr\.?|lead|principal|staff|associate|entry[- ]level|mid[- ]level|new grad)\s+)+")
def core_title(title: str) -> str:
    """'Senior Data Scientist, Core Data - PhD' -> 'data scientist'."""
    t = title.lower()
    t = re.split(r"[,(/|]| - | // ", t)[0]          # drop team / level suffixes
    t = SENIORITY_RE.sub("", t.strip())
    t = re.sub(r"\s+(?:i{1,3}|iv|v|\d)$", "", t)     # trailing level markers
    return t.strip()

# keep 2+ word phrases only: messy titles like "Senior/Staff ML Engineer, ..." or
# "AI/NLP Engineer" otherwise reduce to junk cores such as "senior" or "ai"
KNOWN_TITLES = sorted({t for t in (core_title(m["title_norm"]) for m in _all_meta if m.get("title_norm"))
                       if len(t.split()) >= 2}, key=len, reverse=True)

REMOTE_RE = re.compile(r"\b(?:fully\s+|100%\s+)?(?:remote|work[\s-]from[\s-]home|wfh)(?:\s+only)?\b", re.I)


def parse_query(query: str):
    """Split a query into (semantic_text, where_filter, filters).

    A single embedding can't treat "in Denver" or "at Anthropic" as a hard
    requirement -- it's one word out of ~200 -- so those need to become exact
    Chroma metadata filters instead. Recognized constraints:
      * "remote" / "work from home" / "wfh"        -> is_remote == True
      * any known city or state name, e.g. "in Denver", "in Colorado"
      * "at <known company>", e.g. "at Anthropic"  -> company_norm == "anthropic"
    Several locations / companies ("in Denver or Utah", "at Anthropic or Figma")
    are OR-ed together. Matched phrases are stripped so the leftover text is
    what actually gets embedded for the semantic search.
    """
    q = f" {query.lower()} "
    filters = {}

    if REMOTE_RE.search(q):
        filters["is_remote"] = True
        q = REMOTE_RE.sub(" ", q)

    # Locations can appear anywhere: no known city/state name is an ordinary
    # English word, so a bare match is safe. (Two-letter codes are NOT matched
    # -- "in", "or", "me", "co" are words.) Collect every one mentioned.
    locations = []
    for term in KNOWN_LOCATION_TERMS:
        pat = re.compile(rf"(?<!\w)(?:(?:in|near|around|from|based in|or|and|,)\s*)?{re.escape(term)}(?!\w)")
        if pat.search(q):
            locations.append(KNOWN_LOCATIONS[term])
            q = pat.sub(" ", q, count=1)
    if locations:
        filters["location"] = sorted(set(locations))       # [("city","denver"), ("state","ut")]

    # Companies must follow a preposition: the company list contains ordinary
    # words like "close", "loop", "branch", "ramp", "hive", so a bare match
    # would misfire on "close to Boston" or "feedback loop". Once one company
    # has been found, "or"/"and"/"," also count so "at Anthropic or Figma"
    # picks up both. Two passes because the vocabulary is scanned longest-first,
    # and an "or <company>" may be checked before the "at <company>" that enables it.
    companies = []
    for _ in range(2):
        for comp in KNOWN_COMPANIES:
            if comp in companies:
                continue
            lead = r"(?:at|with|for)" if not companies else r"(?:at|with|for|or|and|,)"
            pat = re.compile(rf"(?<!\w){lead}\s*{re.escape(comp)}(?!\w)")
            if pat.search(q):
                companies.append(comp)
                q = pat.sub(" ", q, count=1)
    # A query that is nothing but company names ("anthropic", "figma or plaid")
    # is treated as a company filter: there's no collision risk without
    # surrounding text, and a bare proper noun has ~no semantic content for the
    # vector search to work with anyway.
    if not companies:
        parts = [p for p in re.split(r"\s*(?:,|/|\bor\b|\band\b)\s*", q.strip()) if p]
        if parts and all(p in KNOWN_COMPANIES for p in parts):
            companies = parts
            q = " "
    if companies:
        filters["company_norm"] = companies[0] if len(companies) == 1 else companies

    semantic = re.sub(r"\s+", " ", q).strip()
    if not semantic:              # e.g. the query was just "remote jobs"
        semantic = query

    def clause(k, v):             # list -> OR within the field
        if k == "location":       # mixed city/state -> OR across the two fields
            by_field = {}
            for field, value in v:
                by_field.setdefault(field, []).append(value)
            parts = [clause(f, vals[0] if len(vals) == 1 else vals) for f, vals in by_field.items()]
            return parts[0] if len(parts) == 1 else {"$or": parts}
        return {k: {"$in": v}} if isinstance(v, list) else {k: {"$eq": v}}

    if not filters:
        where = None
    elif len(filters) == 1:
        where = clause(*next(iter(filters.items())))
    else:                         # Chroma needs an explicit $and for 2+ conditions
        where = {"$and": [clause(k, v) for k, v in filters.items()]}
    return semantic, where, filters


# --- Field-boosted re-ranking ------------------------------------------------
# The embedding stays the ONLY retriever: it fetches the top N_CANDIDATES by
# cosine similarity (inside any filters from parse_query). We then nudge the
# order with exact-term evidence from the structured columns -- skills, title,
# company -- because those are exactly where embeddings are weakest (proper
# nouns have no meaning to match on; a "Data Scientist" and "Data Engineer"
# title look nearly identical in vector space). Boosts are soft, so a false
# positive costs a small nudge rather than excluding a job the way a filter
# would. Retrieval is over chunks, so results are collapsed to each job's
# single best-matching chunk before boosting.
N_CANDIDATES = 40
W_SKILL, W_TITLE, W_COMPANY = 0.10, 0.10, 0.15

# Small add-on: a match-quality label instead of a hard score cutoff, so a
# thin result pool (e.g. after a narrow filter) shows a labeled weak match
# instead of quietly looking like a strong one. Cosine ranges are model-
# specific, so the bands are per model, measured against eval scores
# (MiniLM: off-topic queries never score above 0.42; mpnet: up to 0.46).
# A new model needs its own row here or the defaults will mislabel.
MATCH_BANDS = {                       # model -> (strong, moderate)
    "all-MiniLM-L6-v2": (0.50, 0.40),
    "all-mpnet-base-v2": (0.55, 0.45),
}
STRONG_MATCH, MODERATE_MATCH = MATCH_BANDS.get(EMBED_MODEL, MATCH_BANDS["all-MiniLM-L6-v2"])

def match_label(score: float) -> str:
    if score >= STRONG_MATCH:
        return "🟢 Strong match"
    if score >= MODERATE_MATCH:
        return "🟡 Moderate match"
    return "🔴 Weak match"


def find_terms(query: str):
    """Return the known skills / title phrases / companies mentioned in the query.

    Skills: whole Skills phrases first ("machine learning"), then any leftover
    query tokens that occur inside some job's skills ("ai", "llm"). Each skill
    term carries an IDF weight in skill_weights. A query word counts as either
    a title/company phrase OR a skill term, never both, so "data scientist"
    doesn't also register the skill token "data".
    """
    q = f" {query.lower()} "
    def hits(vocab, max_hits=None):
        found = []
        for term in vocab:
            if re.search(rf"(?<!\w){re.escape(term)}(?!\w)", q):
                found.append(term)
                if max_hits and len(found) >= max_hits:
                    break
        return found

    titles = hits(KNOWN_TITLES, max_hits=1)      # longest match only
    companies = hits(KNOWN_COMPANIES)

    rest = q
    for phrase in titles + companies:
        rest = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", rest)
    skill_weights = {}
    for phrase in KNOWN_SKILLS:
        if re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", rest):
            skill_weights[phrase] = SKILL_IDF[phrase]
            rest = re.sub(rf"(?<!\w){re.escape(phrase)}(?!\w)", " ", rest)   # don't re-count its tokens
    for tok in _TOKEN_RE.findall(rest):
        if tok in SKILL_TOKEN_IDF and tok not in skill_weights:
            skill_weights[tok] = SKILL_TOKEN_IDF[tok]

    return {
        "skills": list(skill_weights),
        "skill_weights": skill_weights,
        "titles": titles,
        "companies": companies,
    }


def boosted_search(semantic: str, where, n_results: int = 5):
    """Vector search, then re-rank candidates with field boosts."""
    terms = find_terms(semantic)
    n_chunks = int(N_CANDIDATES * CHUNKS_PER_JOB) + 5
    cands = {}                       # job_id -> (best chunk text, meta, distance)

    def collect(res):
        for d, m, dist in zip(res["documents"][0], res["metadatas"][0], res["distances"][0]):
            jid = m["job_id"]
            if jid not in cands or dist < cands[jid][2]:     # keep the best-matching chunk
                cands[jid] = (d, m, dist)

    collect(collection.query(query_texts=[semantic], n_results=n_chunks, where=where))

    # Recall guard: make sure a named company's jobs are in the candidate pool
    # at all. With small chunks a bare proper noun's cosine is ~0 against
    # everything, so a +0.15 boost can't reliably out-rank noise unless the
    # job is fetched here first.
    for comp in terms["companies"]:
        comp_where = {"company_norm": {"$eq": comp}} if where is None else {"$and": [where, {"company_norm": {"$eq": comp}}]}
        collect(collection.query(query_texts=[semantic], n_results=n_chunks, where=comp_where))

    ranked = []
    for d, m, dist in cands.values():
        cos = 1 - dist                       # collection is in cosine space
        job_skills = m.get("skills_norm", "")
        # word-boundary match so "spark" hits "apache spark" but "sql" doesn't hit
        # "mysql"; optional trailing "s" so "llm" hits "llms"
        matched_skills = [sk for sk in terms["skills"]
                          if re.search(rf"(?<!\w){re.escape(sk)}s?(?!\w)", job_skills)]
        boosts = {}
        if matched_skills:
            w = terms["skill_weights"]
            boosts["skills"] = W_SKILL * sum(w[sk] for sk in matched_skills) / sum(w.values())
        if terms["titles"] and terms["titles"][0] in m.get("title_norm", ""):
            boosts["title"] = W_TITLE
        if m.get("company_norm") in terms["companies"]:
            boosts["company"] = W_COMPANY
        ranked.append((cos + sum(boosts.values()), cos, boosts, matched_skills, d, m))
    ranked.sort(key=lambda r: r[0], reverse=True)
    return ranked[:n_results], terms


# CLI loop
print("\n\n🔍 Semantic Job Search CLI\n")
print("Type your query to find matching jobs. Type 'exit' to quit.")
print("Tip: mention a city or state, a company ('at Figma'), or 'remote' to filter.\n")

while True:
    query = input("Enter your job-related query: ").strip()
    if query.lower() in ["exit", "quit", "stop"]:
        print("👋 Exiting. Thanks for searching!")
        break

    try:
        semantic, where, filters = parse_query(query)
        query_embedding = sbert_ef([semantic])[0]
        ranked, terms = boosted_search(semantic, where, n_results=5)
        boost_terms = {k: v for k, v in terms.items() if v and k != "skill_weights"}

        if filters:
            print(f"\n🔎 Filters detected: {filters}")
        if boost_terms:
            print(f"⚡ Boost terms: {boost_terms}")
        if not ranked:
            print("\nNo jobs match those filters. Try loosening them.\n")
            continue

        print(f"\n🧠 Semantic query: '{semantic}'")
        print(f"Query embedding: {len(query_embedding)} dimensions")
        print("First 10 values:", np.round(query_embedding[:10], 3))

        print(f"\nTop Matches ({len(ranked)}):\n")
        if ranked[0][0] < MODERATE_MATCH:
            print("ℹ️  Nothing matched strongly — these are the closest postings we have; try different wording.\n")
        for score, cos, boosts, matched_skills, doc, meta in ranked:
            print(match_label(score))
            print(f"🏷️  Title: {meta.get('Job title')}")
            print(f"🏢 Company: {meta.get('Company')}")
            print(f"📍 Location: {meta.get('Location')}")
            print(f"🛠️  Skills: {meta.get('Skills')}")
            print(f"🔗 URL: {meta.get('URL')}")
            print("")
            print(f"📄 {doc[:250]}")
            if boosts:
                boost_str = ", ".join(f"{k}=+{v:.2f}" for k, v in boosts.items())
                if matched_skills:
                    boost_str += f"  (skills matched: {', '.join(matched_skills)})"
                print(f"🥇 Score: {score:.2f}  = cosine {cos:.2f} + boosts [{boost_str}]\n")
            else:
                print(f"🥇 Score: {score:.2f}  (cosine similarity)\n")
            print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
            print("")
    except Exception as e:
        print(f"⚠️ Error during query: {e}\n")