"""Shared RAG core for both music chatbots.

Improvements 1-5 were built in gradioMusicChatbot.py and Improvement 6 in
musicChatbot.py, so the two chatbots had different features, different prompts
and two ways of putting metadata in front of the model. Improvement 7 shared the
filter functions; everything else was still duplicated. This module holds the one
copy of each piece, and the two front-ends only choose configuration:

    musicChatbot.py        mistral 0.7, plain similarity search, k=10
    gradioMusicChatbot.py  phi3 0.4,    MMR, k=4 from fetch_k=20

Those differences are deliberate - Step 4 of the lab asks us to compare models
and top-k - so they are parameters here, not copied code.

Nothing in this module imports either front-end, so there is no import cycle.
"""

import json
import os
import re
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

# The same model embedMusicChunks.py used to build the index. Chroma stores that
# model in the collection config, so plain similarity search worked without
# naming it here - but MMR has to embed the query in Python to compare
# candidates with each other, and raises ValueError without this.
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

vectorstore = Chroma(
    collection_name="musicReviews",
    persist_directory=chroma_path,
    embedding_function=embeddings,
)

# One mistral at temperature 0 for the two structured jobs (pulling filters out
# of a question, and rewriting a follow-up). format="json" is set per call
# because only the filter job wants JSON back.
utility_llm = ChatOllama(model="mistral", temperature=0)
filter_llm = ChatOllama(model="mistral", temperature=0, format="json")


# --- Metadata filters (year / genre / score) -------------------------------
# Plain vector search only compares meaning, so "jazz albums from 2019" ignores
# the year, genre and score stored on every chunk. These functions ask the LLM
# for those constraints, then apply them as a Chroma `where` filter.

def _load_genre_labels(page_size=5000):
    """Read the genre label of every chunk. Chroma errors ("too many SQL
    variables") if we ask for all ~50k chunks at once, so read in pages."""
    labels, offset = set(), 0
    while True:
        page = vectorstore.get(include=["metadatas"], limit=page_size, offset=offset)
        labels.update(m["genre"] for m in page["metadatas"] if m.get("genre"))
        if len(page["ids"]) < page_size:
            return sorted(labels)
        offset += page_size


ALL_GENRE_LABELS = _load_genre_labels()
BASE_GENRES = sorted({part for label in ALL_GENRE_LABELS for part in label.split(" / ")})

FILTER_PROMPT = """You turn a music question into search filters for a database of Pitchfork album reviews.
Return only a JSON object with exactly these keys (use null when the question does not say):
  "genre": one of {genres}, or null
  "year_min": earliest album year as an integer, or null
  "year_max": latest album year as an integer, or null
  "min_score": lowest Pitchfork score (0-10) as a number, or null

Rules:
- A single year like "from 2019" means year_min = 2019 and year_max = 2019.
- A decade like "the 90s" means year_min = 1990 and year_max = 1999.
- Map similar words onto the genre list: hip hop -> Rap, R&B or soul -> Pop/R&B, country or folk -> Folk/Country.
- Only set min_score for an explicit number, or for "highest rated" / "top rated" (use 9.0).
- Words like "best", "good", or "acclaimed" on their own are not filters.
- If the question does not mention a genre, year, or score, every value is null.

Question: {question}
JSON:"""

# The LLM sometimes "helps" by guessing filters from what it knows, e.g. it
# turns "Tell me about Kid A" into Rock + year 2000, which hides the 2009
# reissue review. Asking it not to in the prompt did not work, so the code
# below only keeps a filter when the question contains words that justify it.
GENRE_WORDS = {
    "Rock": ["rock"],
    "Rap": ["rap", "hip hop", "hip-hop"],
    "Pop/R&B": ["pop", "r&b", "rnb", "soul"],
    "Folk/Country": ["folk", "country"],
    "Jazz": ["jazz"],
    "Electronic": ["electronic", "techno", "edm"],
    "Experimental": ["experimental"],
    "Metal": ["metal"],
    "Global": ["global", "world music"],
}
SCORE_WORDS = ["rated", "rating", "score", "perfect"]


def _mentions(question, words):
    """True if any of the words appears in the question as a whole word."""
    return any(re.search(rf"\b{re.escape(word)}\b", question.lower()) for word in words)


def _as_number(value):
    """Return value as a float, or None. The LLM sometimes answers "2019" as a string."""
    if isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_filters(question):
    """Ask the LLM for filters and keep only values that make sense.

    Returns a dict like {"genre": "Jazz", "year_min": 2019.0, ...} holding only
    the valid keys. Bad JSON or out-of-range values are dropped, so the worst
    case is an empty dict, which means the original unfiltered search.
    """
    try:
        response = filter_llm.invoke(
            FILTER_PROMPT.format(genres=", ".join(BASE_GENRES), question=question)
        )
        raw = json.loads(response.content)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(raw, dict):
        return {}

    filters = {}
    genre = raw.get("genre")
    if genre in BASE_GENRES and _mentions(question, GENRE_WORDS.get(genre, [genre.lower()])):
        filters["genre"] = genre
    # A year or decade ("2019", "90s") always involves a digit.
    if re.search(r"\d", question):
        for key in ("year_min", "year_max"):
            year = _as_number(raw.get(key))
            if year is not None and 1900 <= year <= 2100:
                filters[key] = year
    score = _as_number(raw.get("min_score"))
    if score is not None and 0 < score <= 10 and _mentions(question, SCORE_WORDS):
        filters["min_score"] = score
    return filters


def build_where(filters):
    """Turn the parsed filters into a Chroma `where` clause (None = no filter)."""
    conditions = []
    if "genre" in filters:
        matching_labels = [
            label for label in ALL_GENRE_LABELS if filters["genre"] in label.split(" / ")
        ]
        conditions.append({"genre": {"$in": matching_labels}})
    if "year_min" in filters:
        conditions.append({"year": {"$gte": filters["year_min"]}})
    if "year_max" in filters:
        conditions.append({"year": {"$lte": filters["year_max"]}})
    if "min_score" in filters:
        conditions.append({"score": {"$gte": filters["min_score"]}})

    if not conditions:
        return None
    # Chroma requires "$and" to combine two or more conditions.
    return conditions[0] if len(conditions) == 1 else {"$and": conditions}


def resolve_filters(question):
    """parse_filters + build_where, dropping a filter that matches no review.

    An impossible request ("albums from 2030") would otherwise search an empty
    set, so we fall back to an unfiltered search and let the model say there are
    none.
    """
    filters = parse_filters(question)
    where = build_where(filters)
    if where and not vectorstore.get(where=where, limit=1)["ids"]:
        return {}, None
    return filters, where


# --- Follow-up questions ---------------------------------------------------

CONDENSE_PROMPT = """Rewrite the follow-up question as a standalone question that keeps every detail it relies on from the conversation.
Keep any genre, year or rating the user asked about earlier. Do not answer it. Reply with the question only.

Conversation:
{history}

Follow-up question: {question}
Standalone question:"""


def condense_question(question, history):
    """Rewrite a follow-up so the filters and the search see the full request.

    parse_filters reads one question at a time, so "Which of those is the
    highest rated?" used to extract {'min_score': 9.0} and lose the genre from
    the previous turn: the retrieved chunks no longer matched the conversation
    and the model answered from history while citing unrelated reviews.
    Returns the question unchanged when there is no history.
    """
    if not history:
        return question
    try:
        response = utility_llm.invoke(
            CONDENSE_PROMPT.format(history=format_chat_history(history), question=question)
        )
        rewritten = (response.content or "").strip().strip('"')
    except Exception:
        return question
    # A small local model sometimes answers instead of rewriting, or returns
    # nothing useful. Fall back to the original rather than searching on junk.
    if not rewritten or len(rewritten) > 300:
        return question
    return rewritten


# --- Prompting ------------------------------------------------------------

QA_PROMPT = PromptTemplate(
    template="""You are a knowledgeable music recommendation assistant with expertise in album reviews and music analysis.
Your role is to help users discover music based on their preferences and provide insightful recommendations.

Use the following context from music reviews and album information to answer the user's question.
If you don't know the answer based on the context, say so honestly - don't make up information.

When recommending music:
- Consider the mood, genre, and style preferences
- Explain why you're making specific recommendations
- Reference specific albums, artists, or tracks when relevant
- Identify the supporting artist and review title using the provided [Source N] labels
- Be enthusiastic but honest about the music

If you encounter explicit terms in the names of artists, albums, or song titles, blur them out with the
use of asterisks so that the user does not see the full explicit word.

Context from reviews (each source includes its artist, year, genre and score):
{context}

Chat History:
{chat_history}

User question: {question}

Please provide a helpful response based on the music reviews and context available. Keep your answer grounded in the provided sources and cite supporting sources as [Source N].""",
    input_variables=["context", "chat_history", "question"],
)


def _shown(metadata, key):
    """Metadata value for the prompt, or "unknown".

    Missing fields are stored as "" by the current embedding script and left out
    entirely by the original one, and years arrive as floats (2023.0).
    """
    value = metadata.get(key)
    if value in (None, ""):
        return "unknown"
    return int(value) if key == "year" and isinstance(value, float) else value


def _title_of(metadata):
    """Album title. Improvement 1 renamed `album` to `title`; indexes built
    before that still store `album`, so accept either."""
    return metadata.get("title", metadata.get("album", "Unknown title"))


def format_context(documents):
    """Write each chunk into the prompt with the facts the user can ask about.

    The chain's default document template is just '{page_content}', so without
    this the model saw raw prose and could not tell which album, year or score a
    chunk belonged to - even when the filter had picked exactly the right ones.
    """
    formatted = []
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        formatted.append(
            f"[Source {source_number}]\n"
            f"Artist: {metadata.get('artist', 'Unknown artist')}\n"
            f"Review title: {_title_of(metadata)}\n"
            f"Year: {_shown(metadata, 'year')}\n"
            f"Genre: {_shown(metadata, 'genre')}\n"
            f"Score: {_shown(metadata, 'score')}\n"
            f"Review excerpt: {document.page_content}"
        )
    return "\n\n".join(formatted)


def format_sources(documents):
    """List one entry per source review, naming every [Source N] it covers.

    Two chunks often come from the same review. Skipping the duplicate would
    drop its number, so a model citing [Source 3] could find no [Source 3] line
    to check it against. The numbers are collected per review instead, giving
    "[Sources 2, 3]".

    Reviews are grouped by (artist, title) rather than by `source_id`, because
    `source_id` only exists on indexes built after Improvement 1 - on an older
    collection every chunk reports None and nothing would group.
    """
    grouped = {}
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        key = (metadata.get("artist", "Unknown artist"), _title_of(metadata))
        entry = grouped.setdefault(key, {"numbers": [], "source_id": metadata.get("source_id")})
        entry["numbers"].append(source_number)

    source_lines = []
    for (artist, title), entry in grouped.items():
        numbers = ", ".join(str(number) for number in entry["numbers"])
        label = f"Sources {numbers}" if len(entry["numbers"]) > 1 else f"Source {numbers}"
        # Only show the review id when the index actually stored one.
        suffix = f" ({entry['source_id']})" if entry["source_id"] else ""
        source_lines.append(f"- [{label}] {artist} — {title}{suffix}")

    if not source_lines:
        return ""
    return "\n\nSupporting Pitchfork reviews:\n" + "\n".join(source_lines)


def format_chat_history(history):
    """Turn a list of {"role", "content"} messages into prompt text.

    Both front-ends keep history per conversation and pass it in; nothing here
    is stored at module level, so two Gradio visitors cannot see each other's
    turns.
    """
    turns = []
    for message in history or []:
        role = "User" if message.get("role") == "user" else "Assistant"
        turns.append(f"{role}: {message.get('content', '')}")
    return "\n".join(turns) or "(No previous conversation.)"


# --- HyDE and keyword search (Hector's Improvements 4 and 5) ----------------
# Both are off by default. They change *which* chunks come back, and the
# confidence gate's 0.35 threshold was swept against scores for the raw
# question (Improvement 8), so turning either on invalidates that sweep until
# it is re-run. Enable them with HYDE_ENABLED=1 / BM25_ENABLED=1 to measure.

# HyDE: a question like "something for a rainy day" shares almost no words with
# Pitchfork review prose, so its embedding lands nearer whatever chunks are
# generically popular than mood-matched ones. Ask the model to write the review
# excerpt that would answer the question and search with that instead, so the
# search text is already in the corpus's vocabulary.
HYDE_PROMPT = PromptTemplate(
    template="""Write a 2-3 sentence excerpt from a Pitchfork music review that would
directly satisfy the following request. Use the same vocabulary and tone as a real
music review. Write only the excerpt, no preamble or explanation.

Request: {query}

Review excerpt:""",
    input_variables=["query"],
)


def generate_hypothetical_document(question):
    """A made-up review excerpt to search with, or "" if the model call fails.

    Uses utility_llm (temperature 0) rather than the chat model: a different
    excerpt on every run would retrieve different chunks for the same question
    and make a before/after eval unreadable. "" means the caller falls back to
    the real question, so a failure degrades to ordinary search.
    """
    try:
        return utility_llm.invoke(HYDE_PROMPT.format(query=question)).content.strip()
    except Exception:
        return ""


# BM25 is the counterweight to HyDE. Hector's own eval showed HyDE makes artist
# queries worse - the generated excerpt describes a sound in general terms, so
# albums sharing that vocabulary outrank the artist actually named. BM25 scores
# exact word overlap, so "Radiohead" still boosts chunks containing
# "Radiohead" whatever the excerpt said.
_bm25 = None


def _build_bm25():
    """Read every chunk out of Chroma and build the keyword index.

    Returns (index, documents), or None when rank_bm25 is not installed - a
    missing optional dependency turns keyword search off rather than breaking
    both chatbots.
    """
    try:
        from rank_bm25 import BM25Okapi
    except ImportError:
        print("⚠️  rank_bm25 not installed; keyword search off "
              "(pip install rank-bm25). Using dense search only.")
        return None

    print("Building BM25 keyword index (first query only)...", flush=True)
    texts, metadatas, offset, page_size = [], [], 0, 5000
    # Same paging as _load_genre_labels: one big .get() trips Chroma's SQL
    # variable limit.
    while True:
        page = vectorstore.get(include=["documents", "metadatas"],
                               limit=page_size, offset=offset)
        texts.extend(page["documents"])
        metadatas.extend(page["metadatas"])
        if len(page["ids"]) < page_size:
            break
        offset += page_size

    index = BM25Okapi([text.lower().split() for text in texts])
    documents = [Document(page_content=t, metadata=m) for t, m in zip(texts, metadatas)]
    print(f"BM25 index built over {len(documents)} chunks.", flush=True)
    return index, documents


def _bm25_search(question, k):
    """Top k chunks by exact word overlap. [] when keyword search is unavailable.

    The index is built on first use, not at import: evalMetadataFilters.py and
    both front-ends import this module, and none should pay for an index it
    never queries.
    """
    global _bm25
    if _bm25 is None:
        _bm25 = _build_bm25() or False   # False = tried once, unavailable
    if not _bm25:
        return []
    index, documents = _bm25
    scores = index.get_scores(question.lower().split())
    return [documents[i] for i in scores.argsort()[-k:][::-1]]


def _reciprocal_rank_fusion(dense_documents, keyword_documents, rrf_k=60):
    """Merge two ranked lists by Reciprocal Rank Fusion.

    Each document scores sum(1 / (rank + rrf_k)) over the lists it appears in,
    so a chunk both searches liked outranks one only a single search liked.
    Ranks are used rather than raw scores because cosine distance and BM25
    relevance are not on the same scale and cannot be averaged.

    Documents are keyed on (source_id, page_content), the same key the
    confidence gate uses. Hector's version keyed on the first 80 characters,
    which collapses two different chunks that happen to open the same way.
    """
    scores, documents_by_key = {}, {}
    for ranked in (dense_documents, keyword_documents):
        for rank, document in enumerate(ranked):
            key = (document.metadata.get("source_id"), document.page_content)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rrf_k)
            documents_by_key[key] = document
    return sorted(
        documents_by_key.values(),
        key=lambda d: scores[(d.metadata.get("source_id"), d.page_content)],
        reverse=True,
    )


# --- Retrieval and answering ----------------------------------------------

DEFAULT_CONFIDENCE_THRESHOLD = 0.35
REFUSAL_MESSAGE = (
    "I don't have enough relevant evidence in the Pitchfork reviews to answer "
    "that reliably. Please try a more specific music question."
)


def make_retriever(search_type="similarity", **search_kwargs):
    """Retriever for one front-end's configuration.

    search_type="similarity" with k=10 is the CLI; "mmr" with k=4 and
    fetch_k=20 is the Gradio app.
    """
    return vectorstore.as_retriever(search_type=search_type, search_kwargs=dict(search_kwargs))


def retrieve(question, retriever, where=None, search_text=None, use_bm25=False):
    """Retrieve for one question and attach a relevance score to each chunk.

    MMR returns documents without scores, so a second lookup supplies them; the
    documents handed to the model are still exactly the retriever's results.

    The filter goes to BOTH lookups. Filtering only the search would leave the
    score lookup covering all 50k chunks, so the selected documents would be
    missing from score_by_key, 0.0 would win, and the confidence gate would
    refuse every filtered question.

    `search_text` replaces the question for the vector search only (that is how
    HyDE is applied); the model still answers the real question. Scoring uses
    the same text that did the retrieving, so the gate stays internally
    consistent - but the threshold swept for raw questions no longer applies.

    `use_bm25` fuses a keyword leg into the results. It is skipped whenever a
    filter is active: BM25 cannot see the Chroma `where` clause, so it would
    smuggle back exactly the chunks the filter just excluded.
    """
    query_text = search_text or question
    if where:
        retriever.search_kwargs["filter"] = where
    else:
        retriever.search_kwargs.pop("filter", None)

    documents = retriever.invoke(query_text)

    if use_bm25 and not where:
        # The dense leg keeps whatever search_type the front-end chose, so it
        # returns k rather than Hector's k*2; the keyword leg supplies k*2 so
        # fusion still has something to promote.
        k = retriever.search_kwargs.get("k", 10)
        keyword_documents = _bm25_search(question, k * 2)
        if keyword_documents:
            documents = _reciprocal_rank_fusion(documents, keyword_documents)[:k]

    if not documents:
        return [], 0.0

    scored = vectorstore.similarity_search_with_relevance_scores(query_text, k=20, filter=where)
    score_by_key = {(d.metadata.get("source_id"), d.page_content): float(s) for d, s in scored}

    for document in documents:
        key = (document.metadata.get("source_id"), document.page_content)
        document.metadata = {
            **document.metadata,
            "retrieval_confidence": score_by_key.get(key, 0.0),
        }
    best = max(d.metadata["retrieval_confidence"] for d in documents)
    return documents, best


def answer(question, history, llm, retriever, threshold=DEFAULT_CONFIDENCE_THRESHOLD,
           use_hyde=False, use_bm25=False):
    """One question end to end. Returns a dict so callers can show the details.

    Order matters: the follow-up is rewritten first, so the filters and the
    search both see the full request rather than the bare follow-up.

    use_hyde/use_bm25 default to off - see the HyDE section above for why.
    """
    search_question = condense_question(question, history)
    filters, where = resolve_filters(search_question)

    # HyDE is skipped on filtered questions: the filter has already narrowed the
    # search space to what was asked for, which is what HyDE is for.
    hyde_text = ""
    if use_hyde and not where:
        hyde_text = generate_hypothetical_document(search_question)

    documents, confidence = retrieve(search_question, retriever, where,
                                     search_text=hyde_text or None,
                                     use_bm25=use_bm25)

    if confidence < threshold:
        return {
            "answer": REFUSAL_MESSAGE,
            "documents": documents,
            "filters": filters,
            "confidence": confidence,
            "search_question": search_question,
            "hyde_text": hyde_text,
            "refused": True,
        }

    response = llm.invoke(QA_PROMPT.format(
        context=format_context(documents),
        chat_history=format_chat_history(history),
        question=question,
    ))
    text = response.content if hasattr(response, "content") else str(response)
    return {
        "answer": text,
        "documents": documents,
        "filters": filters,
        "confidence": confidence,
        "search_question": search_question,
        "hyde_text": hyde_text,
        "refused": False,
    }
