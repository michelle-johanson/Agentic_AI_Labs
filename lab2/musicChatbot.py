from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.vectorstores import VectorStoreRetriever
from pathlib import Path
from dotenv import load_dotenv
import json
import os
import re
load_dotenv() 

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

llm = ChatOllama(model="mistral", temperature=0.7)

# Second copy of mistral used only to pull search filters out of the question.
# temperature=0 keeps it consistent, and format="json" makes Ollama return JSON.
filter_llm = ChatOllama(model="mistral", temperature=0, format="json")

# Third copy of mistral, used only to write the HyDE excerpt (see below).
# Kept separate from `llm` because the chat model runs at temperature=0.7, and a
# different excerpt on every run means the same question retrieves different
# chunks each time - which makes a before/after eval unreadable.
hyde_llm = ChatOllama(model="mistral", temperature=0)

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

# --- HyDE: Hypothetical Document Embeddings --------------------------------
# A question like "something for a rainy day" shares almost no words with
# Pitchfork review prose, so its embedding lands nearer to whatever chunks are
# generically popular than to mood-matched ones. HyDE closes that vocabulary
# gap: ask the model to write the review excerpt it thinks would answer the
# question, then run the vector search with that excerpt instead of the
# question. The chunks we want are written in the same register as the excerpt,
# so they sit closer to it in vector space.

# Set HYDE_ENABLED=0 to turn HyDE off without editing code, so the same script
# can produce both arms of a before/after eval.
HYDE_ENABLED = os.getenv("HYDE_ENABLED", "1").lower() not in ("0", "false", "no")

HYDE_PROMPT = PromptTemplate(
    template="""Write a 2-3 sentence excerpt from a Pitchfork music review that would
directly satisfy the following request. Use the same vocabulary and tone as a real
music review. Write only the excerpt, no preamble or explanation.

Request: {query}

Review excerpt:""",
    input_variables=["query"],
)


def generate_hypothetical_document(query):
    """Return a made-up review excerpt to search with, or the question itself.

    Any failure (Ollama down, an empty answer) falls back to the original
    question, so a broken HyDE call degrades to the previous behaviour rather
    than searching with nothing.
    """
    try:
        excerpt = hyde_llm.invoke(HYDE_PROMPT.format(query=query)).content.strip()
    except Exception:
        return query
    return excerpt or query


class LabeledRetriever(VectorStoreRetriever):
    """Normal retriever that tidies metadata so every chunk can be labeled.

    The chain labels each excerpt with artist/album/year/genre/score (see
    document_prompt below) and raises an error if any chunk is missing one of
    those fields. ~3,300 chunks have no genre and ~600 have no year, so fill in
    "unknown", and show years as 2023 instead of 2023.0. The original embedding
    script leaves missing fields out; the newer one stores "", so handle both.

    It also applies HyDE. Unless a metadata filter is active, the question is
    swapped for a generated review excerpt for the vector search only - the
    model still answers the user's real question, and `last_hypothetical` keeps
    the excerpt around so the CLI can show what was actually searched for.
    """

    # Declared as a field because VectorStoreRetriever is a pydantic model and
    # will not accept an attribute that was never declared.
    last_hypothetical: str = ""

    def _get_relevant_documents(self, query, *, run_manager, **kwargs):
        # Hector gated HyDE on "does the question name a genre". Improvement 6
        # already answers a wider version of that question, so reuse its result
        # instead of adding a second genre detector: when parse_filters found a
        # year/genre/score constraint, the main loop has put a `filter` in
        # search_kwargs and the search space is already narrow.
        if HYDE_ENABLED and not self.search_kwargs.get("filter"):
            self.last_hypothetical = generate_hypothetical_document(query)
            search_query = self.last_hypothetical
        else:
            self.last_hypothetical = ""
            search_query = query
        docs = super()._get_relevant_documents(search_query, run_manager=run_manager, **kwargs)
        for doc in docs:
            metadata = dict(doc.metadata)
            for key in ("year", "genre"):
                if metadata.get(key) in (None, ""):
                    metadata[key] = "unknown"
            if isinstance(metadata["year"], float):
                metadata["year"] = int(metadata["year"])
            doc.metadata = metadata
        return docs


retriever = LabeledRetriever(vectorstore=vectorstore, search_kwargs={"k": 10})

# --- Metadata filters (year / genre / score) -------------------------------
# Plain vector search only compares meaning, so "jazz albums from 2019" ignores
# the year, genre, and score stored on every chunk. The functions below ask the
# LLM to extract those constraints, then apply them as a Chroma `where` filter.

# Genre labels in the data are sometimes combined, e.g. "Experimental / Jazz".
# Chroma can only match a whole metadata value, so we read every label that
# exists and later expand "Jazz" into all the labels that contain it.
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

qa_prompt = PromptTemplate(
    template="""You are a knowledgeable music recommendation assistant with expertise in album reviews and music analysis. 
Your role is to help users discover music based on their preferences and provide insightful recommendations.

Use the following context from music reviews and album information to answer the user's question.
If you don't know the answer based on the context, say so honestly - don't make up information.

When recommending music:
- Consider the mood, genre, and style preferences
- Explain why you're making specific recommendations
- Reference specific albums, artists, or tracks when relevant
- Be enthusiastic but honest about the music

If you encounter explicit terms in the names of artists, albums, or song titles, blur them out with the
use of asterisks so that the user does not see the full explicit word.

Context from reviews:
{context}

Chat History:
{chat_history}

User question: {question}

Please provide a helpful response based on the music reviews and context available:""",
    input_variables=["context", "chat_history", "question"]
)

# How each retrieved chunk is written into {context}. By default the chain only
# passes the raw excerpt text, so the model could not tell which album, year,
# or score a chunk belonged to, even when the filter had picked the right ones.
document_prompt = PromptTemplate(
    template="[{artist} - {album} | year: {year} | genre: {genre} | Pitchfork score: {score}]\n{page_content}",
    input_variables=["page_content", "artist", "album", "year", "genre", "score"],
)

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # optional, for debugging
    combine_docs_chain_kwargs={"prompt": qa_prompt, "document_prompt": document_prompt}
)

chat_history = []

# The loop only runs when this file is started directly, so the eval script can
# import parse_filters/build_where without launching the chat.
if __name__ == "__main__":
    print("🎵 Music Recommendation Chatbot")
    print("Ask me about albums, artists, or get personalized recommendations!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    while True:
        query = input("🎤 Ask me about an album or review: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Keep listening to great music!")
            break

        # Work out this question's filters, then attach them to the shared
        # retriever. The chain reads retriever.search_kwargs on every search.
        filters = parse_filters(query)
        where = build_where(filters)
        if where and not vectorstore.get(where=where, limit=1)["ids"]:
            # Tell the user rather than silently answering from no reviews.
            print(f"\n⚠️  No reviews match {filters}; searching without filters.")
            where = None
        if where:
            print(f"\n🔎 Filters applied: {filters}")
            retriever.search_kwargs["filter"] = where
        else:
            retriever.search_kwargs.pop("filter", None)

        result = chat_chain.invoke({"question": query, "chat_history": chat_history})
        answer = result["answer"]
        print("\n🎧 Response:\n", answer)

        # Hector's own writeup notes that a bad hypothetical degrades retrieval
        # "silently with no warning to the user" - printing it fixes that.
        if retriever.last_hypothetical:
            print(f"\n🧪 HyDE search text: {retriever.last_hypothetical}")

        print("\n📄 Sources:")
        for i, doc in enumerate(result["source_documents"], 1):
            m = doc.metadata
            print(f"\n[{i}] {m.get('artist')} - {m.get('album')} "
                  f"({m.get('year')}, score {m.get('score')}, {m.get('genre')})")
            print(f"    {doc.page_content}")

        chat_history.append((query, answer))