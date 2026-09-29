from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from langchain_core.retrievers import BaseRetriever
from langchain_core.documents import Document
from langchain_core.callbacks.manager import CallbackManagerForRetrieverRun
from rank_bm25 import BM25Okapi
from typing import List, Any, Dict
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

llm = ChatOllama(model="mistral:7b", temperature=0.7)

vectorstore = Chroma(
    collection_name="musicReviews",
    persist_directory=chroma_path
)

# Build BM25 index at startup from the full ChromaDB corpus.
# BM25 is a keyword (sparse) retriever: it matches exact words in the query
# against words in each chunk. This complements dense (semantic) search,
# which matches meaning but can miss exact artist/album name lookups.
print("Building BM25 index...", flush=True)
# Fetch in batches: a single .get() on a large collection hits SQLite's
# variable-count limit. Paging with limit/offset avoids the crash.
_BATCH = 5000
_bm25_texts: List[str] = []
_bm25_metadatas: List[dict] = []
_offset = 0
while True:
    _batch = vectorstore._collection.get(
        include=["documents", "metadatas"],
        limit=_BATCH,
        offset=_offset,
    )
    if not _batch["documents"]:
        break
    _bm25_texts.extend(_batch["documents"])
    _bm25_metadatas.extend(_batch["metadatas"])
    _offset += _BATCH
_bm25_corpus = [text.lower().split() for text in _bm25_texts]
_bm25_index = BM25Okapi(_bm25_corpus)
_bm25_doc_store = [
    Document(page_content=_bm25_texts[i], metadata=_bm25_metadatas[i])
    for i in range(len(_bm25_texts))
]
print(f"BM25 index built over {len(_bm25_texts)} chunks.", flush=True)


def _reciprocal_rank_fusion(
    list_a: List[Document], list_b: List[Document], rrf_k: int = 60
) -> List[Document]:
    """Merge two ranked document lists using Reciprocal Rank Fusion (RRF).
    Each document's score = sum of 1/(rank + rrf_k) across both lists.
    rrf_k=60 is the standard constant that reduces the dominance of rank-1."""
    scores: Dict[str, float] = {}
    doc_map: Dict[str, Document] = {}
    for ranked_list in [list_a, list_b]:
        for rank, doc in enumerate(ranked_list):
            key = doc.page_content[:80]  # content prefix as dedup key
            scores[key] = scores.get(key, 0.0) + 1.0 / (rank + rrf_k)
            doc_map[key] = doc
    return sorted(doc_map.values(), key=lambda d: scores[d.page_content[:80]], reverse=True)


# HyDE: Hypothetical Document Embeddings.
# Instead of embedding the user's question directly, we first ask the LLM to
# write a short fake Pitchfork review excerpt that would answer the question.
# That generated text uses review vocabulary, so its embedding lands closer to
# real review chunks in vector space than the raw user question does.
HYDE_PROMPT = PromptTemplate(
    input_variables=["query"],
    template="""Write a 2-3 sentence excerpt from a Pitchfork music review that would 
directly satisfy the following request. Use the same vocabulary and tone as a real 
music review. Write only the excerpt, no preamble or explanation.

Request: {query}

Review excerpt:"""
)


def _generate_hypothetical_document(query: str) -> str:
    """Generate a hypothetical Pitchfork review excerpt for the given query.
    Falls back to the original query if the LLM call fails."""
    try:
        response = llm.invoke(HYDE_PROMPT.format(query=query))
        return response.content.strip()
    except Exception:
        return query  # safe fallback


# Maps query keywords to substrings that appear in the ChromaDB genre field.
# Uses $contains so compound genres like "Experimental / Jazz" still match "Jazz".
GENRE_KEYWORDS = {
    "jazz": "Jazz",
    "rock": "Rock",
    "electronic": "Electronic",
    "hip hop": "Rap",
    "hip-hop": "Rap",
    "rap": "Rap",
    "pop": "Pop/R&B",
    "r&b": "Pop/R&B",
    "experimental": "Experimental",
    "folk": "Folk/Country",
    "country": "Folk/Country",
    "metal": "Metal/Hard Rock",
    "punk": "Rock",
    "ambient": "Electronic",
}

def _detect_genre_keyword(query: str) -> str | None:
    """Return the genre substring to filter on if the query mentions a known
    genre, else None. Filtering is done in Python after retrieval rather than
    via a ChromaDB where-clause, since ChromaDB's $contains operator applies
    to document content, not metadata fields."""
    q = query.lower()
    for keyword, genre in GENRE_KEYWORDS.items():
        if keyword in q:
            return genre
    return None

class MetadataNormalizingRetriever(BaseRetriever):
    """Retriever that (1) fetches a larger candidate pool when the query
    mentions a known genre and post-filters by genre in Python, narrowing
    the context to relevant chunks, and (2) replaces None metadata values
    with 'Unknown' so document_prompt can safely format all fields."""
    vectorstore: Any
    k: int = 10

    class Config:
        arbitrary_types_allowed = True

    def _get_relevant_documents(
        self, query: str, *, run_manager: CallbackManagerForRetrieverRun
    ) -> List[Document]:
        genre = _detect_genre_keyword(query)
        if genre:
            # Genre query: use filtered dense search only (BM25 adds little here
            # since the genre filter already restricts the search space effectively).
            candidates = self.vectorstore.similarity_search(query, k=self.k * 4)
            filtered = [
                d for d in candidates
                if genre.lower() in (d.metadata.get("genre") or "").lower()
            ]
            docs = filtered[:self.k] if len(filtered) >= self.k else candidates[:self.k]
        else:
            # Non-genre query (artist lookup, mood query, etc.): hybrid BM25 + dense
            # with HyDE applied to the dense leg.
            # HyDE: generate a hypothetical review excerpt and use it as the dense
            # search query. This aligns the query embedding with review vocabulary.
            # BM25 still uses the original query for exact keyword matching.
            hypothetical = _generate_hypothetical_document(query)
            dense_docs = self.vectorstore.similarity_search(hypothetical, k=self.k * 2)
            tokenized_query = query.lower().split()
            bm25_scores = _bm25_index.get_scores(tokenized_query)
            top_bm25_indices = bm25_scores.argsort()[-(self.k * 2):][::-1]
            bm25_docs = [_bm25_doc_store[i] for i in top_bm25_indices]
            docs = _reciprocal_rank_fusion(dense_docs, bm25_docs)[:self.k]
        for doc in docs:
            for field in ["artist", "album", "score", "genre", "year"]:
                if doc.metadata.get(field) is None:
                    doc.metadata[field] = "Unknown"
        return docs

retriever = MetadataNormalizingRetriever(vectorstore=vectorstore, k=10)

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

# Each retrieved chunk is prefixed with a structured metadata header before being
# injected into {context}. This gives the LLM explicit anchor points (artist, album,
# score, genre, year) so it doesn't have to infer them from the review text alone.
document_prompt = PromptTemplate(
    template="[{artist} - {album} | Score: {score} | Genre: {genre} | Year: {year}]\n{page_content}",
    input_variables=["artist", "album", "score", "genre", "year", "page_content"]
)

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # optional, for debugging
    combine_docs_chain_kwargs={"prompt": qa_prompt, "document_prompt": document_prompt}
)

chat_history = []

print("🎵 Music Recommendation Chatbot")
print("Ask me about albums, artists, or get personalized recommendations!")
print("Type 'exit' or 'quit' to end the conversation.\n")

while True:
    query = input("🎤 Ask me about an album or review: ")
    if query.lower() in ["exit", "quit"]:
        print("👋 Goodbye! Keep listening to great music!")
        break
    
    result = chat_chain.invoke({"question": query, "chat_history": chat_history})
    answer = result["answer"]
    print("\n🎧 Response:\n", answer)

    print("\n📄 Sources:")
    for i, doc in enumerate(result["source_documents"], 1):
        m = doc.metadata
        print(f"\n[{i}] {m.get('artist')} - {m.get('album')} "
              f"({m.get('year')}, score {m.get('score')}, {m.get('genre')})")
        print(f"    {doc.page_content}")
 
    chat_history.append((query, answer))