"""Terminal music recommendation chatbot.

All the RAG logic lives in musicRagCore.py, shared with gradioMusicChatbot.py.
This file only chooses the configuration and prints the results:

    model      mistral, temperature 0.7
    retrieval  plain similarity search, k=10

The Gradio app uses phi3 and MMR k=4, which Step 4 of the lab asks us to
compare - those are settings passed in here, not a second copy of the pipeline.
"""

import os

from dotenv import load_dotenv
from langchain_ollama import ChatOllama

import musicRagCore as core

load_dotenv()

llm = ChatOllama(model="mistral", temperature=0.7)
retriever = core.make_retriever(search_type="similarity", k=10)


def _enabled(name):
    """Read an off-by-default env toggle, so both arms of a before/after eval
    come from the same code."""
    return os.getenv(name, "0").lower() in ("1", "true", "yes")


# Both off by default: they change which chunks come back, which invalidates the
# confidence threshold swept in Improvement 8 until it is re-run.
#   HYDE_ENABLED=1  search with a generated review excerpt instead of the question
#   BM25_ENABLED=1  fuse in a keyword search leg (needs `pip install rank-bm25`)
USE_HYDE = _enabled("HYDE_ENABLED")
USE_BM25 = _enabled("BM25_ENABLED")

# Re-exported so evalMetadataFilters.py keeps working unchanged.
vectorstore = core.vectorstore
parse_filters = core.parse_filters
build_where = core.build_where
qa_prompt = core.QA_PROMPT
script_dir = core.script_dir


def ask(question, history):
    """Answer one question with this chatbot's model and retriever."""
    return core.answer(question, history, llm, retriever,
                       use_hyde=USE_HYDE, use_bm25=USE_BM25)


if __name__ == "__main__":
    print("🎵 Music Recommendation Chatbot")
    print("Ask me about albums, artists, or get personalized recommendations!")
    print("Type 'exit' or 'quit' to end the conversation.\n")

    history = []

    while True:
        query = input("🎤 Ask me about an album or review: ")
        if query.lower() in ["exit", "quit"]:
            print("👋 Goodbye! Keep listening to great music!")
            break

        result = ask(query, history)

        # Show the rewritten question when a follow-up was expanded, so a
        # surprising search can be traced to the rewrite.
        if result["search_question"] != query:
            print(f"\n↻ Searched for: {result['search_question']}")
        if result["filters"]:
            print(f"🔎 Filters applied: {result['filters']}")
        if result["hyde_text"]:
            print(f"🧪 HyDE search text: {result['hyde_text']}")

        print("\n🎧 Response:\n", result["answer"])

        if not result["refused"]:
            print(core.format_sources(result["documents"]))

            print("\n📄 Sources:")
            for i, doc in enumerate(result["documents"], 1):
                m = doc.metadata
                print(f"\n[{i}] {m.get('artist')} - {core._title_of(m)} "
                      f"({core._shown(m, 'year')}, score {m.get('score')}, "
                      f"{core._shown(m, 'genre')})")
                print(f"    {doc.page_content}")

        history.append({"role": "user", "content": query})
        history.append({"role": "assistant", "content": result["answer"]})
