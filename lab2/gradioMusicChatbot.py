from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import gradio as gr

# Improvements 1-5 lived here and Improvement 6 (year/genre/score filters) lived
# in musicChatbot.py, so neither chatbot had the other's work. musicChatbot.py
# keeps its chat loop under `if __name__ == "__main__":`, so importing it gives
# us the filter functions and one shared vectorstore without running the CLI.
from musicChatbot import parse_filters, build_where, vectorstore

load_dotenv()

llm = ChatOllama(model="phi3", temperature=0.4)

# Chosen by sweeping 20 answerable and 10 unanswerable questions
# (`python evalMetadataFilters.py --threshold`), not by eye. 0.35 answers 20/20
# real questions and refuses 7/10 off-topic ones. 0.50 was set before filters
# existed and refused 6 of 15 filter questions that had 4 matching chunks each.
RETRIEVAL_CONFIDENCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.35")
)
REFUSAL_MESSAGE = (
    "I don't have enough relevant evidence in the Pitchfork reviews to answer "
    "that reliably. Please try a more specific music question."
)

retriever = vectorstore.as_retriever(
    search_type="mmr",
    search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5},
)

qa_prompt = PromptTemplate(
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

Context from reviews (each source includes its artist and review title):
{context}

Chat History:
{chat_history}

User question: {question}

Please provide a helpful response based on the music reviews and context available. Keep your answer grounded in the provided sources and cite supporting sources as [Source N].""",
    input_variables=["context", "chat_history", "question"]
)


def resolve_filters(query):
    """Work out this question's year/genre/score filter, or None.

    Same two steps the CLI does: ask the LLM for filters, then check at least
    one review matches. A filter that matches nothing (e.g. "albums from 2030")
    is dropped so the user gets a real answer instead of a refusal.
    """
    filters = parse_filters(query)
    where = build_where(filters)
    if where and not vectorstore.get(where=where, limit=1)["ids"]:
        return {}, None
    return filters, where


def retrieve_with_confidence(query):
    """Retrieve with MMR, then attach relevance scores for the confidence gate.

    LangChain's MMR retriever returns documents without scores. The second
    lookup only obtains normalized relevance scores for the selected chunks;
    the documents sent to the model are still exactly the MMR results.

    The filter has to go to BOTH lookups. Filtering only the MMR search would
    leave the score lookup searching all 50k chunks, so the selected documents
    would usually be missing from score_by_key, score 0.0 would win, and the
    confidence gate would refuse every filtered question.
    """
    filters, where = resolve_filters(query)
    if where:
        retriever.search_kwargs["filter"] = where
    else:
        retriever.search_kwargs.pop("filter", None)

    documents = retriever.invoke(query)
    if not documents:
        return [], 0.0, filters

    scored_documents = vectorstore.similarity_search_with_relevance_scores(
        query, k=20, filter=where
    )
    score_by_key = {
        (doc.metadata.get("source_id"), doc.page_content): float(score)
        for doc, score in scored_documents
    }

    selected = []
    for document in documents:
        key = (document.metadata.get("source_id"), document.page_content)
        score = score_by_key.get(key, 0.0)
        document.metadata = {
            **document.metadata,
            "retrieval_confidence": score,
        }
        selected.append(document)

    return selected, max(
        document.metadata["retrieval_confidence"] for document in selected
    ), filters


def _shown(metadata, key):
    """Metadata value for the prompt, or "unknown".

    Missing fields are stored as "" by the current embedding script and left out
    entirely by the original one, and years arrive as floats (2023.0).
    """
    value = metadata.get(key)
    if value in (None, ""):
        return "unknown"
    return int(value) if key == "year" and isinstance(value, float) else value


def format_context(documents):
    """Write each chunk into the prompt with the facts the user can ask about.

    Year and genre were missing here, so a question like "rock albums from the
    1990s" could retrieve the right chunks and still get an answer that named no
    years, because the model never saw them.
    """
    formatted = []
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        formatted.append(
            f"[Source {source_number}]\n"
            f"Artist: {metadata.get('artist', 'Unknown artist')}\n"
            f"Review title: {metadata.get('title', metadata.get('album', 'Unknown title'))}\n"
            f"Year: {_shown(metadata, 'year')}\n"
            f"Genre: {_shown(metadata, 'genre')}\n"
            f"Score: {metadata.get('score', 'Unknown')}\n"
            f"Review excerpt: {document.page_content}"
        )
    return "\n\n".join(formatted)


def format_sources(documents):
    """List one entry per source review, naming every [Source N] it covers.

    Two chunks often come from the same review. The previous version skipped the
    duplicate with `continue`, which dropped its number, so a model citing
    [Source 3] could find no [Source 3] line to check it against. Here the
    numbers are collected per review instead, giving "[Sources 2, 3]".

    Reviews are grouped by (artist, title) rather than by `source_id`, because
    `source_id` only exists on indexes built after Improvement 1 - on an older
    collection every chunk reports None and nothing would group.
    """
    grouped = {}
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        artist = metadata.get("artist", "Unknown artist")
        title = metadata.get("title", metadata.get("album", "Unknown title"))
        source_id = metadata.get("source_id")
        entry = grouped.setdefault((artist, title), {"numbers": [], "source_id": source_id})
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
    """Turn the Chatbot component's own message list into prompt text.

    This used to read a module-level `chat_history` list. Gradio serves every
    browser from one Python process, so that single list was shared by all
    visitors: one person's questions became another person's context, and
    "Clear Chat" wiped the history for everyone. `history` is the Chatbot
    component's value, which Gradio keeps per session, so reading it instead
    keeps conversations separate with no global state.
    """
    turns = []
    for message in history or []:
        role = "User" if message.get("role") == "user" else "Assistant"
        turns.append(f"{role}: {message.get('content', '')}")
    return "\n".join(turns) or "(No previous conversation.)"


def get_response(message, history):
    documents, best_confidence, filters = retrieve_with_confidence(message)

    # This gate runs before phi3 is invoked. The default is a starting point;
    # production deployments should set it from a held-out validation set.
    if best_confidence < RETRIEVAL_CONFIDENCE_THRESHOLD:
        answer = REFUSAL_MESSAGE
    else:
        prompt = qa_prompt.format(
            context=format_context(documents),
            chat_history=format_chat_history(history),
            question=message,
        )
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        # Show the filter the way the CLI does, so a surprising answer can be
        # traced to a wrong filter rather than to the model.
        if filters:
            answer = f"🔎 Filters applied: {filters}\n\n{answer}"
        answer = f"{answer}{format_sources(documents)}"

    # The returned list is this session's history; Gradio stores it on the
    # Chatbot component and hands it back as `history` on the next question.
    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history


def clear_history():
    """Clear this session's chat. Returning None empties the Chatbot component,
    and since history now lives on that component, nothing else needs resetting
    and other visitors' conversations are untouched."""
    return None

# Create Gradio interface
with gr.Blocks(title="Music Recommendation Chatbot") as demo:
    
    gr.Markdown("""
    # 🎵 Music Recommendation Chatbot
    Ask me about albums, artists, genres, or get personalized music recommendations!
    """)
    
    with gr.Row():
        with gr.Column(scale=4):
            chatbot = gr.Chatbot(
                height=500,
                avatar_images=(None, "🎧"),
                label="Chat"
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    placeholder="Ask about an album, artist, or request recommendations...",
                    show_label=False,
                    scale=4,
                    container=False
                )
                submit_btn = gr.Button("Send 🎤", scale=1, variant="primary")
            
            with gr.Row():
                clear_btn = gr.Button("Clear Chat 🗑️")
        
        with gr.Column(scale=1):
            gr.Markdown("""
            ### 💡 Try asking:
            - "Recommend albums similar to Radiohead"
            - "What are some good jazz albums?"
            - "Tell me about [specific album]"
            - "I want upbeat indie rock recommendations"
            - "What's a good album for studying?"
            """)
            
            gr.Markdown("""
            ### ℹ️ About
            This chatbot uses a RAG system with:
            - Local ChromaDB vector storage
            - Music review embeddings
            - MMR retrieval with a confidence gate
            - Year / genre / score filters
            - Phi-3 LLM via Ollama
            """)
    
    # Event handlers
    msg.submit(
        get_response, 
        inputs=[msg, chatbot], 
        outputs=chatbot
    ).then(
        lambda: "", 
        outputs=msg
    )
    
    submit_btn.click(
        get_response, 
        inputs=[msg, chatbot], 
        outputs=chatbot
    ).then(
        lambda: "", 
        outputs=msg
    )
    
    clear_btn.click(
        clear_history, 
        outputs=chatbot
    )
    
    # Example queries
    gr.Examples(
        examples=[
            "What are some critically acclaimed albums from 2023?",
            "Recommend me some ambient electronic music",
            "Tell me about albums with great guitar work",
            "I'm in the mood for melancholic indie folk",
            "What are the best albums for a road trip?"
        ],
        inputs=msg,
        label="Example Questions"
    )

# Launch the app
if __name__ == "__main__":
    demo.launch(
        theme=gr.themes.Soft(),
        share=False,  # Set to True to create a public link
        server_name="127.0.0.1",  # Makes it accessible on your network
        server_port=7860
    )
