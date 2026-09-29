from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
import os
import gradio as gr

load_dotenv()

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

llm = ChatOllama(model="phi3", temperature=0.4)

RETRIEVAL_CONFIDENCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", "0.5")
)
REFUSAL_MESSAGE = (
    "I don't have enough relevant evidence in the Pitchfork reviews to answer "
    "that reliably. Please try a more specific music question."
)

vectorstore = Chroma(
    collection_name="musicReviews",
    persist_directory=chroma_path
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

chat_history = []


def retrieve_with_confidence(query):
    """Retrieve with MMR, then attach relevance scores for the confidence gate.

    LangChain's MMR retriever returns documents without scores. The second
    lookup only obtains normalized relevance scores for the selected chunks;
    the documents sent to the model are still exactly the MMR results.
    """
    documents = retriever.invoke(query)
    if not documents:
        return [], 0.0

    scored_documents = vectorstore.similarity_search_with_relevance_scores(
        query, k=20
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
    )


def format_context(documents):
    formatted = []
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        formatted.append(
            f"[Source {source_number}]\n"
            f"Artist: {metadata.get('artist', 'Unknown artist')}\n"
            f"Review title: {metadata.get('title', metadata.get('album', 'Unknown title'))}\n"
            f"Score: {metadata.get('score', 'Unknown')}\n"
            f"Review excerpt: {document.page_content}"
        )
    return "\n\n".join(formatted)


def format_sources(documents):
    """Display one traceable artist/title entry per source review."""
    seen_source_ids = set()
    source_lines = []
    for source_number, document in enumerate(documents, 1):
        metadata = document.metadata
        source_id = metadata.get("source_id", f"chunk-{source_number}")
        if source_id in seen_source_ids:
            continue
        seen_source_ids.add(source_id)
        artist = metadata.get("artist", "Unknown artist")
        title = metadata.get("title", metadata.get("album", "Unknown title"))
        source_lines.append(
            f"- [Source {source_number}] {artist} — {title} ({source_id})"
        )
    return "\n\nSupporting Pitchfork reviews:\n" + "\n".join(source_lines)


def format_chat_history():
    return "\n".join(
        f"User: {question}\nAssistant: {answer}"
        for question, answer in chat_history
    ) or "(No previous conversation.)"

def get_response(message, history):
    documents, best_confidence = retrieve_with_confidence(message)

    # This gate runs before phi3 is invoked. The default is a starting point;
    # production deployments should set it from a held-out validation set.
    if best_confidence < RETRIEVAL_CONFIDENCE_THRESHOLD:
        answer = REFUSAL_MESSAGE
    else:
        prompt = qa_prompt.format(
            context=format_context(documents),
            chat_history=format_chat_history(),
            question=message,
        )
        response = llm.invoke(prompt)
        answer = response.content if hasattr(response, "content") else str(response)
        answer = f"{answer}{format_sources(documents)}"
      
    # Update chat history
    chat_history.append((message, answer))

    history = history or []
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer})
    return history

def clear_history():
    """Clear chat history."""
    global chat_history
    chat_history = []
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
