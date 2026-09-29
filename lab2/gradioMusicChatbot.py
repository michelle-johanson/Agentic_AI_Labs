"""Gradio music recommendation chatbot.

All the RAG logic lives in musicRagCore.py, shared with musicChatbot.py. This
file only chooses the configuration and renders the UI:

    model      phi3, temperature 0.4
    retrieval  MMR, k=4 selected from fetch_k=20

The terminal chatbot uses mistral and plain search with k=10, which Step 4 of
the lab asks us to compare - those are settings passed in here, not a second
copy of the pipeline.
"""

import os

import gradio as gr
from dotenv import load_dotenv
from langchain_ollama import ChatOllama

import musicRagCore as core

load_dotenv()

llm = ChatOllama(model="phi3", temperature=0.4)
retriever = core.make_retriever(search_type="mmr", k=4, fetch_k=20, lambda_mult=0.5)

# Chosen by sweeping 20 answerable and 10 unanswerable questions
# (`python evalMetadataFilters.py --threshold`), not by eye. 0.35 answers 20/20
# real questions and refuses 7/10 off-topic ones. 0.50 was set before filters
# existed and refused 6 of 15 filter questions that had 4 matching chunks each.
RETRIEVAL_CONFIDENCE_THRESHOLD = float(
    os.getenv("RETRIEVAL_CONFIDENCE_THRESHOLD", str(core.DEFAULT_CONFIDENCE_THRESHOLD))
)


def get_response(message, history):
    """Answer one question and return this session's updated message list.

    `history` is the Chatbot component's own value, which Gradio keeps per
    session, so two visitors cannot see each other's conversation.
    """
    result = core.answer(message, history, llm, retriever,
                         threshold=RETRIEVAL_CONFIDENCE_THRESHOLD)

    answer_text = result["answer"]
    if not result["refused"]:
        notes = []
        if result["search_question"] != message:
            notes.append(f"↻ Searched for: {result['search_question']}")
        if result["filters"]:
            notes.append(f"🔎 Filters applied: {result['filters']}")
        if notes:
            answer_text = "\n".join(notes) + "\n\n" + answer_text
        answer_text += core.format_sources(result["documents"])

    history = list(history or [])
    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": answer_text})
    return history


def clear_history():
    """Clear this session's chat. Returning None empties the Chatbot component,
    and since history lives on that component, other visitors are untouched."""
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
