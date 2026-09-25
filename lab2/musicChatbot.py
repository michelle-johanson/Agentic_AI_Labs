from langchain_chroma import Chroma
from langchain_ollama import ChatOllama
from langchain_classic.chains import ConversationalRetrievalChain
from langchain_core.prompts import PromptTemplate
from pathlib import Path
from dotenv import load_dotenv
load_dotenv() 

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

llm = ChatOllama(model="mistral", temperature=0.7)

vectorstore = Chroma(
    collection_name="musicReviews",
    persist_directory=chroma_path
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

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

chat_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    return_source_documents=True,  # optional, for debugging
    combine_docs_chain_kwargs={"prompt": qa_prompt}
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