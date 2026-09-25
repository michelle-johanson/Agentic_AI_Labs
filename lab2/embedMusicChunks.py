import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
from pathlib import Path
from dotenv import load_dotenv
import time
load_dotenv() 

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"
df = pd.read_excel(script_dir / "pitchfork_reviews_v3.xlsx")

splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

texts = []
metadatas = []

for _, row in df.iterrows():
    chunks = splitter.split_text(row["review"])
    metadata = {
        "artist": row["artist"],
        "album": row["album"],
        "score": row['score'],
        "year": row['year'],
        "reviewer": row['reviewer'],
        "genre": row['genre'],
        "label": row['label'],
        "reviewdate": row['review_date']
    }
    for chunk in chunks:
        texts.append(chunk)
        metadatas.append(metadata)

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name="sentence-transformers/all-mpnet-base-v2"
)

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(anonymized_telemetry=False)
)

collection = client.create_collection(
    name="musicReviews",
    embedding_function=sentence_transformer_ef
)

BATCH_SIZE = 1000

print(f"Length of texts: {len(texts)}")
print(f"Total batches to encode and load: {len(texts)/BATCH_SIZE}")

start_time = time.perf_counter()

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_metadatas = metadatas[i:i + BATCH_SIZE]

    print(f"Encoding and adding batch {i}")
    
    # Add batch to collection
    collection.add(
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=[f"chunk_{j}" for j in range(i, i + len(batch_texts))]
    )
    
    print(f"✓ Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

print("✅ All documents added successfully!")
