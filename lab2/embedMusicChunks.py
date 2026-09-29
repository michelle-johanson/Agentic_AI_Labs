import pandas as pd
from langchain_text_splitters import RecursiveCharacterTextSplitter
import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions
from chromadb import PersistentClient
from pathlib import Path
from dotenv import load_dotenv
import os
import time

load_dotenv()

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"

# Same pattern as Lab 1's createChromadb.py: keep the build settings together
# near the top, and let EMBED_MODEL select a different model without editing
# the script. Re-running always replaces the one musicReviews collection.
EMBED_MODEL = os.getenv(
    "EMBED_MODEL", "sentence-transformers/all-mpnet-base-v2"
)
CHUNK_SIZE = 300
CHUNK_OVERLAP = 50

# Use the production-data name when available, while retaining compatibility
# with the v3 workbook currently checked into this lab.
dataset_candidates = (
    script_dir / "pitchfork_reviews_v2.xlsx",
    script_dir / "pitchfork_reviews_v3.xlsx",
)
dataset_path = next((path for path in dataset_candidates if path.exists()), None)
if dataset_path is None:
    expected = ", ".join(path.name for path in dataset_candidates)
    raise FileNotFoundError(f"Could not find a Pitchfork workbook. Expected one of: {expected}")

df = pd.read_excel(dataset_path)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

texts = []
metadatas = []
ids = []


def metadata_value(value):
    """Return a Chroma-compatible scalar instead of NaN/Timestamp values."""
    if value is None or pd.isna(value):
        return ""
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    return value


def first_value(row, *column_names):
    """Use the first available column name across workbook versions."""
    for column_name in column_names:
        if column_name in row.index and pd.notna(row[column_name]):
            return row[column_name]
    return ""

for row_number, (_, row) in enumerate(df.iterrows()):
    # All chunks from one review share a source_id for traceability.
    source_id = f"pitchfork_review_{row_number:06d}"
    title = first_value(row, "title", "review_title", "album")
    chunks = splitter.split_text(str(row["review"]))
    metadata = {
        "artist": metadata_value(row["artist"]),
        "title": metadata_value(title),
        "score": metadata_value(row["score"]),
        "source_id": source_id,
        # Preserve the original field name for Lab 2 code that expects album.
        "album": metadata_value(title),
        "year": metadata_value(row["year"]),
        "reviewer": metadata_value(row["reviewer"]),
        "genre": metadata_value(row["genre"]),
        "label": metadata_value(row["label"]),
        "reviewdate": metadata_value(row["review_date"]),
    }
    header = (
        f"Artist: {metadata['artist']}\n"
        f"Album: {metadata['album']}\n"
        f"Year: {metadata['year']}\n"
        f"Genre: {metadata['genre']}\n"
        f"Pitchfork score: {metadata['score']}"
    )
    for chunk_number, chunk in enumerate(chunks):
        # A review excerpt often does not name its own artist or album. Repeating
        # this short identity header gives every chunk enough context to match a
        # name-based query and lets the embedding represent its metadata.
        texts.append(f"{header}\n\n{chunk}")
        chunk_metadata = metadata.copy()
        chunk_metadata["chunk"] = chunk_number
        chunk_metadata["n_chunks"] = len(chunks)
        metadatas.append(chunk_metadata)
        ids.append(f"{source_id}_chunk_{chunk_number:04d}")

sentence_transformer_ef = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBED_MODEL
)

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(anonymized_telemetry=False)
)

# The collection is derived from the workbook. Rebuild it so an index created
# by the old embedder cannot be mixed with chunks carrying the new metadata.
if "musicReviews" in [
    existing_collection.name for existing_collection in client.list_collections()
]:
    client.delete_collection(name="musicReviews")

collection = client.create_collection(
    name="musicReviews",
    embedding_function=sentence_transformer_ef,
)

BATCH_SIZE = 1000

print(f"Length of texts: {len(texts)}")
print(f"Total batches to encode and load: {len(texts)/BATCH_SIZE}")
print(
    f"Building musicReviews with {EMBED_MODEL}, "
    f"chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP}"
)

start_time = time.perf_counter()

for i in range(0, len(texts), BATCH_SIZE):
    batch_texts = texts[i:i + BATCH_SIZE]
    batch_metadatas = metadatas[i:i + BATCH_SIZE]

    print(f"Encoding and adding batch {i}")
    
    # Add batch to collection
    collection.add(
        documents=batch_texts,
        metadatas=batch_metadatas,
        ids=ids[i:i + BATCH_SIZE],
    )
    
    print(f"✓ Processed {min(i + BATCH_SIZE, len(texts))}/{len(texts)} chunks")

end_time = time.perf_counter()
execution_time = end_time - start_time
print(f"Execution time: {execution_time:.6f} seconds")

print(
    f"✅ Rebuilt musicReviews with {collection.count()} chunks. "
    "Run this script again whenever you change the model or chunk settings."
)
