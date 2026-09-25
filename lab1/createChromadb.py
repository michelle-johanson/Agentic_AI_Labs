from sentence_transformers import SentenceTransformer
import pandas as pd
import chromadb
from chromadb.config import Settings
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from pathlib import Path
from dotenv import load_dotenv
import re
import os
from us_states import US_STATES, STATE_CODES
load_dotenv()

# Embedding model. all-MiniLM-L6-v2 (22M params, 384-dim, 256-token window) is
# the baseline; set EMBED_MODEL=all-mpnet-base-v2 (110M params, 768-dim,
# 384-token window) to compare. jobSearch.py reads the same setting.
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")

script_dir = Path(__file__).parent
chroma_path = script_dir / "chroma"
df = pd.read_excel(script_dir / "Jobs for Semantic Search.xlsx")
df = df.fillna("").astype(str)

# Chroma metadata values must be scalars, so every field is coerced to a
# plain string. These raw fields are kept for display; the *_norm / is_remote /
# city / state fields below are derived copies used for filtering and boosting.
metadatas = df[["Job title", "Company", "Location", "Skills", "URL"]].to_dict(orient="records")

# --- Normalize location/company/title/skills for filtering and boosting ----
# This is prep for the second improvement (see jobSearch.py): metadata filters
# and exact-term boosts. Chroma `where` filters are exact-match only, and the
# raw Location column has 100+ distinct spellings ("Remote, US", "Remote (US)",
# "New York, NY (Remote eligible)", ...), so we derive clean lowercase fields
# here so a query can filter on "is this remote / this city / this state"
# without string-matching the raw text.
CITY_STATE_RE = re.compile(r"([A-Za-z .'\-]+?),\s*([A-Z]{2})\b")

def location_fields(loc: str) -> dict:
    pairs = [(c.strip().lower(), st.lower())
             for c, st in CITY_STATE_RE.findall(loc)
             if st != "US" and not re.fullmatch(r"[A-Z]{2}", c.strip())]   # "CT, VT" is not a city
    state = pairs[0][1] if pairs else ""
    if not state:
        # No "City, ST" pair: fall back to a full state name ("Utah",
        # "Arizona (Remote, US)"), then a bare code list ("Hybrid (CT, VT, ...)").
        lower = loc.lower()
        for name, code in sorted(US_STATES.items(), key=lambda kv: -len(kv[0])):
            if re.search(rf"(?<!\w){name}(?!\w)", lower):
                state = code
                break
        else:
            codes = [c.lower() for c in re.findall(r"\b([A-Z]{2})\b", loc) if c.lower() in STATE_CODES]
            state = codes[0] if codes else ""
    return {
        "is_remote": bool(re.search(r"remote|virtual", loc, re.I)),
        "city": pairs[0][0] if pairs else "",
        "state": state,
    }

for meta in metadatas:
    meta.update(location_fields(meta["Location"]))
    meta["company_norm"] = meta["Company"].strip().lower()
    meta["title_norm"] = meta["Job title"].strip().lower()
    meta["skills_norm"] = ",".join(sk.strip().lower() for sk in meta["Skills"].split(",") if sk.strip())

sbert_ef = SentenceTransformerEmbeddingFunction(model_name=EMBED_MODEL)

# --- Chunk long descriptions (main improvement #1) --------------------------
# all-MiniLM-L6-v2 reads at most 256 tokens and silently drops the rest, and a
# single pooled vector for a long posting is a blurrier summary than one for a
# short paragraph. So instead of one vector per posting we store one vector per
# *chunk*: the description is split into sentence-packed pieces, each prefixed
# with a Title+Skills header so it carries context on its own. jobSearch.py
# scores a posting by its single best-matching chunk.
_model = SentenceTransformer(EMBED_MODEL)
tokenizer = _model.tokenizer
MAX_TOKENS = _model.max_seq_length          # 256 for MiniLM, 384 for mpnet
CHUNK_TOKENS = 120  # sweeping 220/160/120/80/50 showed 120 (~one paragraph) as the recall knee

def n_tokens(text: str) -> int:
    return len(tokenizer(text, add_special_tokens=False)["input_ids"])

def chunk_text(text: str, budget: int):
    """Greedy sentence packing under `budget` tokens, 1-sentence overlap."""
    sents = [x for x in re.split(r"(?<=[.!?])\s+", text.strip()) if x]
    chunks, cur, cur_len = [], [], 0
    for sent in sents:
        n = n_tokens(sent)
        if cur and cur_len + n > budget:
            chunks.append(" ".join(cur))
            cur, cur_len = [cur[-1]], n_tokens(cur[-1])      # keep last sentence as overlap
        cur.append(sent)
        cur_len += n
    if cur:
        chunks.append(" ".join(cur))
    return chunks or [""]

client = chromadb.PersistentClient(
    path=chroma_path,
    settings=Settings(anonymized_telemetry=False)
)

# Drop any existing collection so re-running this script replaces the index
# instead of failing with "collection already exists".
if "jobs" in [c.name for c in client.list_collections()]:
    client.delete_collection(name="jobs")

# Cosine space, so `1 - distance` in jobSearch.py is a real cosine similarity
# (Chroma's default is squared L2, which makes that math meaningless).
collection = client.create_collection(
    name="jobs",
    embedding_function=sbert_ef,
    metadata={"hnsw:space": "cosine"}
)

chunk_docs, chunk_metas, chunk_ids = [], [], []
for i, meta in enumerate(metadatas):
    # Title + Skills go into the embedded text alongside the Description, so a
    # query like "Kubernetes" can match even when the free-text description
    # never says the word. Company/Location are deliberately left out of the
    # embedding (proper nouns embed poorly) and are instead handled exactly via
    # the metadata filters and boosts in jobSearch.py.
    header = f"{df.loc[i, 'Job title']}\n{df.loc[i, 'Skills']}"
    pieces = chunk_text(df.loc[i, "Description"], budget=min(CHUNK_TOKENS, MAX_TOKENS - n_tokens(header) - 4))
    for k, piece in enumerate(pieces):
        chunk_docs.append(f"{header}\n{piece}")
        chunk_metas.append({**meta, "job_id": f"job_{i}", "chunk": k, "n_chunks": len(pieces)})
        chunk_ids.append(f"job_{i}_c{k}")

collection.add(documents=chunk_docs, metadatas=chunk_metas, ids=chunk_ids)

over = sum(n_tokens(d) > MAX_TOKENS for d in chunk_docs)
print(f"[{EMBED_MODEL}] Indexed {len(metadatas)} jobs as {collection.count()} chunks into {chroma_path} "
      f"(max {max(m['n_chunks'] for m in chunk_metas)} chunks/job, {over} chunks still over {MAX_TOKENS} tokens)")