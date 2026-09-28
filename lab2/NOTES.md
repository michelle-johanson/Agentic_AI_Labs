# Module 4 Notes: Grounding LLMs with RAG

Source: AI and Agentic Systems, Module 4 slides.

## Why LLMs need help

- LLMs only do well in the domain they were trained on: human language.
- "Performing" means next-token prediction (completion).
- They're great at finding patterns in massive text datasets.
- They will answer even when they're wrong (hallucination).
- They were never meant to be fact checkers or search engines.

## What RAG is

**Retrieval-augmented generation (RAG):** LLM generation that's augmented by a focused or curated information-retrieval process.

- Gives the model information it didn't see during training, from the latest source(s) of truth.
- "Lets the model talk to the real world." / "The model needs to think with your data."

**Grounding:** intentionally connecting the model to verified data sources so answers are as accurate and relevant as possible in a specific domain.

- Open question from the slides: who decides what goes into the curated repository?

## How RAG works

It's the semantic search lab plus an LLM.

**Semantic search (Lab 1):** documents are embedded ahead of time (pre-processing) -> the query is embedded -> return the top k closest results.

**RAG flow:**
1. Pre-process the knowledge repo: each document gets its own embedding.
2. User submits a prompt.
3. The prompt is converted to a vector by the retrieval system's embedding model.
4. The retrieval system returns the most similar ("relevant") documents.
5. The prompt is hydrated: the retrieved text is added to the user's original prompt.
6. The LLM responds using the original prompt plus the injected context, plus any role or system-specific requirements.

Things to remember:
- The retrieval system's embeddings and the LLM's embeddings are entirely different.
- The LLM works with individual tokens; the retrieval system works with whole documents (or chunks).
- **Hydration / context injection** = putting the retrieved documents into the prompt itself.

## Problems with "document stuffing"

- Tempting to include everything that might help, but this backfires.
- Garbage in, garbage out (the "garbage-can model").
- Long documents eat the context window when they're judged relevant.
- **Context rot:** performance degrades as context gets bloated.
- Ask: do you need the whole document, or just specific parts?

## Context windows

- **Context window:** the amount of text (in tokens) an LLM can see and reason over at once. Like short-term memory.
- Words are not tokens: 1 word is about 1.3-1.5 tokens.
- The window has to hold all of these:
  - User prompt
  - System prompt
  - Injected context (retrieved documents, memory facts)
  - Conversation history (cumulative, because generation is autoregressive)
- Many frontier providers deliberately cap usable windows around 200k-300k, well below the architectural maximum.
- Open questions from the slides: how to reduce context use in RAG, and what happens when a conversation goes past the window.

| Model | Context window | Approx. pages (1.5-spaced) |
|---|---|---|
| GPT-3.5 | 4k | 8 |
| GPT-4 | 8k-32k | 16-60 |
| GPT-5.5 | 1.05M | 2,000 |
| Claude Sonnet 5 | 1M | 2,000 |
| Gemini 3 Pro | 1M | 2,000 |
| Mistral Large 3 | 256k | 500 |
| Qwen 3 | 128k | 250 |
| Llama 4 Scout | 10M | 20,000 |

## Three ways to hydrate the prompt

| Method | What gets added | Pros | Cons |
|---|---|---|---|
| **Concatenation** | Full text of the top k documents | Simple; full fidelity; transparent and easy to debug | Hits token limits; no prioritization, so possible dilution; noise can confuse the model |
| **Chunking** | Small chunks of the top k documents | Better retrieval relevance; can rank and filter context; scales well with vector search | Needs a planned strategy; splits can hurt coherence; extra preprocessing and latency |
| **Summarization** | A summary of the top k documents | Fewer tokens; clearer and more focused; can synthesize across docs | Can lose nuance or key details; quality varies by model; may add bias |

## Retrievers

**Sparse (keyword), e.g. BM25**
- Built on TF-IDF: does the document contain the query's words?
  - TF: how often the word appears in this document.
  - IDF: how rare the word is across the whole corpus (rarer = more informative).
- Exact lexical matching, so it doesn't capture meaning, but it's easy to interpret.
- The baseline ("OG") retrieval method.

**Dense (encoders)**
- Uses embedded vectors, like the semantic search lab.
- Matches the meaning of the query to the meaning of each document.
- More expensive than sparse, but much richer in context.

**Example query: "Find all documents that relate to dogs and rivers"**
- Sparse finds: dog, dogs, river, river.
- Dense also finds: canines, Dachshund, puppies, Old Yeller; Seine, Mississippi, irrigation, pond, Utah Lake.

**Hybrid retrievers:** an ensemble of sparse and dense.
- Each retriever produces its own score and rank for every document.
- Combine them with a weighted sum of scores, or rank fusion (based on ranks).
- In the slide's example, Doc 7 wins under both methods: sparse rank 2, dense rank 1.

## Chunking strategies

Chunking decides how source documents get split for indexing and querying. It's a trade-off between semantic coherence, retrieval accuracy, and compute cost.

| Strategy | How it works | Pros | Cons | Good for |
|---|---|---|---|---|
| Fixed-size | Fixed number of tokens, words, or characters | Simple, fast | Can split mid-sentence or mid-paragraph | Baselines |
| Semantic | Splits on natural structure (sentences, paragraphs, sections) | Keeps meaning intact; better relevance | Uneven sizes; harder to parse | Legal or academic text |
| Overlapping (sliding) | Chunks share some tokens or sentences | Less loss at chunk edges; helps queries that span chunks | More storage and indexing work | Conversational agents |
| Recursive | Tries paragraphs, then sentences, then characters until the size limit is met | Balances coherence and size | More complex; some uneven chunks | Needs both structure and size control |
| Embedding-based | Groups or splits text using embeddings | Highly adaptive | Expensive; hard to interpret | Advanced retrieval systems |
| Metadata-aware | Uses headings, timestamps, speaker labels, etc. to guide splits | Better accuracy on structured docs | Depends on good metadata extraction | Transcripts, logs, structured reports |

## How this connects to Lab 2

- **Chunking:** recursive plus overlapping (`chunk_size=300`, `chunk_overlap=50`).
- **Retriever:** dense, using the `all-mpnet-base-v2` embedding model.
- **Hydration:** the top k=10 chunks are inserted into the prompt template as `{context}`.
- **Context window:** the system prompt, the retrieved chunks, and the chat history all share it.
