# Lab Extension Memo

These are the improvements we made to Lab 2. I used simple words on purpose.

## Improvement 1: Better information on every review chunk

### 1. What we extended

In `embedMusicChunks.py`, every small review chunk now keeps four important
pieces of information:

- `artist`
- `title`
- `score`
- `source_id`

The `source_id` tells us which full review the chunk came from.

### 2. Why we chose this

Before, the code used `album` and did not have a `source_id`.

We chose this because the chatbot needs to know where its information came
from. We did not make a separate database because Chroma metadata already gives
us a simple place to store this information.

### 3. Benefits

Before: **3 of the 4 needed fields** were easy to use (`artist`, `album`, and
`score`).

After: **4 of 4 fields** are available (`artist`, `title`, `score`, and
`source_id`).

### 4. Drawbacks and tradeoffs

- The index must be rebuilt after changing the metadata.
- Each review chunk uses a little more storage.
- The `source_id` identifies the review row in this workbook. A larger system
  would need a permanent ID from the original database.

## Improvement 2: More varied search results with MMR

### 1. What we extended

In `gradioMusicChatbot.py`, we changed the retriever to use maximum marginal
relevance, or MMR:

```python
search_type="mmr"
search_kwargs={"k": 4, "fetch_k": 20, "lambda_mult": 0.5}
```

### 2. Why we chose this

Before, the chatbot took the top **10** most similar chunks. That could give it
many chunks that said almost the same thing.

MMR first looks at **20** possible chunks, then chooses **4** that are both
useful and a little different from each other.

We chose MMR instead of simply asking for more chunks because more chunks can
make the prompt noisy and harder for the model to read.

### 3. Benefits

Before: **10** similar chunks were requested.

After: **4** more varied chunks are sent to the model from a pool of **20**.

This gives the model less repeated information and keeps the prompt smaller.

### 4. Drawbacks and tradeoffs

- MMR makes one larger search before choosing the final results.
- Four chunks may leave out useful information.
- The value `0.5` for `lambda_mult` is a reasonable starting point, not a
  perfect answer. It should be tested with real user questions.

## Improvement 3: A safety check before Phi-3 answers

### 1. What we extended

We added a confidence check before the `llm.invoke()` call in
`gradioMusicChatbot.py`.

If none of the retrieved chunks has a high enough relevance score, the chatbot
does not call Phi-3. It politely says that it does not have enough evidence.

### 2. Why we chose this

Before, Phi-3 was called every time, even when the search results were weak.

We chose a confidence check because a short refusal is safer than a made-up
answer. We made the threshold configurable with
`RETRIEVAL_CONFIDENCE_THRESHOLD`.

### 3. Benefits

Before: **0** retrieval safety checks happened before Phi-3.

After: **1** safety check happens before every Phi-3 call.

The threshold starts at `0.5`, but it should be changed after testing a real
validation set.

We do not claim a better accuracy percentage yet because this repo does not
include a labeled validation set.

### 4. Drawbacks and tradeoffs

- A threshold that is too high may refuse helpful questions.
- A threshold that is too low may still allow weak evidence through.
- The score depends on the embedding model, so changing models means testing
  the threshold again.

## Improvement 4: Show where the answer came from

### 1. What we extended

The prompt now labels each review chunk as `[Source 1]`, `[Source 2]`, and so
on. The Gradio answer also shows the supporting artist, review title, and
`source_id`.

### 2. Why we chose this

Before, the user saw an answer but did not see which Pitchfork review helped
make it.

We chose simple source labels because they are easy for both the model and the
user to understand. We did not display the entire review because that would
make the answer long and cluttered.

### 3. Benefits

Before: **0** artist/title source labels were shown in the response.

After: up to **4** retrieved chunks can be labeled, and repeated chunks from
the same review are combined into one supporting-review line.

This makes the answer easier to check.

### 4. Drawbacks and tradeoffs

- Source labels use a little more prompt and response space.
- The chatbot still depends on the model to use the sources correctly.
- Showing a source does not automatically prove that every sentence is true.

## Improvement 5: Rebuild the Chroma collection safely

### 1. What we extended

`embedMusicChunks.py` now removes the old `musicReviews` collection before
creating the new one.

### 2. Why we chose this

The old collection may not have the new metadata fields. Mixing old chunks and
new chunks could give inconsistent results.

We rebuild the collection because the vectors are created from the workbook
and can be made again. We did not keep the old collection as a second active
collection because the chatbot expects one collection named `musicReviews`.

### 3. Benefits

Before: rerunning the script could fail because the collection already existed,
or it could leave old metadata in place.

After: **1** clean collection is created with the new metadata every time the
embedding script runs.

### 4. Drawbacks and tradeoffs

- Rebuilding takes time because every review must be embedded again.
- The old Chroma collection is deleted, so it should be backed up if it is
  needed for another project.
- The workbook must be available when rebuilding.
