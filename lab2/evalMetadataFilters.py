"""Before/after eval for the year/genre/score filters in musicChatbot.py.

For each question we run the retrieval step twice:
  before = the original search (top 10 chunks by meaning only)
  after  = the same search with the filter that parse_filters() extracted
and count how many of the 10 chunks actually satisfy what the question asked
for. No answers are generated, so this measures retrieval only.

Run from the repo root:  python lab2/evalMetadataFilters.py
Add --answers to also generate full chatbot answers (slow, ~5 minutes).
"""
import sys
import time
import pandas as pd
import musicRagCore as core
from musicChatbot import (vectorstore, parse_filters, build_where, llm, qa_prompt,
                          retriever, script_dir)

K = 10

# (question, expected filters). Expected is what a careful person would extract.
FILTER_QUESTIONS = [
    ("What are some critically acclaimed albums from 2023?", {"year_min": 2023, "year_max": 2023}),
    ("Recommend some jazz albums", {"genre": "Jazz"}),
    ("What are the highest rated jazz albums?", {"genre": "Jazz", "min_score": 9.0}),
    ("Rap albums from 2019", {"genre": "Rap", "year_min": 2019, "year_max": 2019}),
    ("Albums with a score of 9.5 or higher", {"min_score": 9.5}),
    ("Good rock albums from the 1990s", {"genre": "Rock", "year_min": 1990, "year_max": 1999}),
    ("Metal albums released after 2015", {"genre": "Metal", "year_min": 2016}),
    ("Electronic music from 2016", {"genre": "Electronic", "year_min": 2016, "year_max": 2016}),
    ("Folk or country records from 2020", {"genre": "Folk/Country", "year_min": 2020, "year_max": 2020}),
    ("Which albums got a perfect 10?", {"min_score": 10.0}),
    ("Hip hop albums from 2021", {"genre": "Rap", "year_min": 2021, "year_max": 2021}),
    ("R&B albums from 2017", {"genre": "Pop/R&B", "year_min": 2017, "year_max": 2017}),
    ("Experimental albums from the 2000s", {"genre": "Experimental", "year_min": 2000, "year_max": 2009}),
    ("What were the best albums of 2018?", {"year_min": 2018, "year_max": 2018}),
    ("Top rated electronic albums from the 90s", {"genre": "Electronic", "year_min": 1990, "year_max": 1999, "min_score": 9.0}),
]

# Questions with no year/genre/score. The filter step should stay out of the way.
NO_FILTER_QUESTIONS = [
    "Tell me about albums with great guitar work",
    "What are the best albums for a road trip?",
    "Tell me about Kid A by Radiohead",
    "What's a good album for studying?",
    "Recommend albums similar to Radiohead",
]


def satisfies(metadata, expected):
    """True if one chunk's metadata meets every expected constraint."""
    genre = metadata.get("genre") or ""
    year, score = metadata.get("year"), metadata.get("score")
    if "genre" in expected and expected["genre"] not in genre.split(" / "):
        return False
    if "year_min" in expected and not (year and year >= expected["year_min"]):
        return False
    if "year_max" in expected and not (year and year <= expected["year_max"]):
        return False
    if "min_score" in expected and not (score and score >= expected["min_score"]):
        return False
    return True


def search(question, where):
    return vectorstore.similarity_search(question, k=K, filter=where) if where else \
        vectorstore.similarity_search(question, k=K)


def main():
    print(f"{'question':<52} {'extracted ok':<13} {'before':>7} {'after':>7}  latency")
    total_before = total_after = extracted_ok = 0
    latencies = []
    for question, expected in FILTER_QUESTIONS:
        start = time.perf_counter()
        filters = parse_filters(question)
        latencies.append(time.perf_counter() - start)
        ok = filters == {k: v if k == "genre" else float(v) for k, v in expected.items()}
        extracted_ok += ok

        before = sum(satisfies(d.metadata, expected) for d in search(question, None))
        after = sum(satisfies(d.metadata, expected) for d in search(question, build_where(filters)))
        total_before += before
        total_after += after
        print(f"{question[:51]:<52} {'yes' if ok else 'NO':<13} {before:>4}/{K} {after:>4}/{K}  {latencies[-1]:.1f}s")
        if not ok:
            print(f"    expected {expected}\n    got      {filters}")

    n = len(FILTER_QUESTIONS)
    print(f"\nChunks matching the request: before {total_before}/{n*K} "
          f"({total_before/(n*K):.0%}), after {total_after}/{n*K} ({total_after/(n*K):.0%})")
    print(f"Filters extracted exactly right: {extracted_ok}/{n}")

    print(f"\n{'no-filter question':<52} {'filters extracted':<30} same top-10")
    for question in NO_FILTER_QUESTIONS:
        start = time.perf_counter()
        filters = parse_filters(question)
        latencies.append(time.perf_counter() - start)
        before_ids = [d.page_content for d in search(question, None)]
        after_ids = [d.page_content for d in search(question, build_where(filters))]
        # Compare position by position; some reviews contain identical chunk text.
        overlap = sum(b == a for b, a in zip(before_ids, after_ids))
        print(f"{question[:51]:<52} {str(filters) or '{}':<30} {overlap}/{K}")

    print(f"\nFilter-extraction latency: mean {sum(latencies)/len(latencies):.1f}s, "
          f"max {max(latencies):.1f}s over {len(latencies)} questions")


ANSWER_QUESTIONS = [FILTER_QUESTIONS[i] for i in (0, 2, 5, 7, 10)]


def albums_named(answer, reviews):
    """Dataset albums whose title appears in the answer, split by whether they
    meet the request. Titles under 5 characters ("Pop", "Tim") are skipped
    because they match ordinary words too easily."""
    good, bad = set(), set()
    for _, row in reviews.iterrows():
        title = str(row["album"])
        if len(title) >= 5 and title in answer:
            meta = {"genre": row["genre"] if isinstance(row["genre"], str) else "",
                    "year": row["year"], "score": row["score"]}
            (good if satisfies(meta, row["expected"]) else bad).add(title)
    return good, bad


def answer_eval():
    """Generate real answers with three setups and count the albums they name.

    Rewritten for the shared core: the setups differ only in whether the filter
    is applied and whether the context carries metadata labels, which is exactly
    what the ConversationalRetrievalChain versions compared before.
    """
    reviews = pd.read_excel(script_dir / "pitchfork_reviews_v3.xlsx")

    def raw_context(documents):
        """The chain's old default document template: text with no metadata."""
        return "\n\n".join(d.page_content for d in documents)

    setups = [("baseline", False, raw_context),
              ("filters only", True, raw_context),
              ("filters + labels", True, core.format_context)]

    totals = {name: [0, 0] for name, _, _ in setups}
    for question, expected in ANSWER_QUESTIONS:
        reviews["expected"] = [expected] * len(reviews)
        _, where = core.resolve_filters(question)
        print(f"\nQ: {question}")
        for name, use_filter, format_fn in setups:
            documents, _ = core.retrieve(question, retriever, where if use_filter else None)
            answer = llm.invoke(core.QA_PROMPT.format(
                context=format_fn(documents),
                chat_history="(No previous conversation.)",
                question=question,
            ))
            text = answer.content if hasattr(answer, "content") else str(answer)
            good, bad = albums_named(text, reviews)
            totals[name][0] += len(good)
            totals[name][1] += len(bad)
            print(f"  {name:<17} matching albums named: {len(good)}  non-matching: {len(bad)}")
            print(f"    {' '.join(text.split())[:300]}...")
    retriever.search_kwargs.pop("filter", None)

    print("\nTotals across questions (dataset albums named in answers):")
    for name, (good, bad) in totals.items():
        print(f"  {name:<17} matching {good:>3}   non-matching {bad:>3}")


# --- Improvement 7: the same filters on the Gradio path --------------------
# Improvements 2-4 (MMR, confidence gate, source labels) were only in
# gradioMusicChatbot.py and Improvement 6 (filters, metadata in the prompt) was
# only in musicChatbot.py. This measures the Gradio retrieval path, which uses
# MMR k=4 rather than the plain k=10 search measured above.

GRADIO_K = 4


def gradio_eval():
    """Before/after for the Gradio path: MMR k=4, with and without the filter."""
    import gradioMusicChatbot as app

    print(f"\n{'question':<52} {'before':>7} {'after':>7}  {'gate':>10}")
    total_before = total_after = 0
    refused_before = refused_after = 0

    for question, expected in FILTER_QUESTIONS:
        filters, where = core.resolve_filters(question)

        # "before" = what the Gradio app did prior to this change: MMR with no
        # filter, and confidence scored over the whole collection.
        app.retriever.search_kwargs.pop("filter", None)
        before_docs = app.retriever.invoke(question)
        before = sum(satisfies(d.metadata, expected) for d in before_docs)
        before_conf = max(
            (s for _, s in vectorstore.similarity_search_with_relevance_scores(question, k=20)),
            default=0.0)

        # "after" = filter applied to both the MMR search and the score lookup.
        after_docs, after_conf = core.retrieve(question, app.retriever, where)
        after = sum(satisfies(d.metadata, expected) for d in after_docs)

        gate_before = before_conf < app.RETRIEVAL_CONFIDENCE_THRESHOLD
        gate_after = after_conf < app.RETRIEVAL_CONFIDENCE_THRESHOLD
        refused_before += gate_before
        refused_after += gate_after

        total_before += before
        total_after += after
        gate = f"{'refuse' if gate_before else 'answer'}->{'refuse' if gate_after else 'answer'}"
        print(f"{question[:51]:<52} {before:>4}/{GRADIO_K} {after:>4}/{GRADIO_K}  {gate:>10}")

    app.retriever.search_kwargs.pop("filter", None)
    n = len(FILTER_QUESTIONS)
    print(f"\nGradio path, chunks matching the request: "
          f"before {total_before}/{n*GRADIO_K} ({total_before/(n*GRADIO_K):.0%}), "
          f"after {total_after}/{n*GRADIO_K} ({total_after/(n*GRADIO_K):.0%})")
    print(f"Confidence gate refusals: before {refused_before}/{n}, after {refused_after}/{n}")

    # Improvement 6 put year/genre into the CLI prompt; format_context did not.
    sample = core.format_context(after_docs[:1]) if after_docs else ""
    print(f"Prompt shows Year:   {'Year:' in sample}")
    print(f"Prompt shows Genre:  {'Genre:' in sample}")





# --- Improvement 8: choosing RETRIEVAL_CONFIDENCE_THRESHOLD from data ------
# Improvement 3 set the gate to 0.5 by eye and said so. Adding filters made that
# value wrong: a filtered search compares against a much smaller pool, so its
# best relevance score is lower, and 6 of the 15 filter questions were refused
# while holding 4 matching chunks each. This sweeps the threshold over questions
# the corpus can answer and questions it cannot, and reports the trade-off.

# Music questions the 1,854 reviews genuinely can answer.
ANSWERABLE = [q for q, _ in FILTER_QUESTIONS] + NO_FILTER_QUESTIONS

# Questions no album review can answer. The gate should refuse these.
UNANSWERABLE = [
    "What is the capital of France?",
    "How do I fix a flat bicycle tire?",
    "Who won the 2022 FIFA World Cup?",
    "Give me a recipe for pizza dough.",
    "What is Apple's current stock price?",
    "Explain quantum entanglement to me.",
    "Write a Python function that reverses a string.",
    "What is the weather in Provo tomorrow?",
    "How many calories are in a banana?",
    "Translate 'good evening' into Japanese.",
]


def threshold_eval():
    """Sweep the confidence threshold over answerable vs unanswerable questions."""
    import gradioMusicChatbot as app

    def confidence(question):
        filters, where = core.resolve_filters(question)
        _, conf = core.retrieve(question, app.retriever, where)
        return conf, filters

    print("Scoring questions the corpus CAN answer...")
    answerable_scores = []
    for question in ANSWERABLE:
        conf, filters = confidence(question)
        answerable_scores.append(conf)
        print(f"  {conf:.3f}  {question[:58]}")

    print("\nScoring questions the corpus CANNOT answer...")
    unanswerable_scores = []
    for question in UNANSWERABLE:
        conf, filters = confidence(question)
        unanswerable_scores.append(conf)
        print(f"  {conf:.3f}  {question[:58]}")

    app.retriever.search_kwargs.pop("filter", None)

    n_ans, n_un = len(answerable_scores), len(unanswerable_scores)
    print(f"\nanswerable   min {min(answerable_scores):.3f}  "
          f"median {sorted(answerable_scores)[n_ans//2]:.3f}  max {max(answerable_scores):.3f}")
    print(f"unanswerable min {min(unanswerable_scores):.3f}  "
          f"median {sorted(unanswerable_scores)[n_un//2]:.3f}  max {max(unanswerable_scores):.3f}")

    print(f"\n{'threshold':>9}  {'answered (want all)':>20}  {'refused (want all)':>19}  {'correct':>8}")
    best = None
    for step in range(0, 21):
        threshold = step / 20
        answered = sum(s >= threshold for s in answerable_scores)
        refused = sum(s < threshold for s in unanswerable_scores)
        correct = (answered + refused) / (n_ans + n_un)
        marker = ""
        if best is None or correct > best[1]:
            best, marker = (threshold, correct), ""
        print(f"{threshold:>9.2f}  {answered:>13}/{n_ans}  {refused:>12}/{n_un}  {correct:>7.0%}")
    print(f"\nBest accuracy {best[1]:.0%} at threshold {best[0]:.2f} "
          f"(Improvement 3 used 0.50 by eye)")


if __name__ == "__main__":
    if "--gradio" in sys.argv:
        gradio_eval()
    elif "--threshold" in sys.argv:
        threshold_eval()
    else:
        main()
        if "--answers" in sys.argv:
            answer_eval()
