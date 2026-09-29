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
from langchain_classic.chains import ConversationalRetrievalChain
from musicChatbot import (vectorstore, parse_filters, build_where, llm, qa_prompt,
                          retriever, chat_chain, script_dir)

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
    """Generate real answers with three setups and count the albums they name."""
    reviews = pd.read_excel(script_dir / "pitchfork_reviews_v3.xlsx")
    baseline_chain = ConversationalRetrievalChain.from_llm(   # the original lab code
        llm=llm, retriever=vectorstore.as_retriever(search_kwargs={"k": 10}),
        return_source_documents=True, combine_docs_chain_kwargs={"prompt": qa_prompt})
    filters_only_chain = ConversationalRetrievalChain.from_llm(  # filters, no labels
        llm=llm, retriever=retriever,
        return_source_documents=True, combine_docs_chain_kwargs={"prompt": qa_prompt})
    setups = [("baseline", baseline_chain), ("filters only", filters_only_chain),
              ("filters + labels", chat_chain)]

    totals = {name: [0, 0] for name, _ in setups}
    for question, expected in ANSWER_QUESTIONS:
        reviews["expected"] = [expected] * len(reviews)
        where = build_where(parse_filters(question))
        print(f"\nQ: {question}")
        for name, chain in setups:
            retriever.search_kwargs.pop("filter", None)
            if where and name != "baseline":
                retriever.search_kwargs["filter"] = where
            answer = chain.invoke({"question": question, "chat_history": []})["answer"]
            good, bad = albums_named(answer, reviews)
            totals[name][0] += len(good)
            totals[name][1] += len(bad)
            print(f"  {name:<17} matching albums named: {len(good)}  non-matching: {len(bad)}")
            print(f"    {' '.join(answer.split())[:300]}...")
    retriever.search_kwargs.pop("filter", None)

    print("\nTotals across questions (dataset albums named in answers):")
    for name, (good, bad) in totals.items():
        print(f"  {name:<17} matching {good:>3}   non-matching {bad:>3}")


if __name__ == "__main__":
    main()
    if "--answers" in sys.argv:
        answer_eval()
