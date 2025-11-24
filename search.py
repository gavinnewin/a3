import argparse
import json
import math
import re
from pathlib import Path

token_regex = re.compile(r"[A-Za-z0-9]+")
stemmer = None 

def tokenize_text(text):
    return [m.group(0).lower() for m in token_regex.finditer(text)]

def stem_token(word):
    word = word.lower()
    if stemmer:
        try:
            return stemmer.stem(word)
        except Exception:
            pass
    for ending in ("ing", "ed", "s"):
        if word.endswith(ending) and len(word) > len(ending) + 2:
            return word[:-len(ending)]
    return word

def preprocess_query(query: str):
    return [stem_token(t) for t in tokenize_text(query)]


def load_doc_table(doc_table_path: Path):
    docid_to_url = {}
    with doc_table_path.open("r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) < 3:
                continue
            doc_id = int(parts[0])
            url = parts[1]
            docid_to_url[doc_id] = url
    N = len(docid_to_url)
    return docid_to_url, N


def load_postings_for_terms(index_path: Path, terms):
    needed = set(terms)
    result = {t: [] for t in terms}

    with index_path.open("r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            t = obj["term"]
            if t in needed:
                result[t] = [(int(doc_id), int(tf)) for doc_id, tf in obj["postings"]]
                needed.remove(t)
                if not needed:
                    break

    return result


def and_search(query_terms, index_path: Path, N, top_k=5):
    postings_dict = load_postings_for_terms(index_path, query_terms)

    # If any term missing -> empty AND
    for t in query_terms:
        if not postings_dict.get(t):
            return []

    term_postings = [postings_dict[t] for t in query_terms]
    candidate_docs = set(doc_id for doc_id, _ in term_postings[0])
    for postings in term_postings[1:]:
        candidate_docs &= {doc_id for doc_id, _ in postings}
        if not candidate_docs:
            return []

    scores = {}
    for term in query_terms:
        postings = postings_dict[term]
        df = len(postings)
        idf = math.log((N + 1) / (df + 1)) + 1.0
        for doc_id, tf in postings:
            if doc_id in candidate_docs:
                scores[doc_id] = scores.get(doc_id, 0.0) + tf * idf

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


def print_results(query, query_terms, results, docid_to_url):
    print(f"\nQuery: {query!r}")
    print(f"Processed terms (AND): {query_terms}")
    if not results:
        print("No results found.")
        return
    print("\nTop results:")
    for rank, (doc_id, score) in enumerate(results, start=1):
        url = docid_to_url.get(doc_id, f"<unknown doc {doc_id}>")
        print(f"{rank}. [doc {doc_id}] score={score:.4f}")
        print(f"   {url}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--index-file", type=Path, required=True,
                        help="Path to index_merged.jsonl")
    parser.add_argument("--doc-table", type=Path, required=True,
                        help="Path to doc_table.tsv")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--query", type=str, default=None,
                        help="Optional single query to run once")
    args = parser.parse_args()

    docid_to_url, N = load_doc_table(args.doc_table)
    print(f"Loaded {N} documents.")

    if args.query:
        qterms = preprocess_query(args.query)
        results = and_search(qterms, args.index_file, N, top_k=args.top_k)
        print_results(args.query, qterms, results, docid_to_url)
        return

    print("\nEnter queries (AND-only). Empty line to quit.\n")
    while True:
        try:
            q = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not q:
            break
        qterms = preprocess_query(q)
        results = and_search(qterms, args.index_file, N, top_k=args.top_k)
        print_results(q, qterms, results, docid_to_url)


if __name__ == "__main__":
    main()