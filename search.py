import argparse
import json
import math
import re
import time
from pathlib import Path

# --------- tokenization + stemming (must match indexer.py) ---------

token_regex = re.compile(r"[A-Za-z0-9]+")

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
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

def preprocess_query(query):
    return [stem_token(t) for t in tokenize_text(query)]


# --------- loading doc table ---------

def load_doc_table(doc_table_path):
    docid_to_url = {}
    with doc_table_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            doc_id = int(parts[0])
            url = parts[1]
            docid_to_url[doc_id] = url
    return docid_to_url


# --------- build / load term → offset lexicon ---------

def build_lexicon(index_path, lexicon_path):
    term_to_offset = {}
    with index_path.open("r", encoding="utf-8") as f:
        while True:
            offset = f.tell()
            line = f.readline()
            if not line:
                break
            obj = json.loads(line)
            term = obj["term"]
            term_to_offset[term] = offset
    with lexicon_path.open("w", encoding="utf-8") as f:
        json.dump(term_to_offset, f)
    return term_to_offset

def load_lexicon(index_path, lexicon_path):
    if lexicon_path.exists():
        with lexicon_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_lexicon(index_path, lexicon_path)


# --------- postings loading using lexicon ---------

def load_postings_for_terms(index_path, term_offsets, terms):
    postings = {t: [] for t in terms}
    with index_path.open("r", encoding="utf-8") as f:
        for term in terms:
            offset = term_offsets.get(term)
            if offset is None:
                continue
            f.seek(offset)
            line = f.readline()
            if not line:
                continue
            obj = json.loads(line)
            postings[term] = [(int(doc), int(tf)) for doc, tf in obj["postings"]]
    return postings


# --------- ranked AND search with OR fallback + tf-idf cosine ---------

def ranked_search(query_terms, index_path, term_offsets, num_docs, top_k=5):
    # load postings for all query terms
    postings_dict = load_postings_for_terms(index_path, term_offsets, query_terms)

    # candidate docs: AND; if empty, fall back to OR (union)
    term_postings = [postings_dict[t] for t in query_terms if postings_dict.get(t)]
    if not term_postings:
        return []

    candidate_docs = set(doc_id for doc_id, _ in term_postings[0])
    for plist in term_postings[1:]:
        candidate_docs &= {doc_id for doc_id, _ in plist}

    if not candidate_docs:
        # OR fallback: union of all postings so the user still sees something
        candidate_docs = set()
        for plist in term_postings:
            candidate_docs |= {doc_id for doc_id, _ in plist}

    # tf-idf cosine scoring
    scores = {}
    doc_norm_sq = {}
    query_norm_sq = 0.0

    for term in query_terms:
        postings = postings_dict.get(term, [])
        if not postings:
            continue
        df = len(postings)
        if df == 0:
            continue
        idf = math.log((num_docs + 1) / (df + 1)) + 1.0
        w_qt = idf            # simple query weight
        query_norm_sq += w_qt * w_qt

        for doc_id, tf in postings:
            if doc_id not in candidate_docs:
                continue
            w_dt = tf * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + w_qt * w_dt
            doc_norm_sq[doc_id] = doc_norm_sq.get(doc_id, 0.0) + w_dt * w_dt

    if not scores:
        return []

    query_norm = math.sqrt(query_norm_sq) or 1.0
    for doc_id in list(scores.keys()):
        doc_norm = math.sqrt(doc_norm_sq.get(doc_id, 0.0)) or 1.0
        scores[doc_id] = scores[doc_id] / (query_norm * doc_norm)

    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


# --------- pretty-print results ---------

def print_results(raw_query, processed_terms, results, docid_to_url, elapsed_ms):
    print(f"\nQuery: {raw_query!r}  |  processed terms: {processed_terms}")
    print(f"Results computed in {elapsed_ms:.1f} ms\n")
    if not results:
        print("No results found.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        url = docid_to_url.get(doc_id, "<missing URL>")
        print(f"{rank}. [doc {doc_id}] score={score:.4f}")
        print(f"   {url}")


# --------- main loop ---------

def main():
    parser = argparse.ArgumentParser(description="Developer search engine (Milestone 3)")
    parser.add_argument("--index-folder", type=Path, required=True,
                        help="Folder with index_merged.jsonl and doc_table.tsv")
    parser.add_argument("--index-file-name", type=str, default="index_merged.jsonl")
    parser.add_argument("--doc-table-name", type=str, default="doc_table.tsv")
    parser.add_argument("--top-k", type=int, default=5)
    args = parser.parse_args()

    index_path = args.index_folder / args.index_file_name
    doc_table_path = args.index_folder / args.doc_table_name

    print("Loading document table...")
    docid_to_url = load_doc_table(doc_table_path)
    num_docs = len(docid_to_url)
    print(f"Loaded {num_docs} documents.")

    lexicon_path = index_path.with_suffix(".lexicon.json")
    print("Building or loading lexicon (term → file offset)...")
    term_offsets = load_lexicon(index_path, lexicon_path)
    print(f"Lexicon loaded for {len(term_offsets)} terms.\n")

    print("=== Ranked Search Interface (tf-idf cosine, AND with OR fallback) ===")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() == "exit":
            break

        query_terms = preprocess_query(raw)
        t0 = time.perf_counter()
        results = ranked_search(query_terms, index_path, term_offsets, num_docs, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print_results(raw, query_terms, results, docid_to_url, elapsed_ms)
        print()

if __name__ == "__main__":
    main()
