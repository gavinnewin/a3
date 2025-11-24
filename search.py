import argparse
import json
import math
import re
import time
from pathlib import Path

# -------------------- TOKENIZATION / STEMMING --------------------

token_regex = re.compile(r"[A-Za-z0-9]+")

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None


def tokenize_text(text):
    return [m.group(0).lower() for m in token_regex.finditer(text or "")]


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


def preprocess_query(raw_query):
    tokens = tokenize_text(raw_query)
    stems = [stem_token(t) for t in tokens if t]
    bigrams = []
    for i in range(len(stems) - 1):
        if stems[i] and stems[i + 1]:
            bigrams.append(f"{stems[i]}_{stems[i+1]}")
    return stems, bigrams


# -------------------- DOC TABLE --------------------

def load_doc_table(doc_table_path: Path):
    docid_to_url = {}
    with doc_table_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            try:
                doc_id = int(parts[0])
            except ValueError:
                continue
            url = parts[1]
            docid_to_url[doc_id] = url
    return docid_to_url


# -------------------- LEXICON (INDEX THE INDEX) --------------------

def build_lexicon(index_path: Path, lexicon_path: Path):
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


def load_lexicon(index_path: Path, lexicon_path: Path):
    if lexicon_path.exists():
        with lexicon_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    return build_lexicon(index_path, lexicon_path)


# -------------------- LOAD POSTINGS FOR SELECTED TERMS --------------------

def load_postings_for_terms(index_path: Path, term_offsets, terms):
    postings = {}
    with index_path.open("r", encoding="utf-8") as f:
        for term in terms:
            offset = term_offsets.get(term)
            if offset is None:
                postings[term] = []
                continue
            f.seek(offset)
            line = f.readline()
            if not line:
                postings[term] = []
                continue
            obj = json.loads(line)
            postings[term] = [(int(doc), int(tf)) for doc, tf in obj["postings"]]
    return postings


# -------------------- RANKED SEARCH (TF-IDF COSINE + BIGRAM BOOST) --------------------

def ranked_search(all_terms, bigram_terms, index_path: Path, term_offsets, num_docs, top_k=5):

    # Load postings for every term in the query (unigrams + bigrams)
    postings_dict = load_postings_for_terms(index_path, term_offsets, all_terms)

    # --- 1) Use ONLY unigrams for AND/OR candidate selection ---
    unigram_terms = [t for t in all_terms if t not in bigram_terms]
    base_terms = unigram_terms or all_terms  # fallback if somehow no unigrams

    # Collect postings for base_terms (ignore terms that don't appear at all)
    nonempty_postings = [postings_dict[t] for t in base_terms if postings_dict.get(t)]
    if not nonempty_postings:
        return []

    # AND intersection over base_terms
    candidate_docs = set(doc_id for doc_id, _ in nonempty_postings[0])
    for plist in nonempty_postings[1:]:
        candidate_docs &= {doc_id for doc_id, _ in plist}

    # OR fallback: if AND is empty, use union over base_terms
    if not candidate_docs:
        candidate_docs = set()
        for plist in nonempty_postings:
            candidate_docs |= {doc_id for doc_id, _ in plist}
        if not candidate_docs:
            return []

    # --- 2) tf-idf + cosine scoring (with bigram boost) ---
    scores = {}
    doc_norm_sq = {}
    query_norm_sq = 0.0

    for term in all_terms:
        postings = postings_dict.get(term)
        if not postings:
            continue

        df = len(postings)
        if df == 0:
            continue

        idf = math.log((num_docs + 1) / (df + 1)) + 1.0

        # Bigram boost: phrase matches count more in the query vector
        if term in bigram_terms:
            w_qt = idf * 2.0
        else:
            w_qt = idf

        query_norm_sq += w_qt * w_qt

        for doc_id, tf in postings:
            if doc_id not in candidate_docs:
                continue
            w_dt = tf * idf
            scores[doc_id] = scores.get(doc_id, 0.0) + w_qt * w_dt
            doc_norm_sq[doc_id] = doc_norm_sq.get(doc_id, 0.0) + w_dt * w_dt

    if not scores:
        return []

    # Cosine normalization
    query_norm = math.sqrt(query_norm_sq) or 1.0
    for doc_id in list(scores.keys()):
        doc_norm = math.sqrt(doc_norm_sq.get(doc_id, 0.0)) or 1.0
        scores[doc_id] = scores[doc_id] / (query_norm * doc_norm)

    # Sort by score desc, doc_id asc
    ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:top_k]


# -------------------- PRETTY PRINT --------------------

def print_results(raw_query, stems, bigrams, results, docid_to_url, elapsed_ms):
    print(f"\nQuery: {raw_query!r}")
    print(f"Processed stems  : {stems}")
    print(f"Processed bigrams: {bigrams}")
    print(f"Results in {elapsed_ms:.1f} ms\n")
    if not results:
        print("No results found.")
        return
    for rank, (doc_id, score) in enumerate(results, start=1):
        url = docid_to_url.get(doc_id, "<missing URL>")
        print(f"{rank}. [doc {doc_id}] score={score:.4f}")
        print(f"   {url}")


# -------------------- MAIN LOOP --------------------

def main():
    parser = argparse.ArgumentParser(description="ICS Search Engine (developer, tf-idf + bigrams)")
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

    print("=== Ranked Search Interface (tf-idf cosine, AND with OR fallback, bigrams) ===")
    print("Type a query, or 'exit' to quit.\n")

    while True:
        try:
            raw = input("query> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw or raw.lower() == "exit":
            break

        stems, bigrams = preprocess_query(raw)
        all_terms = stems + bigrams

        t0 = time.perf_counter()
        results = ranked_search(all_terms, set(bigrams), index_path, term_offsets, num_docs, top_k=args.top_k)
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        print_results(raw, stems, bigrams, results, docid_to_url, elapsed_ms)
        print()


if __name__ == "__main__":
    main()
