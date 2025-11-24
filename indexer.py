import argparse
import json
import re
import collections
import hashlib
import time
from pathlib import Path

from bs4 import BeautifulSoup, XMLParsedAsHTMLWarning
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

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


# -------------------- JSON / HTML HELPERS --------------------

def load_json_page(json_path: Path):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        url = data.get("url", str(json_path))
        html = data.get("content", "")
        return url, html
    except Exception:
        return str(json_path), ""


def collect_json_files(folder: Path):
    return list(folder.rglob("*.json"))


def extract_visible_text(html_content: str) -> str:
    try:
        soup = BeautifulSoup(html_content or "", "lxml")
    except Exception:
        soup = BeautifulSoup(html_content or "", "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()
    return soup.get_text(separator=" ", strip=True)


# -------------------- IMPORTANT WORD WEIGHTING + BIGRAMS --------------------

def extract_weighted_term_counts(html_content: str) -> collections.Counter:
    """
    Give extra weight to:
      - title: 3x
      - h1,h2,h3: 2x
      - bold/strong: 2x
      - body text: 1x
    Also builds 2-gram tokens from the full visible text, stored as "w1_w2".
    """
    try:
        soup = BeautifulSoup(html_content or "", "lxml")
    except Exception:
        soup = BeautifulSoup(html_content or "", "html.parser")

    term_counts: collections.Counter = collections.Counter()

    def add_text(text: str, weight: int):
        for tok in tokenize_text(text):
            stem = stem_token(tok)
            if stem:
                term_counts[stem] += weight

    # title
    if soup.title and soup.title.string:
        add_text(soup.title.get_text(separator=" ", strip=True), weight=3)

    # headings
    for h in soup.find_all(["h1", "h2", "h3"]):
        add_text(h.get_text(separator=" ", strip=True), weight=2)

    # bold / strong
    for b in soup.find_all(["b", "strong"]):
        add_text(b.get_text(separator=" ", strip=True), weight=2)

    # full body text unweighted
    body_text = soup.get_text(separator=" ", strip=True)
    add_text(body_text, weight=1)

    # 2-gram (bigram) tokens from body text
    body_tokens = tokenize_text(body_text)
    stemmed_tokens = [stem_token(t) for t in body_tokens]
    for i in range(len(stemmed_tokens) - 1):
        t1 = stemmed_tokens[i]
        t2 = stemmed_tokens[i + 1]
        if not t1 or not t2:
            continue
        bigram_term = f"{t1}_{t2}"
        term_counts[bigram_term] += 1  # phrase-level signal

    return term_counts


# -------------------- PARTIAL INDEX WRITING / MERGE --------------------

def write_partial_index(part_number, inverted_index, output_folder: Path) -> Path:
    part_path = output_folder / f"partial_{part_number:03d}.jsonl"
    with part_path.open("w", encoding="utf-8") as f:
        for term in sorted(inverted_index.keys()):
            postings = sorted(inverted_index[term].items())
            f.write(json.dumps({"term": term, "postings": postings}) + "\n")
    return part_path


def merge_partial_indexes(part_paths, final_path: Path):
    files = [p.open("r", encoding="utf-8") for p in part_paths]
    try:
        cursors = []
        for idx, fh in enumerate(files):
            line = fh.readline()
            if line:
                obj = json.loads(line)
                cursors.append((obj["term"], obj, idx))

        with final_path.open("w", encoding="utf-8") as out:
            while cursors:
                # smallest term across all partials
                cursors.sort(key=lambda x: x[0])
                smallest_term = cursors[0][0]
                combined = {}
                refill = []

                while cursors and cursors[0][0] == smallest_term:
                    _, entry, file_index = cursors.pop(0)
                    for doc_id, tf in entry["postings"]:
                        combined[doc_id] = combined.get(doc_id, 0) + tf

                    nxt = files[file_index].readline()
                    if nxt:
                        nxt_obj = json.loads(nxt)
                        refill.append((nxt_obj["term"], nxt_obj, file_index))

                cursors.extend(refill)
                sorted_postings = sorted(combined.items())
                out.write(
                    json.dumps({"term": smallest_term, "postings": sorted_postings})
                    + "\n"
                )
    finally:
        for fh in files:
            fh.close()


# -------------------- MAIN INDEX BUILD (WITH DUPLICATE REMOVAL) --------------------

def build_external_index(data_folder: Path, output_folder: Path, memory_limit_mb: int):
    memory_limit_bytes = memory_limit_mb * 1024 * 1024
    json_files = collect_json_files(data_folder)

    inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
    document_table = []
    partial_paths = []
    estimated_bytes = 0

    # For exact-duplicate detection
    seen_hashes = {}
    duplicates = []

    current_doc_id = 0

    for json_file in json_files:
        url, html = load_json_page(json_file)

        # exact duplicate detection based on visible text hash
        visible_text = extract_visible_text(html)
        content_bytes = visible_text.encode("utf-8", errors="ignore")
        content_hash = hashlib.md5(content_bytes).hexdigest()

        if content_hash in seen_hashes:
            canonical_url = seen_hashes[content_hash]
            duplicates.append((url, canonical_url))
            continue
        else:
            seen_hashes[content_hash] = url

        term_counts = extract_weighted_term_counts(html)
        doc_length = sum(term_counts.values())
        document_table.append((url, doc_length))

        for term, count in term_counts.items():
            inverted_index[term][current_doc_id] += count
            estimated_bytes += len(term) + 12

        current_doc_id += 1

        if estimated_bytes >= memory_limit_bytes:
            part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
            partial_paths.append(part_path)
            inverted_index.clear()
            estimated_bytes = 0

    if inverted_index:
        part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
        partial_paths.append(part_path)

    final_index_path = output_folder / "index_merged.jsonl"
    merge_partial_indexes(partial_paths, final_index_path)

    # write out duplicate report (exact duplicates)
    if duplicates:
        dup_path = output_folder / "duplicates.tsv"
        with dup_path.open("w", encoding="utf-8") as f:
            f.write("duplicate_url\tcanonical_url\n")
            for dup_url, canonical_url in duplicates:
                f.write(f"{dup_url}\t{canonical_url}\n")

    return final_index_path, document_table


# -------------------- DOC TABLE + ANALYTICS --------------------

def save_doc_table(document_table, output_folder: Path):
    with (output_folder / "doc_table.tsv").open("w", encoding="utf-8") as f:
        for doc_id, (url, length) in enumerate(document_table):
            f.write(f"{doc_id}\t{url}\t{length}\n")


def compute_analytics(index_file: Path, document_table, output_folder: Path):
    unique_terms = sum(1 for _ in index_file.open("r", encoding="utf-8"))
    index_kb = index_file.stat().st_size // 1024
    with (output_folder / "analytics.csv").open("w", encoding="utf-8") as f:
        f.write("doc_count,unique_terms,index_size_kb\n")
        f.write(f"{len(document_table)},{unique_terms},{index_kb}\n")


# -------------------- CLI ENTRY POINT --------------------

def main():
    parser = argparse.ArgumentParser(description="ICS Search Engine Indexer (developer option, with extras)")
    parser.add_argument("--data-folder", type=Path, required=True,
                        help="Root folder with per-domain JSON files")
    parser.add_argument("--output-folder", type=Path, required=True,
                        help="Where to write index_merged.jsonl and doc_table.tsv")
    parser.add_argument("--memory-limit", type=int, default=256,
                        help="Approximate memory limit (MB) for in-memory index before spilling")
    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    start_time = time.perf_counter()
    index_file, document_table = build_external_index(
        args.data_folder, output_folder, args.memory_limit
    )
    save_doc_table(document_table, output_folder)
    compute_analytics(index_file, document_table, output_folder)
    elapsed = time.perf_counter() - start_time

    print("\n=== Successful Index Build (with exact-duplicate removal + bigrams) ===")
    print(f"Indexed documents : {len(document_table)}")
    print(f"Index file        : {index_file}")
    print(f"Output folder     : {output_folder.resolve()}")
    print(f"Elapsed time      : {elapsed:.2f} seconds")


if __name__ == "__main__":
    main()
