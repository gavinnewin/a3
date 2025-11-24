import argparse
import json
import re
import collections
from pathlib import Path
from bs4 import BeautifulSoup
from collections import Counter
import warnings

warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

# -------------------- TOKENIZATION / STEMMING --------------------

token_regex = re.compile(r"[A-Za-z0-9]+")

try:
    from nltk.stem import PorterStemmer
    stemmer = PorterStemmer()
except Exception:
    stemmer = None


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


def tokenize_text(text):
    return [m.group(0).lower() for m in token_regex.finditer(text)]


# -------------------- IMPORTANT WORD WEIGHTING --------------------

def extract_weighted_term_counts(html_content):
    """
    Give extra weight to:
      - title: 3x
      - h1,h2,h3: 2x
      - bold/strong: 2x
      - body text: 1x
    """
    try:
        soup = BeautifulSoup(html_content or "", "lxml")
    except Exception:
        soup = BeautifulSoup(html_content or "", "html.parser")

    # remove scripts/styles/noscript
    for tag in soup(["script", "style", "noscript"]):
        tag.extract()

    term_counts = Counter()

    def add_text(text, weight):
        if not text:
            return
        for tok in tokenize_text(text):
            term_counts[stem_token(tok)] += weight

    # title
    if soup.title:        add_text(soup.title.get_text(separator=" ", strip=True), weight=3)

    # headings
    for h in soup.find_all(["h1", "h2", "h3"]):
        add_text(h.get_text(separator=" ", strip=True), weight=2)

    # bold/strong
    for b in soup.find_all(["b", "strong"]):
        add_text(b.get_text(separator=" ", strip=True), weight=2)

    # full body
    body_text = soup.get_text(separator=" ", strip=True)
    add_text(body_text, weight=1)

    return term_counts


# -------------------- JSON / FILE HELPERS --------------------

def load_json_page(json_path):
    try:
        data = json.loads(json_path.read_text(encoding="utf-8", errors="ignore"))
        url = data.get("url", str(json_path))
        html = data.get("content", "")
        return url, html
    except Exception:
        return str(json_path), ""


def collect_json_files(folder):
    return list(folder.rglob("*.json"))


# -------------------- PARTIAL INDEX / MERGE --------------------

def write_partial_index(part_number, partial_index, output_folder):
    partial_path = output_folder / f"partial_{part_number:03d}.jsonl"
    with partial_path.open("w", encoding="utf-8") as file:
        for term in sorted(partial_index.keys()):
            postings = sorted(partial_index[term].items())
            file.write(json.dumps({"term": term, "postings": postings}) + "\n")
    return partial_path


def merge_partial_indexes(part_paths, final_path):
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
                out.write(json.dumps({"term": smallest_term,
                                      "postings": sorted_postings}) + "\n")
    finally:
        for fh in files:
            fh.close()


# -------------------- BUILD EXTERNAL INDEX --------------------

def build_external_index(data_folder, output_folder, memory_limit_mb):
    memory_limit_bytes = memory_limit_mb * 1024 * 1024
    json_files = collect_json_files(data_folder)

    inverted_index = collections.defaultdict(lambda: collections.defaultdict(int))
    document_table = []
    partial_paths = []
    estimated_bytes = 0

    for doc_id, json_file in enumerate(json_files):
        url, html = load_json_page(json_file)
        term_counts = extract_weighted_term_counts(html)

        for term, count in term_counts.items():
            inverted_index[term][doc_id] += count
            estimated_bytes += len(term) + 12

        document_table.append((url, sum(term_counts.values())))

        if estimated_bytes >= memory_limit_bytes:
            part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
            partial_paths.append(part_path)
            inverted_index.clear()
            estimated_bytes = 0

    if inverted_index:
        part_path = write_partial_index(len(partial_paths), inverted_index, output_folder)
        partial_paths.append(part_path)

    final_path = output_folder / "index_merged.jsonl"
    merge_partial_indexes(partial_paths, final_path)

    return final_path, document_table


# -------------------- DOC TABLE + ANALYTICS --------------------

def save_doc_table(document_table, output_folder):
    with (output_folder / "doc_table.tsv").open("w", encoding="utf-8") as f:
        for doc_id, (url, length) in enumerate(document_table):
            f.write(f"{doc_id}\t{url}\t{length}\n")


def compute_analytics(index_file, document_table, output_folder):
    unique_terms = sum(1 for _ in open(index_file, encoding="utf-8"))
    index_kb = index_file.stat().st_size // 1024
    with (output_folder / "analytics.csv").open("w", encoding="utf-8") as f:
        f.write("doc_count,unique_terms,index_size_kb\n")
        f.write(f"{len(document_table)},{unique_terms},{index_kb}\n")


# -------------------- MAIN: BUILD INDEX --------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-folder", type=Path, required=True)
    parser.add_argument("--output-folder", type=Path, required=True)
    parser.add_argument("--memory-limit", type=int, default=256)
    args = parser.parse_args()

    output_folder = args.output_folder
    output_folder.mkdir(parents=True, exist_ok=True)

    index_file, document_table = build_external_index(
        args.data_folder, output_folder, args.memory_limit
    )

    save_doc_table(document_table, output_folder)
    compute_analytics(index_file, document_table, output_folder)

    print("\n=== Successful Index Build ===")


if __name__ == "__main__":
    main()
