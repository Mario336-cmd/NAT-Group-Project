import argparse
import json
import random
import sqlite3
import sys
from contextlib import closing
from pathlib import Path
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "Python Scripts"))

CORPUS_DIR = ROOT / "Dataset/Index"
WIKI_DB = ROOT / "Dataset/Cache/fever_wiki_all_sets.sqlite3"
PASSAGES = CORPUS_DIR / "passages.jsonl"

DEMO_CLAIMS = [
    "Nikolaj Coster-Waldau worked with the Fox Broadcasting Company.",
    "Purple elephants run the Danish parliament.",
]

def pick_distractors(count, exclude, seed=42):
    # Random Wikipedia pages no claim cites, so retrievers must skip irrelevant text.
    with closing(sqlite3.connect(WIKI_DB.resolve().as_uri() + "?mode=ro", uri=True)) as con:
        last = con.execute("SELECT MAX(rowid) FROM pages").fetchone()[0]
        chosen = set()
        # sample row numbers instead of loading all ~5.4M page names into memory
        for rowid in random.Random(seed).sample(range(1, last + 1), min(last, count + len(exclude))):
            row = con.execute("SELECT page_id FROM pages WHERE rowid=?", (rowid,)).fetchone()
            if row and row[0] not in exclude:
                chosen.add(row[0])
                if len(chosen) == count:
                    break
    return sorted(chosen)


def build_corpus(max_pages=None, distractors=10000):
    # Collect every Wikipedia page your claims use, add random distractor pages, save all their sentences.
    from prepare_dataset import read_split_file, SPLITS
    from prepare_evidence import EvidenceStore, clean_text

    pages = set()
    for split in SPLITS:
        for row in read_split_file(split, ROOT, require_evidence=True):
            for group in json.loads(row["evidence_sources"]):
                for s in group["sources"]:
                    pages.add(s["page"])
    pages = sorted(pages)
    if max_pages:
        pages = pages[:max_pages]
    extra = pick_distractors(distractors, set(pages))
    extra_set = set(extra)

    store = EvidenceStore(WIKI_DB)
    CORPUS_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    try:
        with PASSAGES.open("w", encoding="utf-8") as f:
            for page in pages + extra:
                title = clean_text(page.replace("_", " "))
                for sid, text in store.sentences(page).items():
                    f.write(json.dumps({
                        "pid": count, "page": page, "sentence_id": sid, "text": text,
                        "rendered": f"{title}: {text}",
                        "url": "https://en.wikipedia.org/wiki/" + quote(page, safe="_()"),
                        "distractor": page in extra_set,
                    }, ensure_ascii=False) + "\n")
                    count += 1
    finally:
        store.close()
    print(f"Corpus built: {len(pages):,} cited pages + {len(extra):,} distractor pages, "
          f"{count:,} sentences -> {PASSAGES}")


def load_corpus():
    with PASSAGES.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def demo(retriever, k=3):
    for claim, hits in zip(DEMO_CLAIMS, retriever.search(DEMO_CLAIMS, k=k)):
        print(f"\n[{retriever.name}] {claim}")
        for h in hits:
            print(f"  {h[retriever.threshold_key]:8.3f}  {h['rendered'][:100]}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Build the sentence corpus")
    p.add_argument("--max-pages", type=int, help="small test corpus, e.g. 2000")
    p.add_argument("--distractors", type=int, default=10000, help="random uncited pages to add")
    args = p.parse_args()
    build_corpus(args.max_pages, args.distractors)
