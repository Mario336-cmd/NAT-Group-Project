import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


# CHUNKING
# Lossless BERT pair construction, independent of torch and model weights.


def pack_units(units, budget):
    if budget < 1:
        raise ValueError("Claim leaves no room for evidence within 512 tokens")
    chunks, current = [], []
    for tokens in units:
        if not tokens:
            continue
        if len(tokens) <= budget:
            if current and len(current) + len(tokens) > budget:
                chunks.append(current)
                current = []
            current.extend(tokens)
        else:
            if current:
                chunks.append(current)
                current = []
            for start in range(0, len(tokens), budget):
                part = tokens[start:start + budget]
                if len(part) == budget:
                    chunks.append(part)
                else:
                    current = part
    if current:
        chunks.append(current)
    if not chunks:
        raise ValueError("Empty evidence token sequence")
    return chunks


def encode_claim(tokenizer, claim, evidence=None, max_length=512):
    claim_ids = tokenizer.encode(claim, add_special_tokens=False, truncation=False)
    if not claim_ids:
        raise ValueError("Empty claim token sequence")
    cls_id, sep_id = tokenizer.cls_token_id, tokenizer.sep_token_id
    if cls_id is None or sep_id is None:
        raise ValueError("BERT input construction requires CLS and SEP tokens")
    special = 2 if evidence is None else 3
    if evidence is None:
        if len(claim_ids) + special > max_length:
            raise ValueError("Claim exceeds the input limit. review it, never silently truncate")
        parts = [None]
    else:
        # Preparation stores one rendered annotated sentence per newline.
        units = [tokenizer.encode(line, add_special_tokens=False, truncation=False)
                 for line in evidence.splitlines() if line.strip()]
        parts = pack_units(units, max_length - len(claim_ids) - special)
    encoded = []
    for part in parts:
        # Standard BERT layout, compatible with Transformers 4 and 5 tokenizers.
        input_ids = [cls_id] + claim_ids + [sep_id]
        type_ids = [0] * len(input_ids)
        if part is not None:
            input_ids += part + [sep_id]
            type_ids += [1] * (len(part) + 1)
        if len(input_ids) > max_length:
            raise AssertionError("Chunk exceeds configured token budget")
        encoded.append({"input_ids": input_ids, "token_type_ids": type_ids,
                        "attention_mask": [1] * len(input_ids)})
    return encoded


# EVIDENCE
# Snapshot lookup and complete gold-set resolution. no NEI retrieval.
from contextlib import closing
from functools import lru_cache
import json
from pathlib import Path
import sqlite3
import unicodedata
from urllib.parse import quote
import zipfile

from prepare_dataset import object_hash

INDEX_VERSION = "truecheck-all-sets-v1"
BRACKETS = {"-LRB-": "(", "-RRB-": ")", "-LSB-": "[", "-RSB-": "]", "-LCB-": "{", "-RCB-": "}"}


def clean_text(text):
    for marker, character in BRACKETS.items():
        text = text.replace(marker, character)
    return " ".join(text.split())


def parse_sentences(lines):
    result = {}
    for line in lines.splitlines():
        fields = line.split("\t")
        if len(fields) >= 2 and fields[0].isdigit():
            sid = int(fields[0])
            if sid in result:
                raise ValueError(f"Duplicate Wikipedia sentence ID: {sid}")
            text = clean_text(fields[1])
            if text:
                result[sid] = text
    return result


def build_index(archive_path, database_path):
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        members = sorted(i for i in archive.namelist() if i.endswith(".jsonl")
                         and "__MACOSX" not in i.split("/") and not Path(i).name.startswith("._"))
        fingerprint = object_hash([(m, archive.getinfo(m).CRC, archive.getinfo(m).file_size) for m in members])
        if not members:
            raise ValueError("No Wikipedia JSONL shards in archive")
        expected = {"archive_fingerprint": fingerprint, "index_version": INDEX_VERSION}
        if database_path.exists():
            with closing(sqlite3.connect(database_path)) as con:
                metadata = dict(con.execute("SELECT key,value FROM metadata"))
            if any(metadata.get(k) != v for k, v in expected.items()) or metadata.get("complete") != "yes":
                raise ValueError("Stale/incomplete Wikipedia index. choose a new cache path")
            return metadata
        staging = database_path.with_suffix(".building.sqlite3")
        with closing(sqlite3.connect(staging)) as con:
            con.execute("CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY,value TEXT)")
            metadata = dict(con.execute("SELECT key,value FROM metadata"))
            if metadata and any(metadata.get(k) != v for k, v in expected.items()):
                raise ValueError("Interrupted index belongs to a different source/version")
            con.executemany("INSERT OR REPLACE INTO metadata VALUES (?,?)", expected.items())
            con.execute("CREATE TABLE IF NOT EXISTS pages (page_id TEXT PRIMARY KEY,lines TEXT NOT NULL)")
            con.execute("CREATE TABLE IF NOT EXISTS completed (name TEXT PRIMARY KEY)")
            con.commit()
            completed = {row[0] for row in con.execute("SELECT name FROM completed")}
            for number, member in enumerate(members, 1):
                if member in completed:
                    continue
                print(f"Wikipedia shard {number}/{len(members)}: {member}", flush=True)
                with con, archive.open(member) as stream:
                    batch = []
                    for raw in stream:
                        row = json.loads(raw)
                        if not isinstance(row.get("id"), str) or not isinstance(row.get("lines"), str):
                            raise ValueError(f"Invalid page in {member}")
                        batch.append((unicodedata.normalize("NFC", row["id"]), row["lines"]))
                        if len(batch) == 1000:
                            con.executemany("INSERT INTO pages VALUES (?,?)", batch)
                            batch.clear()
                    con.executemany("INSERT INTO pages VALUES (?,?)", batch)
                    con.execute("INSERT INTO completed VALUES (?)", (member,))
            with con:
                con.execute("INSERT OR REPLACE INTO metadata VALUES ('complete','yes')")
                con.execute("INSERT OR REPLACE INTO metadata VALUES ('pages',?)",
                            (str(con.execute("SELECT COUNT(*) FROM pages").fetchone()[0]),))
            metadata = dict(con.execute("SELECT key,value FROM metadata"))
        staging.replace(database_path)
        return metadata


class EvidenceStore:
    def __init__(self, path):
        self.connection = sqlite3.connect(Path(path).resolve().as_uri() + "?mode=ro", uri=True)

    def close(self):
        self.sentences.cache_clear()
        self.connection.close()

    @lru_cache(maxsize=1024)
    def sentences(self, page):
        row = self.connection.execute("SELECT lines FROM pages WHERE page_id=?", (page,)).fetchone()
        return parse_sentences(row[0]) if row else {}

    def resolve(self, annotation_records):
        complete, issues = [], []
        for record in sorted(annotation_records, key=lambda r: r["id"]):
            sets = record.get("evidence")
            if not isinstance(sets, list):
                issues.append({"record_id": record["id"], "reason": "invalid_annotation_structure"})
                continue
            for index, evidence_set in enumerate(sets):
                sources, missing, seen = [], [], set()
                if not isinstance(evidence_set, list) or not evidence_set:
                    missing.append("empty_or_invalid_set")
                    evidence_set = []
                for item in evidence_set:
                    if not (isinstance(item, list) and len(item) == 4 and isinstance(item[2], str)
                            and type(item[3]) is int and item[3] >= 0):
                        missing.append("invalid_reference")
                        continue
                    page, sid = unicodedata.normalize("NFC", item[2]), item[3]
                    if (page, sid) in seen:
                        continue
                    seen.add((page, sid))
                    text = self.sentences(page).get(sid)
                    if text is None:
                        missing.append(f"{page}:{sid}")
                        continue
                    title = clean_text(page.replace("_", " "))
                    sources.append({"page": page, "sentence_id": sid, "text": text,
                                    "rendered": f"{title}: {text}",
                                    "url": "https://en.wikipedia.org/wiki/" + quote(page, safe="_()")})
                if missing or not sources:
                    issues.append({"record_id": record["id"], "set_index": index, "missing": missing})
                else:
                    complete.append({"record_id": record["id"], "set_index": index, "sources": sources})
        # Metadata keeps every reference and membership, even when rendered text repeats.
        sentences = list(dict.fromkeys(s["rendered"] for group in complete for s in group["sources"]))
        return {"sentences": sentences, "evidence": "\n".join(sentences),
                "complete_sets": complete, "issues": issues}


# COMMAND LINE WORKFLOW
import argparse
from collections import Counter
from prepare_dataset import (prepared_tokenizer, read_split_file, replace_splits,
                             print_splits, verify_partitions, SPLITS, LABELS, EVIDENCE_COLUMNS)


def main():
    argparse.ArgumentParser(description="Resolve annotated evidence and update the three dataset CSVs.").parse_args()
    rows = [{**r, "split": split, "duplicate_group_id": r["id"]}
            for split in SPLITS for r in read_split_file(split, ROOT)]
    verify_partitions(rows, LABELS[:2])
    print(f"Reading three saved dataset splits: {len(rows):,} claims")
    print("Classes: SUPPORTS and REFUTES. NEI was excluded during dataset preparation.")
    print_splits(rows)
    tokenizer, _ = prepared_tokenizer(ROOT)
    archive = ROOT / "Dataset/Original/wiki-pages.zip"
    path = ROOT / "Dataset/Cache/fever_wiki_all_sets.sqlite3"
    status = "Reusing" if path.exists() else "Resuming" if path.with_suffix('.building.sqlite3').exists() else "Building"
    print(f"\n{status} Wikipedia lookup index from {archive}", flush=True)
    index = build_index(archive, path)
    print(f"Indexed Wikipedia pages: {int(index['pages']):,}")
    kept, excluded, issues, lengths = [], [], [], []
    references, pages = set(), set()
    store = EvidenceStore(path)
    try:
        for number, row in enumerate(rows, 1):
            result = store.resolve(json.loads(row["annotation_records"]))
            issues.extend(result["issues"])
            if not result["sentences"]:
                excluded.append(row)
                print(f"Excluded ID {row['id']} ({row['split']}): no complete usable evidence set")
            else:
                kept.append({**row, "evidence": result["evidence"],
                             "evidence_sources": json.dumps(result["complete_sets"], ensure_ascii=False)})
                for group in result["complete_sets"]:
                    for source in group["sources"]:
                        references.add((source["page"], source["sentence_id"]))
                        pages.add(source["page"])
            if number % 1000 == 0 or number == len(rows):
                print(f"Resolving evidence: {number:,}/{len(rows):,}", flush=True)
    finally:
        store.close()
    print(f"\nIncomplete or invalid evidence annotation entries: {len(issues):,}")
    print(f"Claims excluded without a complete usable evidence set: {len(excluded):,}")
    for split in SPLITS:
        print(f"  {split}: {sum(r['split'] == split for r in excluded):,} excluded")
    print(f"Unique evidence sentence references: {len(references):,}")
    print(f"Source Wikipedia pages represented: {len(pages):,}")
    verify_partitions(kept, LABELS[:2])
    failures = []
    for number, row in enumerate(kept, 1):
        try:
            chunks = encode_claim(tokenizer, row["claim"], row["evidence"])
            total = (len(tokenizer.encode(row["claim"], add_special_tokens=False))
                     + tokenizer.num_special_tokens_to_add(pair=True)
                     + sum(len(tokenizer.encode(s, add_special_tokens=False)) for s in row["evidence"].splitlines()))
            lengths.append({"split": row["split"], "tokens": total, "chunks": len(chunks)})
        except ValueError as exc:
            failures.append(row["id"])
            print(f"Cannot encode claim ID {row['id']}: {exc}")
        if number % 1000 == 0 or number == len(kept):
            print(f"Checking input lengths: {number:,}/{len(kept):,}", flush=True)
    if failures:
        raise ValueError(f"{len(failures)} claims cannot be encoded. No evidence truncated. CSVs were not replaced.")
    print("\nBERT INPUT LENGTHS (512 tokens including claim and special tokens)")
    for split in SPLITS:
        part = [r for r in lengths if r["split"] == split]
        single = sum(r["chunks"] == 1 for r in part)
        print(f"{split}: {single:,}/{len(part):,} fit one input ({single/len(part):.2%}). "
              f"{len(part)-single:,} need multiple chunks. max chunks {max(r['chunks'] for r in part)}. "
              f"max full tokens {max(r['tokens'] for r in part):,}")
    print("All evidence is preserved. BERT averages chunk logits for one prediction per claim.")
    print_splits(kept)
    print("Overlap checks passed. every retained claim kept its original split assignment.")
    example = kept[0]
    print(f"\nFORMATTED EXAMPLE\nID: {example['id']}\nClaim: {example['claim']}\nClass: {example['label']}")
    print(f"Evidence:\n{example['evidence']}")
    replace_splits(kept, EVIDENCE_COLUMNS, ROOT)
    print("\nBoth models use these same three dataset files. Evidence references remain inside the CSVs.")


if __name__ == "__main__":
    main()
