import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


# SETTINGS
from pathlib import Path
import json

ROOT = Path(__file__).resolve().parents[1]
LABELS = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]
SPLITS = ("train", "validation", "test")
EXPERIMENTS = {
    "binary_evidence_classical": ("binary", True, "classical"),
    "binary_evidence_bert": ("binary", True, "bert"),
}


def configuration(path=None):
    path = Path(path) if path else ROOT / "config/dataset_and_training_settings.json"
    config = json.loads(path.read_text(encoding="utf-8-sig"))
    expected = {"source": "train.jsonl", "split_policy": "stratified_70_15_15",
                "train_fraction": 0.7, "validation_fraction": 0.15, "test_fraction": 0.15,
                "questionable_duplicates": "exclude_group", "conflicting_labels": "exclude_group"}
    if config.get("preparation") != expected:
        raise ValueError("Use the agreed train.jsonl 70/15/15 automatic-cleaning configuration.")
    b = config["bert"]
    if b["max_length"] != 512:
        raise ValueError("The agreed BERT maximum length is 512.")
    if b["batch_size_claims"] < 1 or b["gradient_accumulation"] < 1:
        raise ValueError("Batch size and accumulation must be positive.")
    if b["scheduler"] != "constant" or b["precision"] != "fp32":
        raise ValueError("This implementation supports constant LR and fp32 only.")
    if b["resume_checkpoint_every_steps"] < 1:
        raise ValueError("Resume checkpoint interval must be positive")
    return config


def labels_for(experiment):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown model")
    return LABELS[:2]


# IO
from pathlib import Path
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import tempfile


def sha256(path):
    with Path(path).open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def object_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(value, f, indent=2, ensure_ascii=False, allow_nan=False)
            f.write("\n")
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as f:
        for number, line in enumerate(f, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{number}: invalid JSON") from e
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{number}: expected an object")
            yield row


def write_csv(path, rows, columns):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def environment():
    result = {"python": platform.python_version(), "platform": platform.platform()}
    for name in ["torch", "transformers", "scikit-learn", "numpy", "pandas", "scipy", "joblib"]:
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def tree_hashes(directory):
    directory = Path(directory)
    return {p.relative_to(directory).as_posix(): sha256(p)
            for p in sorted(directory.rglob("*")) if p.is_file()}


def code_hashes(root):
    root = Path(root)
    return {p.relative_to(root).as_posix(): sha256(p)
            for folder in [Path(__file__).resolve().parent.name]
            for p in sorted((root / folder).glob("*.py"))}


# PREPARATION
# Pure cleaning and membership functions. no fitting or network access.
from collections import Counter, defaultdict
import unicodedata



def normal_key(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def conservative_key(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def clean_records(records, source):
    clean, excluded, fatal, notes = [], [], [], []
    seen = set()
    for line, original in enumerate(records, 1):
        row = dict(original)
        identity = {"id": row.get("id"), "source": source, "line": line}
        rid = row.get("id")
        if type(rid) is not int or rid < 0 or rid in seen:
            fatal.append({**identity, "reason": "missing_invalid_or_duplicate_id"})
            continue
        seen.add(rid)
        label = row.get("label")
        label = label.strip() if isinstance(label, str) else label
        if label not in LABELS:
            fatal.append({**identity, "reason": "unexpected_label", "value": label})
            continue
        claim = row.get("claim")
        if not isinstance(claim, str) or not claim.strip():
            excluded.append({**identity, "reason": "invalid_claim"})
            continue
        if not isinstance(row.get("evidence"), list):
            notes.append({**identity, "reason": "invalid_evidence_structure"})
        expected = "NOT VERIFIABLE" if label == LABELS[2] else "VERIFIABLE"
        if "verifiable" in row and row["verifiable"] != expected:
            notes.append({**identity, "reason": "verifiable_label_mismatch"})
        row.update(id=rid, claim=claim.strip(), label=label, source=source)
        clean.append(row)
    return clean, excluded, fatal, notes


def deduplicate(rows, token_keys):
    # Clean connected duplicate groups before assigning the custom splits.
    if len(rows) != len(token_keys):
        raise ValueError("Missing full-token duplicate keys")
    parents = list(range(len(rows)))

    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i

    links = []
    for rule, keys in [("exact", [r["claim"] for r in rows]),
                       ("normalised", [normal_key(r["claim"]) for r in rows]),
                       ("full_bert_tokens", token_keys)]:
        first = {}
        for i, key in enumerate(keys):
            if key in first:
                other = first[key]
                a, b = find(i), find(other)
                parents[max(a, b)] = min(a, b)
                links.append((i, other, rule))
            else:
                first[key] = i
    groups = defaultdict(list)
    for i in range(len(rows)):
        groups[find(i)].append(i)
    rules = defaultdict(set)
    for i, _, rule in links:
        rules[find(i)].add(rule)
    retained, manifest, questionable = [], [], []
    for root, indices in groups.items():
        members = [rows[i] for i in indices]
        ids = sorted(r["id"] for r in members)
        group_id = object_hash(ids)[:20]
        ambiguous = len({conservative_key(r["claim"]) for r in members}) > 1
        if ambiguous:
            questionable.append({"group_id": group_id, "action": "exclude_group",
                            "rules": sorted(rules[root]),
                            "records": [{k: r[k] for k in ("id", "claim", "label")} for r in members]})
        conflict = len({r["label"] for r in members}) > 1
        reason = "conflicting_labels" if conflict else "questionable_duplicate" if ambiguous else None
        winner = min(members, key=lambda r: r["id"])
        if reason is None:
            winner = dict(winner)
            # All records come from train.jsonl before creating the custom split.
            winner["annotation_records"] = [
                {"id": r["id"], "evidence": r.get("evidence")}
                for r in sorted(members, key=lambda r: r["id"])]
            winner["duplicate_group_id"] = group_id
            retained.append(winner)
        for r in members:
            keep = reason is None and r["id"] == winner["id"]
            manifest.append({"id": r["id"], "source": r["source"], "source_partition": "original_train",
                             "split": None, "duplicate_group_id": group_id,
                             "status": "retained" if keep else "excluded",
                             "reason": reason or ("retained" if keep else "duplicate"),
                             "retained_id": winner["id"] if reason is None else None,
                             "questionable": ambiguous, "match_rules": sorted(rules[root])})
    return sorted(retained, key=lambda r: r["id"]), manifest, questionable


def assign_splits(rows, seed=42):
    # Seeded stratified 70/15/15. stable if the input file order changes.
    from sklearn.model_selection import train_test_split
    ordered = sorted(rows, key=lambda r: r["id"])
    try:
        train, remaining = train_test_split(ordered, test_size=0.30, random_state=seed,
                                           stratify=[r["label"] for r in ordered])
        validation, test = train_test_split(remaining, test_size=0.5, random_state=seed,
                                            stratify=[r["label"] for r in remaining])
    except ValueError as exc:
        raise ValueError("Cannot create stratified 70/15/15 splits. check retained class counts") from exc
    return sorted([{**r, "split": split} for split, group in zip(SPLITS, (train, validation, test))
                   for r in group], key=lambda r: r["id"])


def verify_partitions(rows, labels):
    ids, groups, norm = set(), set(), set()
    stats = {}
    for split in SPLITS:
        subset = [r for r in rows if r["split"] == split]
        if not subset or set(r["label"] for r in subset) != set(labels):
            raise ValueError(f"{split}: expected every class and nonempty data")
        for r in subset:
            group = r["duplicate_group_id"]
            key = normal_key(r["claim"])
            if r["id"] in ids or group in groups or key in norm:
                raise ValueError("Retained record/duplicate overlap")
            ids.add(r["id"])
            groups.add(group)
            norm.add(key)
        stats[split] = {"records": len(subset), "class_counts": dict(Counter(r["label"] for r in subset))}
    return stats


# DIRECT DATASET FILES
import argparse

SOURCE_COLUMNS = ["id", "claim", "label", "annotation_records", "tokenizer_revision"]
EVIDENCE_COLUMNS = SOURCE_COLUMNS + ["evidence", "evidence_sources"]


def read_split_file(split, root=ROOT, require_evidence=False):
    if split not in SPLITS:
        raise ValueError("Invalid split")
    path = Path(root) / "Dataset/Processed" / f"{split}.csv"
    # Complete annotation/evidence fields can exceed CSV's default 128 KiB limit.
    # Find the largest supported limit, including on Windows Python versions.
    field_limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(field_limit)
            break
        except OverflowError:
            field_limit //= 10
    with path.open(encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        columns = reader.fieldnames
        if columns not in (SOURCE_COLUMNS, EVIDENCE_COLUMNS):
            raise ValueError(f"Unexpected columns in {path}. run prepare_dataset.py")
        if require_evidence and columns != EVIDENCE_COLUMNS:
            raise ValueError("Evidence preparation incomplete. run prepare_evidence.py first")
        rows = list(reader)
    ids = set()
    for row in rows:
        row["id"] = int(row["id"])
        if row["id"] in ids or not row["claim"].strip() or row["label"] not in LABELS[:2]:
            raise ValueError(f"Invalid prepared record {row['id']} in {path}")
        ids.add(row["id"])
        if not row["tokenizer_revision"] or not isinstance(json.loads(row["annotation_records"]), list):
            raise ValueError(f"Missing preparation references for {row['id']}")
        if require_evidence and not row["evidence"].strip():
            raise ValueError(f"Missing evidence for {row['id']}")
    if set(row["label"] for row in rows) != set(LABELS[:2]):
        raise ValueError(f"{path}: expected both classes and nonempty data")
    return rows


def load_split(experiment, split, root=ROOT):
    labels_for(experiment)
    return read_split_file(split, root, require_evidence=True)


def prepared_manifest(root=ROOT):
    # Compute dataset identity in memory. do not create a manifest file.
    all_rows, revisions = [], set()
    for split in SPLITS:
        rows = read_split_file(split, root, require_evidence=True)
        all_rows.extend({**row, "split": split, "duplicate_group_id": row["id"]} for row in rows)
        revisions.update(row["tokenizer_revision"] for row in rows)
    verify_partitions(all_rows, LABELS[:2])
    if len(revisions) != 1:
        raise ValueError("Dataset files contain different tokenizer revisions. rerun preparation")
    return {"policy": "stratified_70_15_15_from_cleaned_train", "model_revision": next(iter(revisions)),
            "files": {f"{split}.csv": sha256(Path(root) / "Dataset/Processed" / f"{split}.csv")
                      for split in SPLITS}}


def preparation_identity(ready):
    return object_hash(ready)


def prepared_tokenizer(root=ROOT):
    from transformers import AutoTokenizer
    revisions = {row["tokenizer_revision"] for split in SPLITS for row in read_split_file(split, root)}
    if len(revisions) != 1:
        raise ValueError("Dataset files contain different tokenizer revisions")
    revision = next(iter(revisions))
    path = Path(root) / "Dataset/Cache" / ("tokenizer-" + revision)
    return AutoTokenizer.from_pretrained(path, local_files_only=True), {"model_revision": revision}


def replace_splits(rows, columns, root=ROOT):
    # Validate first, then replace the three CSVs without permanent sidecar files.
    folder = Path(root) / "Dataset/Processed"
    folder.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="preparing-", dir=folder) as staging:
        for split in SPLITS:
            write_csv(Path(staging) / f"{split}.csv", [r for r in rows if r["split"] == split], columns)
        for split in SPLITS:
            os.replace(Path(staging) / f"{split}.csv", folder / f"{split}.csv")
            print(f"Saved (replaced if present): {folder / (split + '.csv')}")


def print_distribution(rows, labels=LABELS):
    counts = Counter(r.get("label") for r in rows)
    for label in labels:
        count = counts[label]
        print(f"  {label}: {count:,} ({100 * count / len(rows) if rows else 0:.2f}%)")


def print_splits(rows):
    print("\nDATASET SPLITS")
    print(f"{'Split':<12} {'Total':>10} {'Share':>9} {'SUPPORTS':>12} {'Class %':>9} {'REFUTES':>12} {'Class %':>9}")
    for split in SPLITS:
        part = [r for r in rows if r["split"] == split]
        counts = Counter(r["label"] for r in part)
        print(f"{split:<12} {len(part):>10,} {len(part)/len(rows):>9.2%} "
              f"{counts['SUPPORTS']:>12,} {counts['SUPPORTS']/len(part):>9.2%} "
              f"{counts['REFUTES']:>12,} {counts['REFUTES']/len(part):>9.2%}")
    print(f"Total retained: {len(rows):,}")
    print_distribution(rows, LABELS[:2])


def print_issues(title, issues):
    print(f"{title}: {len(issues):,}")
    for reason, count in Counter(r.get("reason", "incomplete_evidence_set") for r in issues).items():
        print(f"  {reason}: {count:,}")


def main():
    parser = argparse.ArgumentParser(description="Clean claims and save training, validation and test CSVs.")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = configuration(args.config)
    original = ROOT / "Dataset/Original/train.jsonl"
    print(f"Reading: {original}", flush=True)
    raw = list(read_jsonl(original))
    print(f"Original records: {len(raw):,}")
    print_distribution(raw)
    clean, excluded, fatal, notes = clean_records(raw, "train.jsonl")
    print_issues("Invalid claims excluded", excluded)
    print_issues("Fatal validation errors", fatal)
    print_issues("Annotation issues", notes)
    trimmed = sum(isinstance(r.get("claim"), str) and r["claim"] != r["claim"].strip() for r in raw)
    print(f"Claims with surrounding whitespace trimmed: {trimmed:,}")
    if fatal:
        for issue in fatal:
            print(issue)
        raise ValueError("Input validation failed. see terminal details. Existing outputs were not replaced.")
    from huggingface_hub import HfApi
    from transformers import AutoTokenizer
    revision = HfApi().model_info(config["tokenizer"], revision=config["model_revision"]).sha
    tokenizer = AutoTokenizer.from_pretrained(config["tokenizer"], revision=revision, use_fast=True)
    tokenizer.save_pretrained(ROOT / "Dataset/Cache" / ("tokenizer-" + revision))
    token_keys = []
    for start in range(0, len(clean), 2048):
        batch = clean[start:start + 2048]
        encoded = tokenizer([r["claim"] for r in batch], add_special_tokens=False,
                            truncation=False, padding=False, verbose=False)
        token_keys.extend(object_hash(ids) for ids in encoded["input_ids"])
        print(f"Duplicate token checking: {min(start + 2048, len(clean)):,}/{len(clean):,}", flush=True)
    retained, decisions, questionable = deduplicate(clean, token_keys)
    reasons = Counter(r["reason"] for r in decisions)
    print("\nCLEANING SUMMARY")
    print(f"Duplicate copies removed: {reasons['duplicate']:,}")
    print(f"Records excluded for conflicting labels: {reasons['conflicting_labels']:,}")
    print(f"Records excluded for questionable duplicate matches: {reasons['questionable_duplicate']:,}")
    print(f"Retained after cleaning across source classes: {len(retained):,}")
    print_distribution(retained)
    # Preserve the established assignments without writing an intermediate dataset.
    retained = assign_splits(retained, config["seed"])
    verify_partitions(retained, LABELS)
    keys = {r["id"]: key for r, key in zip(clean, token_keys)}
    if len({keys[r["id"]] for r in retained}) != len(retained):
        raise ValueError("Full-token duplicate survived cleaning")
    rows = [{**r, "annotation_records": json.dumps(r["annotation_records"], ensure_ascii=False),
             "tokenizer_revision": revision} for r in retained if r["label"] in LABELS[:2]]
    verify_partitions(rows, LABELS[:2])
    print(f"\nStratified 70/15/15. seed {config['seed']}. Assignments are made before excluding NEI.")
    print("Overlap checks passed: no repeated IDs, normalised claims, duplicate groups or full token keys.")
    print_splits(rows)
    print("\nSAMPLE RETAINED CLAIMS")
    for label in LABELS[:2]:
        print(label)
        for r in [r for r in rows if r["label"] == label][:3]:
            print(f"  ID {r['id']} ({r['split']}): {r['claim']}")
    replace_splits(rows, SOURCE_COLUMNS, ROOT)
    print('\nNext: python "Python Scripts/prepare_evidence.py"')


if __name__ == "__main__":
    main()
