import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


# COMMAND LINE WORKFLOW
# Evaluate saved classifiers. completed runs are locked automatically after training.
import argparse
from pathlib import Path
import tempfile
import time

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_dataset import prepared_manifest, preparation_identity, load_split
from prepare_dataset import read_json, write_json, sha256, tree_hashes
from train_classical import save_results
from train_classical import locked_run, results_directory
from prepare_dataset import EXPERIMENTS, labels_for


def print_metrics(name, split, metrics, output):
    print("\n" + "=" * 72)
    print(f"{name}: {split} results ({metrics['records']:,} claims)", flush=True)
    print('Confusion matrix: rows = actual, columns = predicted')
    width = max(16, max(map(len, metrics['labels'])) + 2)
    print(''.ljust(width) + ''.join(label.rjust(width) for label in metrics['labels']))
    for label, row in zip(metrics['labels'], metrics['confusion_matrix']):
        print(label.ljust(width) + ''.join(str(value).rjust(width) for value in row))
    print("\nPer-class metrics")
    print(f"Accuracy: {metrics['accuracy']:.4%} | Macro F1: {metrics['macro_f1']:.6f}")
    print(f"{'Class':<20} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Count':>8}")
    for label in metrics['labels']:
        row = metrics['classification_report'][label]
        print(f"{label:<20} {row['precision']:>10.4f} {row['recall']:>10.4f} {row['f1-score']:>10.4f} {int(row['support']):>8}")
    print(f"Saved results: {output}\n", flush=True)


def progress_batches(loader):
    completed = 0
    total = len(loader.dataset)
    started = time.monotonic()
    for index, batch in enumerate(loader, 1):
        yield batch
        completed += len(batch['ids'])
        if index % 100 == 0 or completed == total:
            print(f"Inference: {completed:,}/{total:,} claims ({completed / total:.1%}), "
                  f"elapsed {time.monotonic() - started:.0f}s", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=list(EXPERIMENTS) + ["all"], default="all")
    parser.add_argument("--split", choices=["test"], default="test")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    args = parser.parse_args()
    ready = prepared_manifest(ROOT)
    identity = preparation_identity(ready)
    names = list(EXPERIMENTS) if args.experiment == "all" else [args.experiment]
    for number, name in enumerate(names, 1):
        print(f"\n[{number}/{len(names)}] Evaluating {name} on {args.split}", flush=True)
        model_dir = locked_run(name, ROOT)
        run = read_json(model_dir / "configuration.json")
        summary = read_json(model_dir / "summary.json")
        if run["prepared_identity"] != identity or summary["status"] != "complete":
            raise ValueError("Run is incomplete or belongs to another prepared dataset")
        output = results_directory(name, model_dir.name, ROOT) / args.split
        if output.exists():
            previous = read_json(output / "metrics.json").get("evaluation")
            if previous is None and (output / "evaluation.json").exists():
                previous = read_json(output / "evaluation.json")
            if previous is None:
                raise ValueError("Existing test results lack evaluation identity")
            if previous["prepared_identity"] != identity or previous["model_files"] != tree_hashes(model_dir):
                raise ValueError(f"Existing test evaluation does not match {name}")
            print(f"Skipping {name}: test output already exists. preserve the original evaluation.")
            print_metrics(name, args.split, read_json(output / 'metrics.json'), output)
            continue
        labels = labels_for(name)
        rows = load_split(name, args.split, ROOT)
        print(f"Run: {model_dir.name} | Claims: {len(rows):,}", flush=True)
        if EXPERIMENTS[name][2] == "classical":
            import joblib
            model = joblib.load(model_dir / "model.joblib")
            print('Device: CPU | Running classical inference...', flush=True)
            values = model.probabilities(rows, labels)
        else:
            import torch
            from torch.utils.data import DataLoader
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
            from train_transformer import ClaimDataset, ClaimCollator, probabilities
            checkpoint = model_dir / "checkpoint"
            tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
            model = AutoModelForSequenceClassification.from_pretrained(checkpoint, local_files_only=True,
                                                                        attn_implementation="eager")
            if [model.config.id2label[i] for i in range(len(labels))] != labels:
                raise ValueError("Checkpoint label mapping mismatch")
            cfg = run["configuration"]["bert"]
            device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
            model.to(device)
            print(f'Device: {device} | Encoding evaluation claims...', flush=True)
            dataset = ClaimDataset(rows, tokenizer, labels, EXPERIMENTS[name][1], cfg["max_length"])
            loader = DataLoader(dataset, batch_size=cfg["batch_size_claims"], shuffle=False,
                                num_workers=0, collate_fn=ClaimCollator(tokenizer))
            print('Running BERT inference...', flush=True)
            values, ids = probabilities(model, progress_batches(loader), device, cfg["chunk_forward_batch_size"])
            if ids != [r["id"] for r in rows]:
                raise AssertionError("Evaluation order changed")
        # An interrupted evaluation must not appear as a completed test result.
        output.parent.mkdir(parents=True, exist_ok=True)
        target = Path(tempfile.mkdtemp(prefix="evaluation-", dir=output.parent))
        metrics = save_results(target, rows, values, labels)
        metrics["evaluation"] = {"experiment": name, "run_id": model_dir.name, "split": args.split,
                   "prepared_identity": identity, "model_files": tree_hashes(model_dir),
                   "lock_sha256": sha256(model_dir / "LOCKED.json")}
        write_json(target / "metrics.json", metrics)
        target.replace(output)
        print_metrics(name, args.split, metrics, output)
        del model
        if EXPERIMENTS[name][2] == "bert" and torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
