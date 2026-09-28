import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Stable module name lets saved joblib models load in another Python process.
if __name__ == "__main__":
    sys.modules["train_classical"] = sys.modules[__name__]


# RUNS
from datetime import datetime, timezone
from pathlib import Path
import re
from uuid import uuid4

from prepare_dataset import read_json, write_json, environment, code_hashes, tree_hashes
from prepare_dataset import preparation_identity
from prepare_dataset import ROOT, EXPERIMENTS


def run_directory(experiment, root=ROOT, run_id=None):
    if experiment not in EXPERIMENTS:
        raise ValueError("Unknown experiment")
    root = Path(root)
    if run_id is None:
        run_id = read_json(root / "Models" / experiment / "latest.json")["run_id"]
    if not isinstance(run_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", run_id):
        raise ValueError("Invalid run ID")
    return root / "Models" / experiment / "runs" / run_id


def results_directory(experiment, run_id, root=ROOT):
    run_directory(experiment, root, run_id)  # Validate identity/path components.
    return Path(root) / "Evaluation" / experiment / "runs" / run_id


def begin_run(experiment, config, ready, root=ROOT):
    root = Path(root)
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S") + "_" + uuid4().hex[:8]
    destination = run_directory(experiment, root, run_id)
    destination.mkdir(parents=True, exist_ok=False)
    write_json(destination / "configuration.json", {
        "experiment": experiment, "run_id": run_id, "configuration": config,
        "prepared_identity": preparation_identity(ready), "environment": environment(),
        "code": code_hashes(root), "split_policy": ready["policy"]})
    write_json(destination.parent.parent / "latest.json", {"run_id": run_id})
    print(f"{experiment}: new run {run_id}", flush=True)
    return destination


def resume_run(experiment, run_id, config, ready, root=ROOT):
    root = Path(root)
    destination = run_directory(experiment, root, run_id)
    saved = read_json(destination / "configuration.json")
    if (destination / "summary.json").exists():
        raise ValueError("This run is already complete. Start a new run instead of resuming it.")
    if saved["configuration"] != config or saved["prepared_identity"] != preparation_identity(ready):
        raise ValueError("Resume requires the same configuration and prepared data")
    if saved["code"] != code_hashes(root) or saved["environment"] != environment():
        raise ValueError("Resume requires unchanged experiment code and package environment")
    if not (destination / "resume.pt").exists():
        raise FileNotFoundError("No saved training state yet. start a new run")
    write_json(destination.parent.parent / "latest.json", {"run_id": run_id})
    return destination


def locked_files(directory):
    return {name: digest for name, digest in tree_hashes(directory).items()
            if name != "LOCKED.json"}


def lock_run(experiment, run_id, root=ROOT):
    # Seal a complete run and publish its ID for evaluation automatically.
    root = Path(root)
    directory = run_directory(experiment, root, run_id)
    run = read_json(directory / "configuration.json")
    summary = read_json(directory / "summary.json")
    if summary["status"] != "complete":
        raise ValueError("Only complete training runs can be locked")
    if run["code"] != code_hashes(root):
        raise ValueError("Code changed during training")
    lock = {"run_id": run_id, "prepared_identity": run["prepared_identity"],
            "code": run["code"], "model_files": locked_files(directory)}
    path = directory / "LOCKED.json"
    if path.exists() and read_json(path) != lock:
        raise ValueError("Locked model changed")
    write_json(path, lock)
    write_json(root / "Models" / experiment / "completed.json", {"run_id": run_id})
    print(f"{experiment}: best validation model locked for evaluation", flush=True)


def locked_run(experiment, root=ROOT):
    root = Path(root)
    pointer = root / "Models" / experiment / "completed.json"
    if not pointer.exists():
        raise FileNotFoundError(f"{experiment}: no completed, locked training run")
    run_id = read_json(pointer)["run_id"]
    directory = run_directory(experiment, root, run_id)
    lock = read_json(directory / "LOCKED.json")
    if lock["run_id"] != run_id or locked_files(directory) != lock["model_files"]:
        raise ValueError(f"Locked model changed: {experiment}")
    if code_hashes(root) != lock["code"]:
        raise ValueError("Code changed after training. preserve the trained run's code for evaluation")
    return directory


# METRICS
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

from prepare_dataset import write_json, write_csv


def classification_metrics(truth, predicted, labels):
    report = classification_report(truth, predicted, labels=labels, output_dict=True, zero_division=0)
    cm = confusion_matrix(truth, predicted, labels=labels)
    rates = {}
    for i, label in enumerate(labels):
        tp = int(cm[i, i])
        fn = int(cm[i, :].sum() - tp)
        fp = int(cm[:, i].sum() - tp)
        tn = int(cm.sum() - tp - fn - fp)
        rates[label] = {"tp": tp, "fn": fn, "fp": fp, "tn": tn,
                        "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                        "false_negative_rate": fn / (fn + tp) if fn + tp else None}
    return {"records": len(truth), "accuracy": accuracy_score(truth, predicted),
            "macro_f1": f1_score(truth, predicted, labels=labels, average="macro", zero_division=0),
            "classification_report": report, "one_vs_rest_rates": rates,
            "labels": labels, "confusion_matrix": cm.tolist()}


def save_results(directory, rows, probabilities, labels):
    directory = Path(directory)
    probabilities = np.asarray(probabilities)
    if probabilities.shape != (len(rows), len(labels)) or not np.isfinite(probabilities).all():
        raise ValueError("Invalid probability outputs")
    predicted = [labels[i] for i in probabilities.argmax(axis=1)]
    metrics = classification_metrics([r["label"] for r in rows], predicted, labels)
    write_json(directory / "metrics.json", metrics)
    records = [{"id": r["id"], "claim": r["claim"], "label": r["label"], "prediction": pred,
                **{f"probability_{label}": float(p) for label, p in zip(labels, probs)}}
               for r, pred, probs in zip(rows, predicted, probabilities)]
    write_csv(directory / "predictions.csv", records,
              ["id", "claim", "label", "prediction"] + [f"probability_{label}" for label in labels])
    write_csv(directory / "confusion_matrix.csv",
              [{"true_label": label, **dict(zip(labels, line))} for label, line in zip(labels, metrics["confusion_matrix"])],
              ["true_label"] + labels)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    fig, ax = plt.subplots(figsize=(7, 6))
    ConfusionMatrixDisplay(np.array(metrics["confusion_matrix"]), display_labels=labels).plot(ax=ax, colorbar=False, cmap="Blues")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(directory / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    save_normalized_matrix(directory, metrics)
    return metrics


def save_normalized_matrix(directory, metrics):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import ConfusionMatrixDisplay
    counts = np.asarray(metrics["confusion_matrix"], dtype=float)
    totals = counts.sum(axis=1, keepdims=True)
    percentages = np.divide(counts * 100, totals, out=np.zeros_like(counts), where=totals != 0)
    fig, ax = plt.subplots(figsize=(7, 6))
    display = ConfusionMatrixDisplay(percentages, display_labels=metrics["labels"])
    display.plot(ax=ax, cmap="Blues", colorbar=False, values_format=".2f", im_kw={"vmin": 0, "vmax": 100})
    for text in display.text_.flat:
        text.set_text(text.get_text() + "%")
    ax.set_title("Row-normalised confusion matrix (%)")
    ax.tick_params(axis="x", rotation=15)
    fig.tight_layout()
    fig.savefig(Path(directory) / "confusion_matrix_normalized.png", dpi=160)
    plt.close(fig)


# FEATURES
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


class ClassicalClassifier:
    def __init__(self, settings, evidence, C, seed=42):
        opts = {k: settings[k] for k in ["max_features", "min_df", "max_df", "lowercase", "sublinear_tf", "norm", "smooth_idf"]}
        opts.update(ngram_range=tuple(settings["ngram_range"]), stop_words=None, use_idf=True)
        self.claim_vectorizer = TfidfVectorizer(**opts)
        self.evidence_vectorizer = TfidfVectorizer(**opts) if evidence else None
        self.classifier = LogisticRegression(C=C, penalty="l2", solver="lbfgs", class_weight="balanced",
                                             max_iter=settings["max_iter"], tol=settings["tol"],
                                             fit_intercept=True, random_state=seed)

    def features(self, rows, fit=False):
        method = "fit_transform" if fit else "transform"
        claim = getattr(self.claim_vectorizer, method)([r["claim"] for r in rows])
        if self.evidence_vectorizer is None:
            return claim
        evidence = getattr(self.evidence_vectorizer, method)([r["evidence"] for r in rows])
        return hstack([claim, evidence], format="csr")

    def fit(self, rows):
        self.classifier.fit(self.features(rows, fit=True), [r["label"] for r in rows])
        return self

    def probabilities(self, rows, labels):
        values = self.classifier.predict_proba(self.features(rows))
        order = [list(self.classifier.classes_).index(label) for label in labels]
        return values[:, order]


ClassicalClassifier.__module__ = "train_classical"

# COMMAND LINE WORKFLOW
# Fit TF-IDF/LR on train, choose C by validation macro F1. Never load test rows.
import argparse
from pathlib import Path
import time
import warnings
import joblib
from sklearn.exceptions import ConvergenceWarning

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_dataset import prepared_manifest, load_split
from prepare_dataset import write_json
from prepare_dataset import EXPERIMENTS, configuration, labels_for


def main():
    names = [name for name, spec in EXPERIMENTS.items() if spec[2] == "classical"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=names + ["all"], default="binary_evidence_classical")
    parser.add_argument("--config", type=Path)
    args = parser.parse_args()
    config = configuration(args.config)
    ready = prepared_manifest(ROOT)
    for name in names if args.experiment == "all" else [args.experiment]:
        labels = labels_for(name)
        train = load_split(name, "train", ROOT)
        validation = load_split(name, "validation", ROOT)
        destination = begin_run(name, config, ready, ROOT)
        c_values = sorted(set(config["classical"]["C_candidates"]))
        if not c_values or any(c <= 0 for c in c_values):
            raise ValueError("C candidates must be positive")
        model = ClassicalClassifier(config["classical"], EXPERIMENTS[name][1], c_values[0], config["seed"])
        started = time.perf_counter()
        x_train = model.features(train, fit=True)
        x_validation = model.features(validation)
        history, best, best_c = [], -1.0, None
        for C in c_values:
            candidate = ClassicalClassifier(config["classical"], EXPERIMENTS[name][1], C, config["seed"])
            model.classifier = candidate.classifier
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always", ConvergenceWarning)
                model.classifier.fit(x_train, [r["label"] for r in train])
            converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
            predicted = model.classifier.predict(x_validation).tolist()
            score = classification_metrics([r["label"] for r in validation], predicted, labels)
            history.append({"C": C, "validation_macro_f1": score["macro_f1"],
                            "converged": converged, "iterations": model.classifier.n_iter_.tolist(),
                            "warnings": [str(w.message) for w in caught]})
            write_json(destination / "training_history.json", history)
            if not converged:
                raise RuntimeError("Logistic regression did not converge. Review training_history.json before proceeding.")
            if score["macro_f1"] > best:
                best, best_c = score["macro_f1"], C
                joblib.dump(model, destination / "model.joblib")
        model = joblib.load(destination / "model.joblib")
        result = save_results(results_directory(name, destination.name, ROOT) / "validation", validation,
                              model.probabilities(validation, labels), labels)
        write_json(destination / "summary.json", {"status": "complete", "run_id": destination.name, "selected_C": best_c,
                   "selection_metric": "validation_macro_f1", "validation": result,
                   "training_seconds": time.perf_counter() - started, "labels": labels,
                   "train_records": len(train), "validation_records": len(validation)})
        lock_run(name, destination.name, ROOT)
        print(f"{name}: selected C={best_c}, validation macro F1={best:.6f}")


if __name__ == "__main__":
    main()
