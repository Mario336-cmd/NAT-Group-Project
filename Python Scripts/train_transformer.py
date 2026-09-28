import os
os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("USE_FLAX", "0")
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))


# BERT MODEL
import random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset

from prepare_evidence import encode_claim


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def class_weights(rows, labels):
    counts = [sum(r["label"] == label for r in rows) for label in labels]
    if not all(counts):
        raise ValueError("Every class must appear in training")
    return torch.tensor([len(rows) / (len(labels) * n) for n in counts], dtype=torch.float32)


class ClaimDataset(Dataset):
    def __init__(self, rows, tokenizer, labels, evidence, max_length=512):
        self.records = []
        for index, row in enumerate(rows):
            if index and index % 10000 == 0:
                print(f"Encoding claim groups: {index}/{len(rows)}", flush=True)
            self.records.append({"chunks": encode_claim(tokenizer, row["claim"],
                                 row["evidence"] if evidence else None, max_length),
                                 "label": labels.index(row["label"]), "id": row["id"]})

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


class ClaimCollator:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, records):
        chunks, owners = [], []
        for owner, record in enumerate(records):
            chunks.extend(record["chunks"])
            owners.extend([owner] * len(record["chunks"]))
        return {"inputs": dict(self.tokenizer.pad(chunks, padding=True, return_tensors="pt")),
                "owners": torch.tensor(owners, dtype=torch.long),
                "labels": torch.tensor([r["label"] for r in records], dtype=torch.long),
                "ids": [r["id"] for r in records]}


def mean_claim_logits(chunk_logits, owners, claim_count):
    counts = torch.bincount(owners, minlength=claim_count)
    if counts.shape[0] != claim_count or torch.any(counts == 0):
        raise ValueError("Each claim must have at least one chunk")
    sums = chunk_logits.new_zeros((claim_count, chunk_logits.shape[-1]))
    sums = sums.index_add(0, owners, chunk_logits)
    return sums / counts.to(chunk_logits.dtype).unsqueeze(1)


def forward_claims(model, batch, device, chunk_batch_size):
    if chunk_batch_size < 1:
        raise ValueError("chunk_forward_batch_size must be positive")
    total = len(batch["owners"])
    outputs = []
    for start in range(0, total, chunk_batch_size):
        inputs = {k: v[start:start + chunk_batch_size].to(device) for k, v in batch["inputs"].items()}
        outputs.append(model(**inputs).logits)
    return mean_claim_logits(torch.cat(outputs), batch["owners"].to(device), len(batch["labels"]))


def claim_loss_sum(logits, labels, weights):
    # The caller divides by the total class-weight mass for the entire accumulation window.
    return nn.functional.cross_entropy(logits, labels, weight=weights, reduction="sum")


@torch.no_grad()
def probabilities(model, loader, device, chunk_batch_size):
    model.eval()
    result, ids = [], []
    for batch in loader:
        logits = forward_claims(model, batch, device, chunk_batch_size)
        result.append(logits.softmax(-1).cpu().numpy())
        ids.extend(batch["ids"])
    return np.concatenate(result), ids


# CHECKPOINTS
# Resume state at optimizer-update boundaries. independent of pretrained downloads.
import os
from pathlib import Path
import random
import tempfile

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def capture_rng():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []}


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"]:
        if not torch.cuda.is_available() or len(state["cuda"]) != torch.cuda.device_count():
            raise ValueError("Resume CUDA device configuration differs from the checkpoint")
        torch.cuda.set_rng_state_all([value.cpu() for value in state["cuda"]])


def epoch_loader(dataset, collator, batch_size, seed, epoch, next_batch=0):
    order = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(seed + epoch)).tolist()
    start = next_batch * batch_size
    if next_batch < 0 or next_batch > (len(dataset) + batch_size - 1) // batch_size:
        raise ValueError("Invalid saved batch position")
    # The sampler and DataLoader have private RNGs. making a resumed iterator must
    # not consume the model's dropout RNG or replay already completed batches.
    return DataLoader(Subset(dataset, order[start:]), batch_size=batch_size, shuffle=False,
                      collate_fn=collator, num_workers=0,
                      generator=torch.Generator().manual_seed(seed + 1000000 + epoch))


def save_resume(path, model, optimizer, progress, identity):
    path = Path(path)
    state = {"version": 1, "identity": identity, "model": model.state_dict(),
             "optimizer": optimizer.state_dict(), "progress": progress, "rng": capture_rng()}
    fd, temporary = tempfile.mkstemp(dir=path.parent, suffix=".pt.tmp")
    os.close(fd)
    try:
        torch.save(state, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_resume(path, model, optimizer, identity):
    # Only load the project's own local checkpoint. It includes Python/NumPy RNG
    # state as well as tensors, so weights_only is deliberately false.
    state = torch.load(path, map_location="cpu", weights_only=False)
    if state.get("version") != 1 or state["identity"] != identity:
        raise ValueError("Resume checkpoint identity/configuration mismatch")
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    restore_rng(state["rng"])
    return state["progress"]


# COMMAND LINE WORKFLOW
# Fine-tune BERT with one loss per claim and resumable optimizer-boundary checkpoints.
import argparse
from itertools import islice
from pathlib import Path
import time

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from prepare_dataset import prepared_manifest, load_split, prepared_tokenizer, preparation_identity
from prepare_dataset import write_json, read_json, object_hash
from train_classical import classification_metrics, save_results
from train_classical import lock_run, begin_run, resume_run, run_directory, results_directory
from prepare_dataset import EXPERIMENTS, configuration, labels_for


def main():
    names = [name for name, spec in EXPERIMENTS.items() if spec[2] == "bert"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", choices=names + ["all"], default="binary_evidence_bert")
    parser.add_argument("--config", type=Path)
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--resume", metavar="RUN_ID", help="Continue one interrupted run from its saved progress")
    args = parser.parse_args()
    if args.resume and args.experiment == "all":
        parser.error("Resume one named experiment/run at a time")
    import torch
    from torch.utils.data import DataLoader
    from transformers import AutoModelForSequenceClassification
    if args.resume and args.config is None:
        config = read_json(run_directory(args.experiment, ROOT, args.resume) / "configuration.json")["configuration"]
    else:
        config = configuration(args.config)
    cfg = config["bert"]
    ready = prepared_manifest(ROOT)
    tokenizer, tokenizer_info = prepared_tokenizer(ROOT)
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if args.device == "auto" else args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA is unavailable in this Python environment")
    print(f"Device: {device}. effective target batch: {cfg['batch_size_claims'] * cfg['gradient_accumulation']} claims")
    for name in names if args.experiment == "all" else [args.experiment]:
        set_seed(config["seed"])
        labels = labels_for(name)
        rows = load_split(name, "train", ROOT)
        val_rows = load_split(name, "validation", ROOT)
        destination = (resume_run(name, args.resume, config, ready, ROOT) if args.resume
                       else begin_run(name, config, ready, ROOT))
        train_data = ClaimDataset(rows, tokenizer, labels, EXPERIMENTS[name][1], cfg["max_length"])
        val_data = ClaimDataset(val_rows, tokenizer, labels, EXPERIMENTS[name][1], cfg["max_length"])
        collate = ClaimCollator(tokenizer)
        val_loader = DataLoader(val_data, batch_size=cfg["batch_size_claims"], shuffle=False,
                                collate_fn=collate, num_workers=0,
                                generator=torch.Generator().manual_seed(config["seed"] + 2000000))
        model = AutoModelForSequenceClassification.from_pretrained(
            config["tokenizer"], revision=tokenizer_info["model_revision"], num_labels=len(labels),
            id2label=dict(enumerate(labels)), label2id={label: i for i, label in enumerate(labels)},
            attn_implementation="eager")
        if cfg["gradient_checkpointing"]:
            model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.to(device)
        weights = class_weights(rows, labels).to(device)
        decay, no_decay = [], []
        for param_name, parameter in model.named_parameters():
            (no_decay if param_name.endswith("bias") or "LayerNorm.weight" in param_name else decay).append(parameter)
        optimizer = torch.optim.AdamW([{"params": decay, "weight_decay": cfg["weight_decay"]},
                                       {"params": no_decay, "weight_decay": 0.0}],
                                      lr=cfg["learning_rate"], betas=tuple(cfg["adam_betas"]), eps=cfg["adam_epsilon"])
        identity = {"experiment": name, "run_id": destination.name,
                    "prepared_identity": preparation_identity(ready), "config": object_hash(config),
                    "device": str(device),
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None}
        progress = {"epoch": 1, "next_batch": 0, "global_steps": 0, "epoch_steps": 0,
                    "total_loss": 0.0, "total_weight": 0.0, "history": [], "best": -1.0,
                    "stale": 0, "best_epoch": None, "best_metrics": None, "training_seconds": 0.0}
        if args.resume:
            progress = load_resume(destination / "resume.pt", model, optimizer, identity)
            print(f"Resuming {name}/{destination.name}: epoch {progress['epoch']}, next batch {progress['next_batch']}")
        started = time.perf_counter()
        previous_seconds = progress["training_seconds"]

        def save_progress():
            progress["training_seconds"] = previous_seconds + time.perf_counter() - started
            save_resume(destination / "resume.pt", model, optimizer, progress, identity)

        if not args.resume:
            save_progress()
        while progress["epoch"] <= cfg["max_epochs"] and progress["stale"] < cfg["patience"]:
            epoch = progress["epoch"]
            train_loader = epoch_loader(train_data, collate, cfg["batch_size_claims"], config["seed"],
                                        epoch, progress["next_batch"])
            model.train()
            iterator = iter(train_loader)
            while window := list(islice(iterator, cfg["gradient_accumulation"])):
                optimizer.zero_grad(set_to_none=True)
                denominator = sum(weights[b["labels"].to(device)].sum().item() for b in window)
                for batch in window:
                    logits = forward_claims(model, batch, device, cfg["chunk_forward_batch_size"])
                    loss_sum = claim_loss_sum(logits, batch["labels"].to(device), weights)
                    if not torch.isfinite(loss_sum):
                        raise FloatingPointError("Non-finite training loss. checkpoint not marked complete")
                    (loss_sum / denominator).backward()
                    progress["total_loss"] += loss_sum.detach().item()
                progress["total_weight"] += denominator
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["gradient_clipping"], error_if_nonfinite=True)
                optimizer.step()
                progress["next_batch"] += len(window)
                progress["epoch_steps"] += 1
                progress["global_steps"] += 1
                if progress["global_steps"] % cfg["resume_checkpoint_every_steps"] == 0:
                    save_progress()
                if progress["epoch_steps"] % 100 == 0:
                    print(f"{name} epoch {epoch}, optimizer step {progress['epoch_steps']}, "
                          f"loss {progress['total_loss'] / progress['total_weight']:.5f}", flush=True)
            # Save the end of training before validation, so interruption during validation
            # resumes without replaying the epoch's training updates.
            save_progress()
            val_probs, ids = probabilities(model, val_loader, device, cfg["chunk_forward_batch_size"])
            if ids != [r["id"] for r in val_rows]:
                raise AssertionError("Validation record order changed")
            predicted = [labels[i] for i in val_probs.argmax(axis=1)]
            metrics = classification_metrics([r["label"] for r in val_rows], predicted, labels)
            progress["history"].append({"epoch": epoch,
                "train_weighted_loss": progress["total_loss"] / progress["total_weight"],
                "validation_macro_f1": metrics["macro_f1"], "optimizer_steps": progress["epoch_steps"]})
            write_json(destination / "training_history.json", progress["history"])
            if metrics["macro_f1"] > progress["best"]:
                progress["best"] = metrics["macro_f1"]
                progress["stale"] = 0
                model.save_pretrained(destination / "checkpoint", safe_serialization=True)
                tokenizer.save_pretrained(destination / "checkpoint")
                progress["best_metrics"] = save_results(
                    results_directory(name, destination.name, ROOT) / "validation", val_rows, val_probs, labels)
                progress["best_epoch"] = epoch
            else:
                progress["stale"] += 1
            print(f"{name} epoch {epoch}: validation macro F1={metrics['macro_f1']:.6f}", flush=True)
            progress.update(epoch=epoch + 1, next_batch=0, epoch_steps=0, total_loss=0.0, total_weight=0.0)
            save_progress()
        write_json(destination / "summary.json", {"status": "complete", "run_id": destination.name,
                   "best_epoch": progress["best_epoch"], "selection_metric": "validation_macro_f1",
                   "validation": progress["best_metrics"], "labels": labels,
                   "training_seconds": progress["training_seconds"], "optimizer_steps": progress["global_steps"],
                   "class_weights": weights.cpu().tolist(), "device": str(device), "gpu": identity["gpu"],
                   "train_records": len(rows), "validation_records": len(val_rows)})
        lock_run(name, destination.name, ROOT)
        del model, optimizer, train_data, val_data
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
