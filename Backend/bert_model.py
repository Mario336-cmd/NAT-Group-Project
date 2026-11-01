# Local inference with the restored claim-and-evidence BERT checkpoint.

from __future__ import annotations

import os
from pathlib import Path
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from Backend.settings import Settings


class ClaimInputError(ValueError):
    # A claim/evidence pair cannot be encoded without discarding input.
    pass


class BertClassifier:
    def __init__(self, settings: Settings, progress=None):
        completed = progress if progress is not None else lambda step: None
        os.environ["USE_TF"] = "0"
        os.environ["USE_FLAX"] = "0"
        os.environ["HF_HUB_OFFLINE"] = "1"

        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        scripts = str(Path(settings.root) / "Python Scripts")
        if scripts not in sys.path:
            sys.path.insert(0, scripts)
        # This is the same lossless pair construction used in training/evaluation.
        from prepare_evidence import encode_claim

        checkpoint = Path(settings.checkpoint_dir)
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"BERT checkpoint directory is missing: {checkpoint}")
        completed("dependencies")
        self.tokenizer = AutoTokenizer.from_pretrained(checkpoint, local_files_only=True)
        if any(token_id is None for token_id in (
            self.tokenizer.cls_token_id, self.tokenizer.sep_token_id, self.tokenizer.pad_token_id
        )):
            raise ValueError("The saved BERT tokenizer requires CLS, SEP and PAD tokens.")
        completed("tokenizer")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            checkpoint, local_files_only=True
        )
        self.labels = {int(index): label for index, label in self.model.config.id2label.items()}
        if self.labels != {0: "SUPPORTS", 1: "REFUTES"}:
            raise ValueError("The checkpoint must map 0 to SUPPORTS and 1 to REFUTES.")
        if self.model.config.label2id != {"SUPPORTS": 0, "REFUTES": 1}:
            raise ValueError("The checkpoint contains an inconsistent label mapping.")
        completed("weights")

        self.max_length = settings.max_length
        if not 4 <= self.max_length <= self.model.config.max_position_embeddings:
            raise ValueError("BERT max_length must fit the checkpoint's position limit.")
        self.chunk_batch_size = settings.chunk_batch_size
        if self.chunk_batch_size < 1:
            raise ValueError("chunk_batch_size must be positive.")
        device = settings.device
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model.to(self.device)
        self.model.eval()
        self._torch = torch
        self._encode_claim = encode_claim
        completed("device")

    def predict(self, claim: str, evidence: str) -> dict:
        if not isinstance(claim, str) or not claim.strip():
            raise ClaimInputError("Enter a nonempty claim.")
        if not isinstance(evidence, str) or not evidence.strip():
            raise ClaimInputError("BERT needs nonempty evidence alongside the claim.")
        try:
            chunks = self._encode_claim(
                self.tokenizer, claim, evidence, max_length=self.max_length
            )
        except ValueError as exc:
            raise ClaimInputError(str(exc)) from exc

        torch = self._torch
        outputs = []
        with torch.inference_mode():
            for start in range(0, len(chunks), self.chunk_batch_size):
                batch = self.tokenizer.pad(
                    chunks[start : start + self.chunk_batch_size],
                    padding=True,
                    return_tensors="pt",
                )
                inputs = {key: value.to(self.device) for key, value in batch.items()}
                outputs.append(self.model(**inputs).logits)
            # Aggregate logits, not probabilities; retain every evidence chunk.
            probabilities = torch.cat(outputs).mean(dim=0).softmax(dim=-1)
            if not torch.isfinite(probabilities).all():
                raise RuntimeError("BERT returned nonfinite probabilities.")
            probabilities = probabilities.cpu().tolist()
        predicted = max(range(len(probabilities)), key=probabilities.__getitem__)
        return {
            "label": self.labels[predicted],
            "confidence": float(probabilities[predicted]),
            "probabilities": {
                self.labels[index]: float(value) for index, value in enumerate(probabilities)
            },
            "chunks": len(chunks),
        }
