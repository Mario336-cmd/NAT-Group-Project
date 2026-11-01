# Local assets and application limits. API credentials are supplied per session.

from dataclasses import dataclass, field
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    root: Path = ROOT
    index_dir: Path = ROOT / "Dataset" / "Index"
    checkpoint_dir: Path = (
        ROOT / "Models" / "binary_evidence_bert" / "runs"
        / "20260928T123142_b12ec98f" / "checkpoint"
    )
    device: str = field(default_factory=lambda: os.getenv("TRUECHECK_DEVICE", "auto"))
    top_k: int = 1
    max_length: int = 512
    chunk_batch_size: int = 8
    max_claim_characters: int = 4000
    explanation_timeout: float = 60.0
    extraction_timeout: float = 60.0
    transcription_timeout: float = 60.0
    max_audio_bytes: int = 5 * 1024 * 1024
    max_upload_bytes: int = 10 * 1024 * 1024
    max_pdf_pages: int = 10
    max_extracted_characters: int = 20000
    session_idle_seconds: int = 1800
    session_max_seconds: int = 8 * 60 * 60
    max_sessions: int = 256
    allowed_hosts: tuple[str, ...] = field(default_factory=lambda: tuple(
        host.strip() for host in os.getenv("TRUECHECK_ALLOWED_HOSTS", "127.0.0.1,localhost").split(",")
        if host.strip()
    ))

    def __post_init__(self):
        if not self.allowed_hosts or any("*" in host or "/" in host or ":" in host for host in self.allowed_hosts):
            raise ValueError("TRUECHECK_ALLOWED_HOSTS must contain explicit hostnames without schemes, ports or wildcards.")

    def asset_issues(self) -> list[str]:
        required = [
            self.index_dir / "passages.jsonl",
            self.index_dir / "embeddings.faiss",
            *(self.index_dir / "bm25" / name for name in (
                "data.csc.index.npy", "indices.csc.index.npy", "indptr.csc.index.npy",
                "params.index.json", "vocab.index.json",
            )),
            *(self.checkpoint_dir / name for name in (
                "config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
            )),
        ]
        return [path.name for path in required if not path.is_file()]
