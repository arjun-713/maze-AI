"""Explicit runtime setup and versioned atomic checkpoint I/O. No import-time work."""

import hashlib
import json
import os
import random
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch

from .activation import require_ready
from .config import Config, from_dict

CHECKPOINT_VERSION = 1
FEATURE_VERSION = 1


def select_device(name: str) -> torch.device:
    require_ready()
    if name not in ("cpu", "cuda", "mps", "auto"):
        raise ValueError("device must be cpu, cuda, mps, or auto")
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
        # MPS is opt-in until evaluated; auto keeps predictable CPU behavior on Macs.
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; use --device cpu (or mps on Apple Silicon)")
    if name == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError(
            "MPS unavailable. Use native ARM64 Python/PyTorch on a supported Mac, or --device cpu"
        )
    return torch.device(name)


def configure(config: Config) -> torch.device:
    require_ready()
    config.validate()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    torch.set_num_threads(config.threads)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(config.deterministic)
    torch.backends.cudnn.benchmark = False
    return select_device(config.device)


def source_digest() -> str:
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()


def save_checkpoint(path: Path, model, config: Config, *, optimizer=None, state=None) -> None:
    require_ready()
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": CHECKPOINT_VERSION,
        "feature_version": FEATURE_VERSION,
        "config": asdict(config),
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict() if optimizer is not None else None,
        "state": state,
        "torch_rng": torch.get_rng_state(),
        "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "mps_rng": torch.mps.get_rng_state() if next(model.parameters()).device.type == "mps" else None,
        "python_rng": random.getstate(),
        "source_digest": source_digest(),
        "torch_version": str(torch.__version__),
        "numpy_version": str(np.__version__),
    }
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            torch.save(payload, stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def read_checkpoint(path: str | Path) -> tuple[dict, Config]:
    require_ready()
    # No arbitrary agent objects or NumPy pickles: tensor/primitive-only payload.
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("version") != CHECKPOINT_VERSION or payload.get("feature_version") != FEATURE_VERSION:
        raise ValueError("checkpoint/observation feature version is incompatible")
    config = from_dict(payload["config"])
    return payload, config


def restore_rng(payload: dict, device: torch.device) -> None:
    require_ready()
    torch.set_rng_state(payload["torch_rng"].cpu())
    random.setstate(payload["python_rng"])
    if device.type == "cuda" and payload["cuda_rng"]:
        if len(payload["cuda_rng"]) != torch.cuda.device_count():
            raise ValueError("CUDA device count changed; resume on CPU or use the original device layout")
        torch.cuda.set_rng_state_all(payload["cuda_rng"])
    if device.type == "mps" and payload.get("mps_rng") is not None:
        torch.mps.set_rng_state(payload["mps_rng"].cpu())


def append_json(path: Path, item: dict) -> None:
    require_ready()
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(item, allow_nan=False) + "\n")
