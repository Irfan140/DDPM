"""Checkpoint serialization shared by training and inference."""

from pathlib import Path

import torch
from torch import nn


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    epoch: int,
    loss: float,
    config: dict,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "loss": loss,
            "config": config,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def load_model_state(path: Path, device: torch.device) -> tuple[dict, dict]:
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                metadata = checkpoint.get("config", {})
                state_dict = checkpoint[key]
                break
        else:
            metadata, state_dict = {}, checkpoint
    else:
        raise ValueError("Checkpoint must contain a model state dictionary.")
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict, metadata
