"""Shared model and inference configuration."""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_PATH = Path(os.getenv("DDPM_WEIGHTS", PROJECT_ROOT / "ddpm_fashion_mnist.pth"))
DEVICE_SETTING = os.getenv("DDPM_DEVICE", "auto").lower()
TIMESTEPS = 1500
IMAGE_SIZE = 28
BASE_CHANNELS = 64
MAX_BATCH_SIZE = 8


def resolve_device():
    import torch

    if DEVICE_SETTING == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(DEVICE_SETTING)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("DDPM_DEVICE requests CUDA, but CUDA is not available.")
    return device
