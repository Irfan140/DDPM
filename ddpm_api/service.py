"""Checkpoint loading and image generation service."""

import base64
import io
import secrets
import threading

import torch
from PIL import Image

from .config import BASE_CHANNELS, IMAGE_SIZE, TIMESTEPS, WEIGHTS_PATH, resolve_device
from .model import DDPMUNet
from .sampling import sample


class InferenceService:
    def __init__(self):
        self.device = resolve_device()
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_PATH}. Set DDPM_WEIGHTS to the checkpoint path."
            )
        self.model = DDPMUNet(BASE_CHANNELS).to(self.device)
        checkpoint = torch.load(WEIGHTS_PATH, map_location=self.device, weights_only=True)
        if isinstance(checkpoint, dict):
            for key in ("state_dict", "model_state_dict", "model"):
                if key in checkpoint and isinstance(checkpoint[key], dict):
                    checkpoint = checkpoint[key]
                    break
        if not isinstance(checkpoint, dict):
            raise ValueError("Checkpoint must contain a model state dictionary.")
        if checkpoint and all(key.startswith("module.") for key in checkpoint):
            checkpoint = {key.removeprefix("module."): value for key, value in checkpoint.items()}
        self.model.load_state_dict(checkpoint, strict=True)
        self.model.eval()
        self._generation_lock = threading.Lock()

    def generate(self, count: int, seed: int | None) -> tuple[list[str], int]:
        generator = torch.Generator(device=self.device)
        if seed is None:
            seed = secrets.randbits(63)
        generator.manual_seed(seed)
        with self._generation_lock:
            images = sample(
                self.model,
                count=count,
                timesteps=TIMESTEPS,
                image_size=IMAGE_SIZE,
                device=self.device,
                generator=generator,
            ).cpu()
        encoded = []
        for tensor in images:
            pixels = ((tensor.squeeze(0) + 1.0) * 127.5).round().byte().numpy()
            buffer = io.BytesIO()
            Image.fromarray(pixels, mode="L").save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return encoded, seed
