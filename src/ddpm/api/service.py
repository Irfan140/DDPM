"""Checkpoint loading and image generation service."""

import base64
import io
import secrets
import threading

import torch
from PIL import Image

from ddpm.config import BASE_CHANNELS, IMAGE_SIZE, TIMESTEPS, WEIGHTS_PATH, resolve_device
from ddpm.diffusion import sample
from ddpm.model import DDPMUNet
from ddpm.training.checkpoint import load_model_state


class InferenceService:
    def __init__(self):
        self.device = resolve_device()
        if not WEIGHTS_PATH.is_file():
            raise FileNotFoundError(
                f"Model weights not found at {WEIGHTS_PATH}. Set DDPM_WEIGHTS to the checkpoint path."
            )
        state_dict, metadata = load_model_state(WEIGHTS_PATH, self.device)
        base_channels = int(metadata.get("base_channels", BASE_CHANNELS))
        self.model = DDPMUNet(base_channels).to(self.device)
        self.model.load_state_dict(state_dict, strict=True)
        self.timesteps = int(metadata.get("timesteps", TIMESTEPS))
        self.image_size = int(metadata.get("image_size", IMAGE_SIZE))
        self.model.eval()
        self._generation_lock = threading.Lock()

    def generate(self, count: int, seed: int | None) -> tuple[list[str], int]:
        if seed is None:
            seed = secrets.randbits(63)
        generator = torch.Generator(device=self.device).manual_seed(seed)
        with self._generation_lock:
            images = sample(
                self.model,
                count=count,
                timesteps=self.timesteps,
                image_size=self.image_size,
                device=self.device,
                generator=generator,
            ).cpu()
        encoded = []
        for tensor in images:
            pixels = ((tensor.squeeze(0) + 1.0) * 127.5).round().byte().numpy()
            buffer = io.BytesIO()
            Image.fromarray(pixels).save(buffer, format="PNG")
            encoded.append(base64.b64encode(buffer.getvalue()).decode("ascii"))
        return encoded, seed
