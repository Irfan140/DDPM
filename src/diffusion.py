"""Forward and reverse diffusion operations."""

import torch
from torch import nn


class DiffusionSchedule:
    def __init__(self, timesteps: int, device: torch.device):
        self.timesteps = timesteps
        self.betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)

    def add_noise(
        self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if noise is None:
            noise = torch.randn_like(x0)
        alpha_bar = self.alpha_bars[t][:, None, None, None]
        xt = torch.sqrt(alpha_bar) * x0 + torch.sqrt(1.0 - alpha_bar) * noise
        return xt, noise


@torch.inference_mode()
def sample(
    model: nn.Module,
    *,
    count: int,
    timesteps: int,
    image_size: int,
    device: torch.device,
    generator: torch.Generator,
) -> torch.Tensor:
    schedule = DiffusionSchedule(timesteps, device)
    x = torch.randn((count, 1, image_size, image_size), device=device, generator=generator)
    for step in reversed(range(timesteps)):
        t = torch.full((count,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t)
        alpha = schedule.alphas[step]
        alpha_bar = schedule.alpha_bars[step]
        beta = schedule.betas[step]
        x = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if step > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + torch.sqrt(beta) * noise
    return x.clamp(-1.0, 1.0)
