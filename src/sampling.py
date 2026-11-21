"""Ancestral DDPM sampling using the training notebook's noise schedule."""

import torch
from torch import nn


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
    betas = torch.linspace(1e-4, 0.02, timesteps, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)

    x = torch.randn(
        (count, 1, image_size, image_size), device=device, generator=generator
    )
    for step in reversed(range(timesteps)):
        t = torch.full((count,), step, device=device, dtype=torch.long)
        predicted_noise = model(x, t)
        alpha = alphas[step]
        alpha_bar = alpha_bars[step]
        beta = betas[step]
        x = (x - beta / torch.sqrt(1.0 - alpha_bar) * predicted_noise) / torch.sqrt(alpha)
        if step > 0:
            noise = torch.randn(x.shape, device=device, generator=generator)
            x = x + torch.sqrt(beta) * noise
    return x.clamp(-1.0, 1.0)
