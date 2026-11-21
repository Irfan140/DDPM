"""Training loop with optional Weights & Biases tracking."""

from pathlib import Path

import torch
import wandb
from torch.utils.data import DataLoader

from ddpm.diffusion import DiffusionSchedule
from ddpm.model import DDPMUNet
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .data import build_training_dataset


def train(config: TrainingConfig, *, use_wandb: bool = True) -> Path:
    if config.base_channels % 32 != 0:
        raise ValueError("base_channels must be divisible by 32 for GroupNorm.")
    if config.epochs < 1 or config.batch_size < 1 or config.timesteps < 1:
        raise ValueError("epochs, batch_size, and timesteps must be positive.")
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on {device}")

    dataset = build_training_dataset(config.data_dir, download=config.download)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = DDPMUNet(config.base_channels).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    schedule = DiffusionSchedule(config.timesteps, device)
    run = None
    if use_wandb:
        run = wandb.init(
            project=config.wandb_project,
            entity=config.wandb_entity,
            name=config.run_name,
            config=config.to_dict(),
        )

    config.checkpoint.parent.mkdir(parents=True, exist_ok=True)
    best_loss = float("inf")
    try:
        for epoch in range(config.epochs):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in loader:
                images = batch[0] if isinstance(batch, (tuple, list)) else batch
                images = images.to(device, non_blocking=True)
                t = torch.randint(0, config.timesteps, (images.shape[0],), device=device)
                noisy_images, noise = schedule.add_noise(images, t)
                predicted_noise = model(noisy_images, t)
                loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                batch_size = images.shape[0]
                total_loss += loss.item() * batch_size
                seen += batch_size

            epoch_loss = total_loss / max(seen, 1)
            metrics = {"epoch": epoch + 1, "train/loss": epoch_loss}
            print(f"Epoch {epoch + 1}/{config.epochs} | loss={epoch_loss:.6f}")
            if run is not None:
                run.log(metrics, step=epoch + 1)
            save_checkpoint(
                config.checkpoint,
                model,
                optimizer,
                epoch=epoch + 1,
                loss=epoch_loss,
                config=config.to_dict(),
            )
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_path = config.checkpoint.with_name("best_" + config.checkpoint.name)
                save_checkpoint(
                    best_path,
                    model,
                    optimizer,
                    epoch=epoch + 1,
                    loss=epoch_loss,
                    config=config.to_dict(),
                )
    finally:
        if run is not None:
            if config.checkpoint.is_file():
                artifact = wandb.Artifact("fashion-mnist-ddpm", type="model")
                artifact.add_file(str(config.checkpoint))
                run.log_artifact(artifact)
            run.finish()
    return config.checkpoint
