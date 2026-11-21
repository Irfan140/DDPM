"""Training loop with optional Weights & Biases tracking."""

from pathlib import Path

import torch
import wandb
from accelerate import Accelerator
from accelerate.utils import set_seed
from torch.utils.data import DataLoader

from diffusion import DiffusionSchedule
from model import DDPMUNet
from .checkpoint import save_checkpoint
from .config import TrainingConfig
from .data import build_training_dataset


def train(config: TrainingConfig, *, use_wandb: bool = True) -> Path:
    if config.base_channels % 32 != 0:
        raise ValueError("base_channels must be divisible by 32 for GroupNorm.")
    if config.epochs < 1 or config.batch_size < 1 or config.timesteps < 1:
        raise ValueError("epochs, batch_size, and timesteps must be positive.")
    accelerator = Accelerator(mixed_precision=config.mixed_precision)
    set_seed(config.seed, device_specific=True)
    device = accelerator.device
    accelerator.print(f"Training on {device} with {accelerator.num_processes} process(es)")

    with accelerator.main_process_first():
        dataset = build_training_dataset(config.data_dir, download=config.download)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = DDPMUNet(config.base_channels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate)
    model, optimizer, loader = accelerator.prepare(model, optimizer, loader)
    schedule = DiffusionSchedule(config.timesteps, device)
    run = None
    if use_wandb and accelerator.is_main_process:
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
                with accelerator.autocast():
                    predicted_noise = model(noisy_images, t)
                    loss = torch.nn.functional.mse_loss(predicted_noise, noise)

                optimizer.zero_grad(set_to_none=True)
                accelerator.backward(loss)
                optimizer.step()
                gathered_loss = accelerator.gather(loss.detach().reshape(1))
                total_loss += gathered_loss.sum().item()
                seen += gathered_loss.numel()

            epoch_loss = total_loss / max(seen, 1)
            metrics = {"epoch": epoch + 1, "train/loss": epoch_loss}
            accelerator.print(f"Epoch {epoch + 1}/{config.epochs} | loss={epoch_loss:.6f}")
            if run is not None:
                run.log(metrics, step=epoch + 1)
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                unwrapped_model = accelerator.unwrap_model(model)
                save_checkpoint(
                    config.checkpoint,
                    unwrapped_model,
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
                        unwrapped_model,
                        optimizer,
                        epoch=epoch + 1,
                        loss=epoch_loss,
                        config=config.to_dict(),
                    )
            accelerator.wait_for_everyone()
    finally:
        try:
            if run is not None:
                if config.checkpoint.is_file():
                    artifact = wandb.Artifact("fashion-mnist-ddpm", type="model")
                    artifact.add_file(str(config.checkpoint))
                    run.log_artifact(artifact)
                run.finish()
        finally:
            accelerator.wait_for_everyone()
            accelerator.end_training()
    return config.checkpoint
