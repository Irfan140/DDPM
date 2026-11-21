"""Command-line entry point for local or Kaggle training."""

import argparse
import os
from pathlib import Path

from config import load_environment

from .config import TrainingConfig
from .engine import train


def build_parser() -> argparse.ArgumentParser:
    load_environment()
    parser = argparse.ArgumentParser(description="Train the Fashion-MNIST DDPM.")
    parser.add_argument("--data-dir", type=Path, default=Path(os.getenv("DDPM_DATA_DIR", "./data")))
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path(os.getenv("DDPM_CHECKPOINT", "./artifacts/ddpm_fashion_mnist.pth")),
    )
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--timesteps", type=int, default=1500)
    parser.add_argument("--base-channels", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument(
        "--mixed-precision",
        choices=("no", "fp16", "bf16"),
        default=os.getenv("DDPM_MIXED_PRECISION", "no"),
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--download", action="store_true", help="Download Fashion-MNIST if it is not mounted locally.")
    parser.add_argument("--wandb-project", default=os.getenv("WANDB_PROJECT", "fashion-mnist-ddpm"))
    parser.add_argument("--wandb-entity", default=os.getenv("WANDB_ENTITY"))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--no-wandb", action="store_true", help="Disable W&B tracking for a local run.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = TrainingConfig(
        data_dir=args.data_dir,
        checkpoint=args.checkpoint,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        timesteps=args.timesteps,
        base_channels=args.base_channels,
        num_workers=args.num_workers,
        mixed_precision=args.mixed_precision,
        seed=args.seed,
        download=args.download,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=args.run_name,
    )
    result = train(config, use_wandb=not args.no_wandb)
    print(f"Saved inference checkpoint: {result}")


if __name__ == "__main__":
    main()
