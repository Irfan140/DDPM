"""Training configuration and defaults."""

from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass
class TrainingConfig:
    data_dir: Path = Path("./data")
    checkpoint: Path = Path("./artifacts/ddpm_fashion_mnist.pth")
    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    timesteps: int = 1500
    image_size: int = 28
    base_channels: int = 64
    num_workers: int = 2
    seed: int = 42
    download: bool = False
    mixed_precision: str = "no"
    wandb_project: str = "fashion-mnist-ddpm"
    wandb_entity: str | None = None
    run_name: str | None = None

    def to_dict(self) -> dict:
        return {key: str(value) if isinstance(value, Path) else value for key, value in asdict(self).items()}
