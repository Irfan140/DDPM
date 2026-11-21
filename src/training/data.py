"""Fashion-MNIST data loading for Kaggle CSVs and torchvision datasets."""

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import datasets, transforms


class FashionMNISTCsv(Dataset):
    def __init__(self, csv_path: Path):
        data = np.loadtxt(csv_path, delimiter=",", skiprows=1, dtype=np.uint8)
        if data.ndim != 2 or data.shape[1] != 785:
            raise ValueError(f"Expected label + 784 pixels in {csv_path}, got shape {data.shape}.")
        self.images = torch.from_numpy(data[:, 1:].copy()).reshape(-1, 1, 28, 28).float()
        self.images = self.images.div(127.5).sub(1.0)

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.images[index]


def _find_training_csv(data_dir: Path) -> Path | None:
    preferred = ("fashion-mnist_train.csv", "fashion_mnist_train.csv", "train.csv")
    candidates = [path for path in data_dir.rglob("*.csv") if path.is_file()]
    candidates.sort(key=lambda path: (path.name.lower() not in preferred, path.name.lower()))
    for path in candidates:
        try:
            with path.open(newline="", encoding="utf-8-sig") as stream:
                header = next(csv.reader(stream))
                first_row = next(csv.reader(stream))
            if len(header) == 785 and len(first_row) == 785:
                return path
        except (OSError, StopIteration, UnicodeError):
            continue
    return None


def build_training_dataset(data_dir: Path, download: bool = False) -> Dataset:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = _find_training_csv(data_dir)
    if csv_path is not None:
        print(f"Loading Kaggle Fashion-MNIST CSV: {csv_path}")
        return FashionMNISTCsv(csv_path)
    print(f"No 785-column Fashion-MNIST CSV found under {data_dir}; using torchvision FashionMNIST.")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]
    )
    return datasets.FashionMNIST(
        root=str(data_dir), train=True, download=download, transform=transform
    )
