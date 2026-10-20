"""Shared MNIST loading and train/validation/test split."""

from pathlib import Path

import torch
from torch.utils.data import random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"


def prepare_datasets(seed: int = 42):
    """Load MNIST and make the same 50k/10k split for each model."""
    transform = transforms.ToTensor()
    full_train = datasets.MNIST(
        root=str(DATA_DIR), train=True, download=True, transform=transform
    )
    test_data = datasets.MNIST(
        root=str(DATA_DIR), train=False, download=True, transform=transform
    )
    train_data, val_data = random_split(
        full_train,
        [50_000, 10_000],
        generator=torch.Generator().manual_seed(seed),
    )
    return train_data, val_data, test_data
