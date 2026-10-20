"""Shared MNIST loading and train/validation/test split."""

from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Subset, random_split
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


def flatten_dataset(dataset) -> tuple[np.ndarray, np.ndarray]:
    """Flatten MNIST images into normalized 784-pixel features and labels."""
    if isinstance(dataset, Subset):
        indices = torch.as_tensor(dataset.indices, dtype=torch.long)
        base_dataset = dataset.dataset
        images = base_dataset.data[indices]
        labels = base_dataset.targets[indices]
    else:
        images = dataset.data
        labels = dataset.targets

    features = images.reshape(len(images), -1).to(torch.float32).div_(255.0)
    return features.numpy(), labels.numpy()
