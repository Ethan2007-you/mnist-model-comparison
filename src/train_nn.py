"""Train a small MNIST MLP and compare dropout rates."""

from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42


def create_model(dropout_rate: float) -> nn.Sequential:
    """Create a fully connected network for 28x28 grayscale images."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 128),
        nn.ReLU(),
        nn.Dropout(p=dropout_rate),
        nn.Linear(128, 10),
    )


def prepare_datasets(seed: int):
    """Download MNIST and split its official training set into train/validation."""
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


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device) -> float:
    """Return classification accuracy as a percentage."""
    model.eval()
    correct = 0
    total = 0

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    return 100.0 * correct / total


def train_model(
    train_data,
    val_loader: DataLoader,
    dropout_rate: float,
    epochs: int,
    batch_size: int,
    seed: int,
    device: torch.device,
):
    """Train one model and restore the weights with the best validation accuracy."""
    # Reset model initialization and batch shuffling for each experiment.
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    shuffle_generator = torch.Generator().manual_seed(seed)

    model = create_model(dropout_rate).to(device)
    train_loader = DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    best_val_acc = -1.0
    best_weights = None
    history = []

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = total_loss / total
        train_acc = 100.0 * correct / total
        val_acc = evaluate(model, val_loader, device)
        history.append(
            {
                "dropout_rate": dropout_rate,
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_acc,
                "validation_accuracy": val_acc,
            }
        )

        print(
            f"p={dropout_rate:.1f} | Epoch {epoch + 1}/{epochs} | "
            f"Loss={train_loss:.4f} | Train={train_acc:.2f}% | "
            f"Validation={val_acc:.2f}%"
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    return model, history, best_val_acc


def save_history(history: list[dict], output_path: Path) -> None:
    """Write per-epoch metrics to a CSV file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dropout_rate",
        "epoch",
        "train_loss",
        "train_accuracy",
        "validation_accuracy",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare MNIST MLP models with different dropout rates."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--dropout-rates",
        type=float,
        nargs="+",
        default=[0.0, 0.2, 0.5],
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must both be positive.")
    if any(rate < 0.0 or rate >= 1.0 for rate in args.dropout_rates):
        raise ValueError("Each dropout rate must be in the range [0, 1).")

    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, val_data, test_data = prepare_datasets(SEED)
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    all_history = []
    best_model = None
    best_rate = None
    best_val_acc = -1.0

    for dropout_rate in args.dropout_rates:
        print(f"\n===== Dropout p={dropout_rate} =====")
        model, history, val_acc = train_model(
            train_data=train_data,
            val_loader=val_loader,
            dropout_rate=dropout_rate,
            epochs=args.epochs,
            batch_size=args.batch_size,
            seed=SEED,
            device=device,
        )
        all_history.extend(history)

        if val_acc > best_val_acc:
            best_model = model
            best_rate = dropout_rate
            best_val_acc = val_acc

    test_acc = evaluate(best_model, test_loader, device)
    output_path = RESULTS_DIR / "dropout_comparison.csv"
    save_history(all_history, output_path)

    print("\n===== Final result =====")
    print(f"Best dropout rate: {best_rate}")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Training history saved to: {output_path}")


if __name__ == "__main__":
    main()
