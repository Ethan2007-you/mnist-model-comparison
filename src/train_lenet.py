"""Train a classic LeNet-5-style network on MNIST."""

from __future__ import annotations

import argparse
import csv
import random
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch import nn
from torch.utils.data import DataLoader

from mnist_data import PROJECT_ROOT, prepare_datasets


RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42


class LeNet5(nn.Module):
    """LeNet-5-style classifier adapted to MNIST and modern cross-entropy."""

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5)
        self.pool1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.AvgPool2d(kernel_size=2, stride=2)
        # For 32x32 input: 32 -> 28 -> 14 -> 10 -> 5 -> 1.
        self.conv3 = nn.Conv2d(16, 120, kernel_size=5)
        self.fc1 = nn.Linear(120, 84)
        self.fc2 = nn.Linear(84, 10)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        # The original LeNet-5 receives 32x32 images; MNIST is 28x28.
        images = F.pad(images, (2, 2, 2, 2))
        x = torch.tanh(self.conv1(images))
        x = self.pool1(x)
        x = torch.tanh(self.conv2(x))
        x = self.pool2(x)
        x = torch.tanh(self.conv3(x))
        x = torch.flatten(x, start_dim=1)
        x = torch.tanh(self.fc1(x))
        return self.fc2(x)


@torch.no_grad()
def evaluate(model: nn.Module, data_loader: DataLoader, device: torch.device):
    """Return accuracy and predictions while running in evaluation mode."""
    model.eval()
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []

    for images, labels in data_loader:
        images, labels = images.to(device), labels.to(device)
        predictions = model(images).argmax(dim=1)
        correct += (predictions == labels).sum().item()
        total += labels.size(0)
        all_predictions.extend(predictions.cpu().tolist())
        all_labels.extend(labels.cpu().tolist())

    return 100.0 * correct / total, all_labels, all_predictions


def save_history(history: list[dict], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["epoch", "train_loss", "train_accuracy", "validation_accuracy"]
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(history)


def save_metrics(row: dict, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "epochs",
                "batch_size",
                "learning_rate",
                "train_samples",
                "best_validation_accuracy",
                "test_accuracy",
            ],
        )
        writer.writeheader()
        writer.writerow(row)


def save_confusion_matrix(labels: list[int], predictions: list[int], output_path: Path):
    matrix = confusion_matrix(labels, predictions, labels=np.arange(10))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true\\predicted", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row.tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a LeNet-5-style network to classify MNIST digits."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        raise ValueError("epochs and batch-size must both be positive.")
    if args.learning_rate <= 0:
        raise ValueError("learning-rate must be positive.")

    random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_data, val_data, test_data = prepare_datasets(SEED)
    shuffle_generator = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        generator=shuffle_generator,
    )
    val_loader = DataLoader(val_data, batch_size=args.batch_size)
    test_loader = DataLoader(test_data, batch_size=args.batch_size)

    model = LeNet5().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    best_val_accuracy = -1.0
    best_weights = None
    history = []

    for epoch in range(args.epochs):
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
        train_accuracy = 100.0 * correct / total
        val_accuracy, _, _ = evaluate(model, val_loader, device)
        history.append(
            {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "validation_accuracy": val_accuracy,
            }
        )
        print(
            f"Epoch {epoch + 1}/{args.epochs} | Loss={train_loss:.4f} | "
            f"Train={train_accuracy:.2f}% | Validation={val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            best_weights = deepcopy(model.state_dict())

    model.load_state_dict(best_weights)
    test_accuracy, test_labels, test_predictions = evaluate(model, test_loader, device)

    history_path = RESULTS_DIR / "lenet_training_history.csv"
    metrics_path = RESULTS_DIR / "lenet_metrics.csv"
    matrix_path = RESULTS_DIR / "lenet_confusion_matrix.csv"
    save_history(history, history_path)
    save_metrics(
        {
            "model": "LeNet5",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "train_samples": len(train_data),
            "best_validation_accuracy": round(best_val_accuracy, 2),
            "test_accuracy": round(test_accuracy, 2),
        },
        metrics_path,
    )
    save_confusion_matrix(test_labels, test_predictions, matrix_path)

    print("\n===== LeNet final result =====")
    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")
    print(f"Test accuracy: {test_accuracy:.2f}%")
    print(f"Training history saved to: {history_path}")
    print(f"Metrics saved to: {metrics_path}")
    print(f"Test confusion matrix saved to: {matrix_path}")


if __name__ == "__main__":
    main()
