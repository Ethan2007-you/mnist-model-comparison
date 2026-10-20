"""Train and evaluate a linear SVM on MNIST."""

from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import LinearSVC

from mnist_data import PROJECT_ROOT, flatten_dataset, prepare_datasets


RESULTS_DIR = PROJECT_ROOT / "results"
SEED = 42


def save_metrics(rows: list[dict], output_path: Path, append: bool = False) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    should_append = append and output_path.exists() and output_path.stat().st_size > 0
    mode = "a" if should_append else "w"
    with output_path.open(mode, newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "model",
                "c_value",
                "train_samples",
                "fit_seconds",
                "validation_accuracy",
                "test_accuracy",
            ],
        )
        if not should_append:
            writer.writeheader()
        writer.writerows(rows)


def save_confusion_matrix(matrix: np.ndarray, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(["true\\predicted", *range(10)])
        for digit, row in enumerate(matrix):
            writer.writerow([digit, *row.tolist()])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a linear SVM to classify MNIST handwritten digits."
    )
    parser.add_argument(
        "--c-value",
        type=float,
        default=1.0,
        help="Regularization strength parameter C (default: 1.0).",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50_000,
        help="Use at most this many training images; default uses all 50,000.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=5_000,
        help="Maximum optimizer iterations for LinearSVC.",
    )
    parser.add_argument(
        "--skip-test",
        action="store_true",
        help="Only report validation accuracy while tuning C; do not evaluate the test set.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.c_value <= 0:
        raise ValueError("c-value must be greater than zero.")
    if args.max_train_samples < 1 or args.max_train_samples > 50_000:
        raise ValueError("max-train-samples must be between 1 and 50000.")
    if args.max_iter < 1:
        raise ValueError("max-iter must be positive.")

    train_data, val_data, test_data = prepare_datasets(SEED)
    train_x, train_y = flatten_dataset(train_data)
    val_x, val_y = flatten_dataset(val_data)
    test_x, test_y = flatten_dataset(test_data)

    if args.max_train_samples < len(train_x):
        train_x = train_x[: args.max_train_samples]
        train_y = train_y[: args.max_train_samples]

    model = LinearSVC(
        C=args.c_value,
        dual="auto",
        max_iter=args.max_iter,
        random_state=SEED,
    )

    print(f"Training linear SVM on {len(train_x):,} MNIST images...")
    start_time = time.perf_counter()
    model.fit(train_x, train_y)
    fit_seconds = time.perf_counter() - start_time

    val_predictions = model.predict(val_x)
    val_accuracy = 100.0 * accuracy_score(val_y, val_predictions)

    # Keep the test set untouched while tuning C on the validation split.
    test_accuracy = None
    matrix = None
    if not args.skip_test:
        test_predictions = model.predict(test_x)
        test_accuracy = 100.0 * accuracy_score(test_y, test_predictions)
        matrix = confusion_matrix(test_y, test_predictions, labels=np.arange(10))

    metrics_path = RESULTS_DIR / (
        "svm_tuning.csv" if args.skip_test else "svm_metrics.csv"
    )
    save_metrics(
        [
            {
                "model": "LinearSVC",
                "c_value": args.c_value,
                "train_samples": len(train_x),
                "fit_seconds": round(fit_seconds, 2),
                "validation_accuracy": round(val_accuracy, 2),
                "test_accuracy": "" if test_accuracy is None else round(test_accuracy, 2),
            }
        ],
        metrics_path,
        append=args.skip_test,
    )

    print("\n===== Linear SVM results =====")
    print(f"C: {args.c_value}")
    print(f"Training time: {fit_seconds:.2f} seconds")
    print(f"Validation accuracy: {val_accuracy:.2f}%")
    if test_accuracy is None:
        print("Test evaluation skipped; validation trial appended to the tuning CSV.")
    else:
        matrix_path = RESULTS_DIR / "svm_confusion_matrix.csv"
        save_confusion_matrix(matrix, matrix_path)
        print(f"Test accuracy: {test_accuracy:.2f}%")
        print(f"Test confusion matrix saved to: {matrix_path}")
    print(f"Metrics saved to: {metrics_path}")


if __name__ == "__main__":
    main()
