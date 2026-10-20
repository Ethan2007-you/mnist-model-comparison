"""Run all MNIST experiments in sequence and build the HTML report."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from mnist_data import PROJECT_ROOT


RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_SCRIPTS = {
    "mlp": "train_nn.py",
    "early-cnn": "train_early_cnn.py",
    "modern-cnn": "train_modern_cnn.py",
    "lenet": "train_lenet.py",
    "svm": "train_svm.py",
    "random-forest": "train_random_forest.py",
}


def run_and_log(command: list[str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"\n{'=' * 72}\nRunning: {' '.join(command)}\nLog: {log_path}\n", flush=True)
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        for line in process.stdout:
            print(line, end="", flush=True)
            log_file.write(line)
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    # Windows terminals may use a legacy code page that cannot display paths
    # containing Chinese characters; keep logging from aborting on such output.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    parser = argparse.ArgumentParser(
        description="Train and test every MNIST model, then generate results/index.html."
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--models",
        nargs="+",
        choices=list(MODEL_SCRIPTS),
        default=list(MODEL_SCRIPTS),
        help="Run selected models; default runs every model.",
    )
    parser.add_argument(
        "--max-train-samples",
        type=int,
        default=50_000,
        help="Training image count for SVM and random forest (default: all 50,000).",
    )
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error("--epochs and --batch-size must be positive")
    if not 1 <= args.max_train_samples <= 50_000:
        parser.error("--max-train-samples must be between 1 and 50000")

    for model_name in args.models:
        script = MODEL_SCRIPTS[model_name]
        command = [sys.executable, "-u", str(PROJECT_ROOT / "src" / script)]
        if script in {
            "train_nn.py",
            "train_early_cnn.py",
            "train_modern_cnn.py",
            "train_lenet.py",
        }:
            command.extend(
                ["--epochs", str(args.epochs), "--batch-size", str(args.batch_size)]
            )
        elif script in {"train_svm.py", "train_random_forest.py"}:
            command.extend(["--max-train-samples", str(args.max_train_samples)])
        run_and_log(command, RESULTS_DIR / "logs" / f"{Path(script).stem}.log")

    run_and_log(
        [sys.executable, str(PROJECT_ROOT / "src" / "build_report.py")],
        RESULTS_DIR / "logs" / "build_report.log",
    )
    print(f"\nAll experiments completed. Open {RESULTS_DIR / 'index.html'}")


if __name__ == "__main__":
    main()
