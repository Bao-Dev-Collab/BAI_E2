from __future__ import annotations

import argparse
import csv
import importlib.util
import tempfile
import shutil
from pathlib import Path
import random
from typing import TYPE_CHECKING, Callable

import torch
from torch import nn

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter

from data_loader import create_dataloaders
from engine import evaluate, train_one_epoch
from models import build_model, count_parameters


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer learning experiments with VGG16 and DenseNet121."
    )
    parser.add_argument("--data-dir", required=True, help="Root containing train/val/test.")
    parser.add_argument(
        "--models",
        nargs="+",
        choices=("vgg16", "densenet121"),
        default=("vgg16", "densenet121"),
    )
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument(
        "--patience",
        type=int,
        default=3,
        help="Stop after this many epochs without lower validation loss.",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--tune", action="store_true", help="Run all LR/batch-size combinations.")
    parser.add_argument("--learning-rates", nargs="+", type=float, default=(0.01, 0.001))
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=(16, 32))
    parser.add_argument(
        "--fine-tune-last-block",
        action="store_true",
        help="Unfreeze the final feature block in addition to the classifier.",
    )
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")
    if args.patience < 1:
        parser.error("--patience must be at least 1.")
    if args.batch_size < 1 or any(size < 1 for size in args.batch_sizes):
        parser.error("Batch sizes must be at least 1.")
    if args.learning_rate <= 0 or any(rate <= 0 for rate in args.learning_rates):
        parser.error("Learning rates must be greater than 0.")
    if args.num_workers < 0:
        parser.error("--num-workers cannot be negative.")
    return args


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_experiment(
    model_name: str,
    learning_rate: float,
    batch_size: int,
    args: argparse.Namespace,
    device: torch.device,
    summary_writer_type: Callable[..., SummaryWriter],
    class_names: list[str] | None = None,
) -> dict[str, object]:
    set_seed(args.seed)
    (
        train_loader,
        train_eval_loader,
        val_loader,
        _,
        loader_class_names,
    ) = create_dataloaders(
        args.data_dir, batch_size=batch_size, num_workers=args.num_workers, seed=args.seed
    )
    if class_names is not None and loader_class_names != class_names:
        raise ValueError("Class order changed between experiments.")

    model = build_model(
        model_name,
        num_classes=len(loader_class_names),
        fine_tune_last_block=args.fine_tune_last_block,
    ).to(device)
    trainable, total = count_parameters(model)
    print(f"\n--- {model_name} | lr={learning_rate:g} | batch_size={batch_size} ---")
    print(f"Classes: {loader_class_names}")
    print(f"Trainable parameters: {trainable:,} / {total:,}")
    print(model)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=learning_rate,
    )
    run_name = f"{model_name}_lr{learning_rate:g}_bs{batch_size}"
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = run_dir / "best_model.pth"
    best_val_loss = float("inf")
    validation_accuracy_at_best_loss = -1.0
    train_eval_loss_at_best_loss = float("inf")
    train_eval_accuracy_at_best_loss = -1.0
    best_epoch = 0
    epochs_without_improvement = 0
    epochs_completed = 0
    best_state: dict[str, torch.Tensor] | None = None

    with summary_writer_type(log_dir=str(run_dir)) as writer:
        for epoch in range(1, args.epochs + 1):
            epochs_completed = epoch
            train_metrics = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )
            train_eval_metrics = evaluate(
                model, train_eval_loader, criterion, device
            )
            val_metrics = evaluate(model, val_loader, criterion, device)
            writer.add_scalar("Loss/train", train_metrics["loss"], epoch)
            writer.add_scalar("Loss/train_eval", train_eval_metrics["loss"], epoch)
            writer.add_scalar("Loss/validation", val_metrics["loss"], epoch)
            writer.add_scalar("Accuracy/train", train_metrics["accuracy"], epoch)
            writer.add_scalar(
                "Accuracy/train_eval", train_eval_metrics["accuracy"], epoch
            )
            writer.add_scalar("Accuracy/validation", val_metrics["accuracy"], epoch)
            writer.add_scalar("Hyperparameters/learning_rate", learning_rate, epoch)
            writer.add_scalar("Hyperparameters/batch_size", batch_size, epoch)
            print(
                f"Epoch {epoch:02d}/{args.epochs} | "
                f"train-aug loss={train_metrics['loss']:.4f}, "
                f"train-aug acc={train_metrics['accuracy']:.4f} | "
                f"train-eval loss={train_eval_metrics['loss']:.4f}, "
                f"train-eval acc={train_eval_metrics['accuracy']:.4f} | "
                f"val loss={val_metrics['loss']:.4f}, "
                f"val acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                validation_accuracy_at_best_loss = val_metrics["accuracy"]
                train_eval_loss_at_best_loss = train_eval_metrics["loss"]
                train_eval_accuracy_at_best_loss = train_eval_metrics["accuracy"]
                best_epoch = epoch
                epochs_without_improvement = 0
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                torch.save(best_state, checkpoint_path)
            else:
                epochs_without_improvement += 1
                if epochs_without_improvement >= args.patience:
                    print(
                        f"Early stopping after {epoch} epochs: validation loss "
                        f"did not improve for {args.patience} consecutive epochs."
                    )
                    break

        if best_state is None:
            raise RuntimeError("Training did not produce a best validation checkpoint.")
        writer.flush()

    print(
        f"Best validation loss={best_val_loss:.4f}, "
        f"validation accuracy at selected checkpoint="
        f"{validation_accuracy_at_best_loss:.4f} at epoch {best_epoch}"
    )
    print(f"Best checkpoint: {checkpoint_path}")
    return {
        "model": model_name,
        "learning_rate": learning_rate,
        "batch_size": batch_size,
        "best_epoch": best_epoch,
        "epochs_completed": epochs_completed,
        "best_validation_loss": best_val_loss,
        "validation_accuracy_at_best_loss": validation_accuracy_at_best_loss,
        "train_eval_loss_at_best_loss": train_eval_loss_at_best_loss,
        "train_eval_accuracy_at_best_loss": train_eval_accuracy_at_best_loss,
        "selected_by_validation": False,
        "checkpoint": str(checkpoint_path),
    }


def evaluate_selected_checkpoint(
    result: dict[str, object],
    args: argparse.Namespace,
    device: torch.device,
    summary_writer_type: Callable[..., SummaryWriter],
    class_names: list[str] | None = None,
) -> None:
    _, _, _, test_loader, loader_class_names = create_dataloaders(
        args.data_dir,
        batch_size=int(result["batch_size"]),
        num_workers=args.num_workers,
        seed=args.seed,
    )
    if class_names is not None and loader_class_names != class_names:
        raise ValueError("Class order changed before final test evaluation.")

    model = build_model(
        str(result["model"]),
        num_classes=len(loader_class_names),
        fine_tune_last_block=args.fine_tune_last_block,
    ).to(device)
    checkpoint_path = Path(str(result["checkpoint"]))
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    test_metrics = evaluate(model, test_loader, nn.CrossEntropyLoss(), device)
    result["test_loss"] = test_metrics["loss"]
    result["test_accuracy"] = test_metrics["accuracy"]

    with summary_writer_type(log_dir=str(checkpoint_path.parent)) as writer:
        writer.add_scalar("Loss/test", test_metrics["loss"], int(result["best_epoch"]))
        writer.add_scalar(
            "Accuracy/test", test_metrics["accuracy"], int(result["best_epoch"])
        )
        writer.flush()

    print(
        f"Final test for validation-selected configuration "
        f"({result['model']}, lr={result['learning_rate']}, "
        f"batch_size={result['batch_size']}): loss={test_metrics['loss']:.4f}, "
        f"accuracy={test_metrics['accuracy']:.4f}"
    )


def append_results(output_dir: Path, results: list[dict[str, object]]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "results.csv"
    fieldnames = [
        "model",
        "learning_rate",
        "batch_size",
        "best_epoch",
        "epochs_completed",
        "best_validation_loss",
        "validation_accuracy_at_best_loss",
        "train_eval_loss_at_best_loss",
        "train_eval_accuracy_at_best_loss",
        "selected_by_validation",
        "test_loss",
        "test_accuracy",
        "checkpoint",
    ]

    existing_rows: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if results_path.exists() and results_path.stat().st_size > 0:
        with results_path.open("r", newline="", encoding="utf-8") as results_file:
            reader = csv.DictReader(results_file)
            if reader.fieldnames is None:
                raise ValueError(f"CSV has no header: {results_path}")
            existing_fieldnames = reader.fieldnames
            existing_rows = list(reader)

    merged_fieldnames = list(dict.fromkeys(existing_fieldnames + fieldnames))
    if existing_fieldnames and existing_fieldnames != merged_fieldnames:
        with tempfile.NamedTemporaryFile(
            mode="w",
            newline="",
            encoding="utf-8",
            dir=output_dir,
            prefix=f".{results_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary_file:
            temporary_path = Path(temporary_file.name)
            writer = csv.DictWriter(temporary_file, fieldnames=merged_fieldnames)
            writer.writeheader()
            writer.writerows(existing_rows)
        temporary_path.replace(results_path)

    write_header = not results_path.exists() or results_path.stat().st_size == 0
    with results_path.open("a", newline="", encoding="utf-8") as results_file:
        writer = csv.DictWriter(results_file, fieldnames=merged_fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(results)
    print(f"Experiment summary: {results_path}")


def export_latest_run(output_dir: Path) -> Path:
    """Copy the newest training run into a fixed folder for quick access."""
    output_dir.mkdir(parents=True, exist_ok=True)
    latest_dir = output_dir / "latest_run"
    if latest_dir.exists():
        shutil.rmtree(latest_dir)
    latest_dir.mkdir(parents=True, exist_ok=True)

    candidate_dirs = [
        path for path in output_dir.iterdir() if path.is_dir() and path.name != "latest_run"
    ]
    newest_run_dir = max(candidate_dirs, key=lambda path: path.stat().st_mtime, default=None)

    if newest_run_dir is not None:
        shutil.copytree(newest_run_dir, latest_dir, dirs_exist_ok=True)

    summary_path = output_dir / "results.csv"
    if summary_path.exists():
        shutil.copy2(summary_path, latest_dir / "results.csv")

    metadata_path = latest_dir / "run_info.txt"
    metadata_path.write_text(
        f"Latest run exported at: {latest_dir}\n"
        f"Source directory: {newest_run_dir}\n"
        f"Generated by: main.py\n",
        encoding="utf-8",
    )
    print(f"Latest run copied to: {latest_dir}")
    return latest_dir


def main() -> None:
    args = parse_args()
    missing = [
        package
        for package in ("torchvision", "tensorboard")
        if importlib.util.find_spec(package) is None
    ]
    if missing:
        raise ModuleNotFoundError(
            f"Missing dependencies: {', '.join(missing)}. "
            "Install them with 'pip install -r requirements.txt'."
        )

    from torch.utils.tensorboard import SummaryWriter

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if args.tune:
        configurations = [
            (learning_rate, batch_size)
            for learning_rate in args.learning_rates
            for batch_size in args.batch_sizes
        ]
    else:
        configurations = [(args.learning_rate, args.batch_size)]

    results: list[dict[str, object]] = []
    expected_classes = None
    for model_name in args.models:
        model_results: list[dict[str, object]] = []
        for learning_rate, batch_size in configurations:
            result = run_experiment(
                model_name,
                learning_rate,
                batch_size,
                args,
                device,
                SummaryWriter,
                expected_classes,
            )
            if expected_classes is None:
                expected_classes = create_dataloaders(
                    args.data_dir,
                    batch_size=batch_size,
                    num_workers=args.num_workers,
                    seed=args.seed,
                )[4]
            results.append(result)
            model_results.append(result)

        selected_result = min(
            model_results, key=lambda result: float(result["best_validation_loss"])
        )
        selected_result["selected_by_validation"] = True
        evaluate_selected_checkpoint(
            selected_result,
            args,
            device,
            SummaryWriter,
            expected_classes,
        )

    append_results(args.output_dir, results)
    export_latest_run(args.output_dir)


if __name__ == "__main__":
    main()
