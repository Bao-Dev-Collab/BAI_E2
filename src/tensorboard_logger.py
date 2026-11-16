from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Mapping

if TYPE_CHECKING:
    from torch.utils.tensorboard import SummaryWriter


def create_summary_writer(log_dir: str | Path) -> SummaryWriter:
    from torch.utils.tensorboard import SummaryWriter

    return SummaryWriter(log_dir=str(log_dir))


def log_epoch_metrics(
    writer: SummaryWriter,
    epoch: int,
    train_metrics: Mapping[str, float],
    train_eval_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    learning_rate: float,
    batch_size: int,
) -> None:
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


def log_test_metrics(
    writer: SummaryWriter,
    test_metrics: Mapping[str, float],
    epoch: int,
) -> None:
    writer.add_scalar("Loss/test", test_metrics["loss"], epoch)
    writer.add_scalar("Accuracy/test", test_metrics["accuracy"], epoch)


def flush_summary_writer(writer: SummaryWriter) -> None:
    writer.flush()
