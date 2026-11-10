import torch
import time
from torch import nn
from torch.utils.data import DataLoader


PROGRESS_INTERVAL = 20


def _run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    training = optimizer is not None
    phase = "Train" if training else "Evaluate"
    model.train(training)
    if training:
        for module in model.modules():
            if isinstance(module, nn.modules.batchnorm._BatchNorm):
                parameters = tuple(module.parameters(recurse=False))
                if not any(parameter.requires_grad for parameter in parameters):
                    module.eval()

    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    total_batches = len(loader)
    start_time = time.monotonic()
    print(f"{phase}: processing {total_batches} batches...", flush=True)

    with torch.set_grad_enabled(training):
        for batch_index, (images, labels) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            if training:
                optimizer.zero_grad(set_to_none=True)

            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            total_correct += (logits.argmax(dim=1) == labels).sum().item()
            total_samples += batch_size

            if batch_index % PROGRESS_INTERVAL == 0 or batch_index == total_batches:
                elapsed = time.monotonic() - start_time
                print(
                    f"{phase}: batch {batch_index}/{total_batches} "
                    f"({elapsed:.1f}s elapsed)",
                    flush=True,
                )

    if total_samples == 0:
        raise ValueError("The data loader produced no samples.")

    return {
        "loss": total_loss / total_samples,
        "accuracy": total_correct / total_samples,
    }


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    return _run_epoch(model, loader, criterion, device, optimizer)


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> dict[str, float]:
    return _run_epoch(model, loader, criterion, device, optimizer=None)
