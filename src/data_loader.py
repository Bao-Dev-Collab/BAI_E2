from pathlib import Path

import torch
from torch.utils.data import DataLoader


def create_dataloaders(
    data_dir: str | Path,
    batch_size: int,
    num_workers: int = 0,
    seed: int = 42,
) -> tuple[DataLoader, DataLoader, DataLoader, DataLoader, list[str]]:
    """Build augmented-train, train-eval, validation, and test loaders."""
    from torchvision import datasets, transforms

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")
    if num_workers < 0:
        raise ValueError("num_workers cannot be negative.")

    data_path = Path(data_dir)
    split_paths = {split: data_path / split for split in ("train", "val", "test")}
    for split, split_path in split_paths.items():
        if not split_path.is_dir():
            raise FileNotFoundError(
                f"Missing '{split}' directory: {split_path}. "
                "Expected data_dir/train, data_dir/val, and data_dir/test."
            )

    imagenet_mean = (0.485, 0.456, 0.406)
    imagenet_std = (0.229, 0.224, 0.225)
    image_size = 224

    train_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(imagenet_mean, imagenet_std),
        ]
    )

    train_dataset = datasets.ImageFolder(split_paths["train"], transform=train_transform)
    train_eval_dataset = datasets.ImageFolder(
        split_paths["train"], transform=evaluation_transform
    )
    val_dataset = datasets.ImageFolder(split_paths["val"], transform=evaluation_transform)
    test_dataset = datasets.ImageFolder(split_paths["test"], transform=evaluation_transform)
    for split, dataset in (
        ("train", train_dataset),
        ("train-eval", train_eval_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        if len(dataset) == 0:
            raise ValueError(f"The '{split}' split contains no readable images.")

    class_names = train_dataset.classes
    for split, dataset in (
        ("train-eval", train_eval_dataset),
        ("val", val_dataset),
        ("test", test_dataset),
    ):
        if dataset.classes != class_names:
            raise ValueError(
                f"Class folders in '{split}' must match train exactly. "
                f"Train: {class_names}; {split}: {dataset.classes}."
            )

    generator = torch.Generator().manual_seed(seed)
    pin_memory = torch.cuda.is_available()
    common_loader_args = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        generator=generator,
        **common_loader_args,
    )
    train_eval_loader = torch.utils.data.DataLoader(
        train_eval_dataset,
        shuffle=False,
        **common_loader_args,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        shuffle=False,
        **common_loader_args,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        shuffle=False,
        **common_loader_args,
    )
    return train_loader, train_eval_loader, val_loader, test_loader, class_names
