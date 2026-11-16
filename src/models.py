import torch.nn as nn

if __package__:
    from .densenet121_model import build_densenet121
    from .vgg16_model import build_vgg16
else:
    from densenet121_model import build_densenet121
    from vgg16_model import build_vgg16


def build_model(
    model_name: str,
    num_classes: int,
    fine_tune_last_block: bool = False,
) -> nn.Module:
    """Create the requested pretrained classifier for the target classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    builders = {
        "vgg16": build_vgg16,
        "densenet121": build_densenet121,
    }
    try:
        builder = builders[model_name]
    except KeyError as error:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose 'vgg16' or 'densenet121'."
        ) from error

    return builder(num_classes, fine_tune_last_block)


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total
