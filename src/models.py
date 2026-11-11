import torch.nn as nn


def build_model(
    model_name: str,
    num_classes: int,
    fine_tune_last_block: bool = False,
) -> nn.Module:
    """Create an ImageNet-pretrained classifier adapted to the target classes."""
    from torchvision import models

    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    if model_name == "vgg16":
        model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
        in_features = model.classifier[6].in_features
        model.classifier[6] = nn.Linear(in_features, num_classes)
        for parameter in model.classifier[:6].parameters():
            parameter.requires_grad = False
        backbone_parameters = model.features.parameters()
        head_parameters = model.classifier[6].parameters()
        last_block_parameters = model.features[24:].parameters()
    elif model_name == "densenet121":
        model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
        in_features = model.classifier.in_features
        model.classifier = nn.Linear(in_features, num_classes)
        backbone_parameters = model.features.parameters()
        head_parameters = model.classifier.parameters()
        last_block_parameters = (
            *model.features.denseblock4.parameters(),
            *model.features.norm5.parameters(),
        )
    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. Choose 'vgg16' or 'densenet121'."
        )

    for parameter in backbone_parameters:
        parameter.requires_grad = False
    for parameter in head_parameters:
        parameter.requires_grad = True
    if fine_tune_last_block:
        for parameter in last_block_parameters:
            parameter.requires_grad = True

    return model


def count_parameters(model: nn.Module) -> tuple[int, int]:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
    return trainable, total
