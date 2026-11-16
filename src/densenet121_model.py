import torch.nn as nn


def build_densenet121(
    num_classes: int,
    fine_tune_last_block: bool = False,
) -> nn.Module:
    """Build an ImageNet-pretrained DenseNet121 adapted to the target classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    from torchvision import models

    model = models.densenet121(weights=models.DenseNet121_Weights.DEFAULT)
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_classes)

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True

    if fine_tune_last_block:
        for parameter in model.features.denseblock4.parameters():
            parameter.requires_grad = True
        for parameter in model.features.norm5.parameters():
            parameter.requires_grad = True

    return model
