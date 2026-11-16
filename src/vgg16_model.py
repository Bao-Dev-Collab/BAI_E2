import torch.nn as nn


def build_vgg16(
    num_classes: int,
    fine_tune_last_block: bool = False,
) -> nn.Module:
    """Build an ImageNet-pretrained VGG16 adapted to the target classes."""
    if num_classes < 2:
        raise ValueError("num_classes must be at least 2.")

    from torchvision import models

    model = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
    in_features = model.classifier[6].in_features
    model.classifier[6] = nn.Linear(in_features, num_classes)

    for parameter in model.features.parameters():
        parameter.requires_grad = False
    for parameter in model.classifier[:6].parameters():
        parameter.requires_grad = False
    for parameter in model.classifier[6].parameters():
        parameter.requires_grad = True

    if fine_tune_last_block:
        for parameter in model.features[24:].parameters():
            parameter.requires_grad = True

    return model
