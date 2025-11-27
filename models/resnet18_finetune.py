import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


def ResNet18(num_classes=10, pretrained=True):
    """
    ImageNet으로 pretrain된 ResNet18을 불러와
    마지막 FC layer만 CIFAR-10에 맞게 교체.
    """
    if pretrained:
        weights = ResNet18_Weights.IMAGENET1K_V1
        model = resnet18(weights=weights)
    else:
        model = resnet18(weights=None)

    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model