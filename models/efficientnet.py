import torch
import torch.nn as nn
from timm import create_model


class EfficientNetB0(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super().__init__()

        # load pretrained model
        self.model = create_model(
            'efficientnet_b0',
            pretrained=pretrained,
            drop_rate=0.2,
            drop_path_rate=0.2
        )

        # replace classifier to match CIFAR-10 output
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


def get_efficientnet(num_classes=10, pretrained=True):
    return EfficientNetB0(num_classes=num_classes, pretrained=pretrained)
