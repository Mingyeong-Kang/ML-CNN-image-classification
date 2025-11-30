## resnet18_finetune.py
## Machine Learning Team Project - Ewha 
## Original author: Sanna Ascard-Soederstroem

import torch.nn as nn
import torchvision.models as models

class ResNet18(nn.Module):
    def __init__(self, num_classes=10, pretrained=True):
        super(ResNet18, self).__init__()

        self.model = models.resnet18(pretrained=pretrained)

        ## replace the final layer (1000 → 10)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)
