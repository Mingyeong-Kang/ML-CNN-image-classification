from timm import create_model

def EfficientNetB0(num_classes=10, pretrained=True):
    model = create_model('efficientnet_b0', pretrained=pretrained)
    model.classifier = torch.nn.Linear(model.classifier.in_features, num_classes)
    return model
