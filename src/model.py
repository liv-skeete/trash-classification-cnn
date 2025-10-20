# -*- coding: utf-8 -*-
"""Model architecture for trash classification."""

import torch
import torch.nn as nn
import timm


class TrashClassifier(nn.Module):
    """Trash classifier using EfficientNetV2 backbone."""
    
    def __init__(self, model_name='tf_efficientnetv2_s', num_classes=3, pretrained=True):
        super(TrashClassifier, self).__init__()
        self.model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
        
        # Replace the classifier head
        if hasattr(self.model, 'classifier'):
            self.model.classifier = nn.Linear(self.model.classifier.in_features, num_classes)
        elif hasattr(self.model, 'head'):
            self.model.head = nn.Linear(self.model.head.in_features, num_classes)
        else:
            raise AttributeError("Model does not have a recognized classifier attribute")

    def forward(self, x):
        return self.model(x)


def create_model(model_name='tf_efficientnetv2_s', num_classes=3, pretrained=True):
    """Create and return a trash classifier model."""
    model = TrashClassifier(model_name, num_classes, pretrained)
    return model


def load_model(model_path, model_name='tf_efficientnetv2_s', num_classes=3, device='cpu'):
    """Load a trained model from a checkpoint."""
    model = create_model(model_name, num_classes)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()
    return model