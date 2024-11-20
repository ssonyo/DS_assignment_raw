import torch
import torch.nn as nn


class CustomCLIPClassifier(nn.Module):
    def __init__(self, clip_model):
        super(CustomCLIPClassifier, self).__init__()
        self.clip_model = clip_model
        self.classifier = nn.Linear(512, 90)  # Assuming 90 classes, adjust accordingly

    def forward(self, images):
        features = self.clip_model.encode_image(images)
        return self.classifier(features.float())