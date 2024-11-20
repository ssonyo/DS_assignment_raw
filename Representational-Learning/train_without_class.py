import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import UnsupervisedDataset, UnsupervisedContrastiveLoss

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
model = model.float()

for name, param in model.named_parameters():
    if "visual.transformer.resblocks.11" in name:
        param.requires_grad = True
    else:
        param.requires_grad = False

# Initialize custom classifier model
classifier_model = CustomCLIPClassifier(model).to(device)

trainable_params = list(classifier_model.classifier.parameters()) + [
    param for name, param in model.named_parameters() if param.requires_grad
]

optimizer = torch.optim.Adam(trainable_params, lr=5e-4)
criterion = nn.CrossEntropyLoss()

dataset = load_from_disk("/root/DS_assignment_raw/Representational-Learning/dataset/train")
custom_dataset = UnsupervisedDataset(dataset)
dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

contrastive_loss_fn = UnsupervisedContrastiveLoss(temperature=0.001)

# Training loop
classifier_model.train()
for epoch in range(5):
    total_loss = 0
    for image1, image2, labels in tqdm(dataloader):
        image1, image2, labels = image1.to(device), image2.to(device), labels.to(device)

        # Forward pass to extract features
        features1 = classifier_model.clip_model.encode_image(image1)
        features2 = classifier_model.clip_model.encode_image(image2)

        # Concatenate features for contrastive loss
        features = torch.cat([features1, features2], dim=0)

        # Forward pass through the classifier
        logits1 = classifier_model.classifier(features1)
        logits2 = classifier_model.classifier(features2)

        logits = torch.cat([logits1, logits2], dim=0)
        labels_concat = torch.cat([labels, labels], dim=0)

        # Compute classification loss
        classification_loss = criterion(logits, labels_concat)

        # Compute contrastive loss
        contrastive_loss_value = contrastive_loss_fn(features)

        # Combine losses
        alpha = 0.5 
        loss = classification_loss + alpha * contrastive_loss_value

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the trained model
model_save_path = "/root/DS_assignment_raw/Representational-Learning/saved_model.pth"
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
