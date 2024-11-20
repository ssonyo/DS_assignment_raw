import torch
import torch.nn as nn
import clip
from torch.utils.data import DataLoader
from PIL import Image
import numpy as np
from tqdm import tqdm
from datasets import load_from_disk
from model import CustomCLIPClassifier
from utils import CustomDataset, SupConLoss

# Load CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)


# 마지막 vit layer만 재학습
for name, param in model.named_parameters():
    if "visual.transformer.resblocks.11" in name:  # 마지막 ViT 레이어만 재학습
        param.requires_grad = True


# Initialize custom classifier model
classifier_model = CustomCLIPClassifier(model).to(device)
optimizer = torch.optim.Adam(classifier_model.classifier.parameters(), lr=1e-4)
criterion = nn.CrossEntropyLoss()

dataset = load_from_disk("/root/DS_assignment_raw/Representational-Learning/dataset/train")
custom_dataset = CustomDataset(dataset, preprocess)
dataloader = DataLoader(custom_dataset, batch_size=64, shuffle=True)

contrastive_loss = SupConLoss(temperature=0.07)

# Training loop
classifier_model.train()
for epoch in range(3):  # Adjust epoch count as needed
    total_loss = 0
    for images, labels in tqdm(dataloader):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        features = classifier_model.clip_model.encode_image(images)
        outputs = classifier_model.classifier(features.float())
        
        # Compute supervised contrastive loss
        loss = contrastive_loss(features, labels)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(classifier_model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Epoch {epoch + 1}, Contrastive Loss: {total_loss / len(dataloader)}")


# Save the trained model
model_save_path = "/root/DS_assignment_raw/Representational-Learning/saved_model.pth"
torch.save(classifier_model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")
