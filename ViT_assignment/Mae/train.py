import torch
import torch.optim as optim
import torchvision.transforms as transforms
import numpy as np
import os
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder

from model import MaskedAutoencoderViT

def train_mae_model():
    # Tiny ImageNet 데이터셋 경로 설정
    data_dir = './tiny-imagenet-200'  # Tiny ImageNet 데이터를 저장한 경로

    # 데이터셋 전처리 설정 (Tiny ImageNet은 64x64 해상도)
    transform = transforms.Compose([
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),  # Tiny ImageNet 이미지를 적절히 크롭
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4802, 0.4481, 0.3975), (0.2302, 0.2265, 0.2262))  # Tiny ImageNet 평균 및 표준편차로 정규화
    ])

    # Tiny ImageNet 데이터셋 로드
    train_dataset = ImageFolder(os.path.join(data_dir, 'train'), transform=transform)
    val_dataset = ImageFolder(os.path.join(data_dir, 'val'), transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)

    # 모델 생성 (Tiny ImageNet에 맞춰 patch_size를 8로 설정)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MaskedAutoencoderViT(img_size=64, patch_size=8, embed_dim=512, depth=12, num_heads=8, 
                                 decoder_embed_dim=256, decoder_depth=4, decoder_num_heads=8).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.05)

    # 학습 루프
    epochs = 50
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, _ in train_loader:
            images = images.to(device)

            # MAE 모델의 출력 및 손실 계산
            loss, _, _ = model(images)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}')
    imagenet_mean = np.array([0.485, 0.456, 0.406])
    imagenet_std = np.array([0.229, 0.224, 0.225])
    import sys
    from PIL import Image
    
    def show_image(image, title=''):
        # image is [H, W, 3]
        assert image.shape[2] == 3
        plt.imshow(torch.clip((image * imagenet_std + imagenet_mean) * 255, 0, 255).int())
        plt.title(title, fontsize=16)
        plt.axis('off')
        return
    
    def run_one_image(img, model, device):
        x = torch.tensor(img).to(device)
    
        # make it a batch-like
        x = x.unsqueeze(dim=0)
        x = torch.einsum('nhwc->nchw', x)
    
        # run MAE
        loss, y, mask = model(x.float(), mask_ratio=0.75)
        y = model.unpatchify(y)
        y = torch.einsum('nchw->nhwc', y).detach().cpu()
    
        # visualize the mask
        mask = mask.detach()
        mask = mask.unsqueeze(-1).repeat(1, 1, model.patch_embed.patch_size[0]**2 *3)  # (N, H*W, p*p*3)
        mask = model.unpatchify(mask)  # 1 is removing, 0 is keeping
        mask = torch.einsum('nchw->nhwc', mask).detach().cpu()
        
        x = torch.einsum('nchw->nhwc', x).cpu()
    
        # masked image
        im_masked = x * (1 - mask)
    
        # MAE reconstruction pasted with visible patches
        im_paste = x * (1 - mask) + y * mask
    
        # make the plt figure larger
        plt.rcParams['figure.figsize'] = [24, 24]
    
        plt.subplot(1, 4, 1)
        show_image(x[0], "original")
    
        plt.subplot(1, 4, 2)
        show_image(im_masked[0], "masked")
    
        plt.subplot(1, 4, 3)
        show_image(y[0], "reconstruction")
    
        plt.subplot(1, 4, 4)
        show_image(im_paste[0], "reconstruction + visible")
        plt.savefig('result.png')
        
    sample_image_fp = 'tiny-imagenet-200/test/images/test_1006.JPEG'
    img = Image.open(sample_image_fp).convert("RGB")
    img = np.array(img) / 255
    img = img - imagenet_mean
    img = img / imagenet_std
    run_one_image(img, model, device)

if __name__ == "__main__":
    train_mae_model()
