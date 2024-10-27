import os
import time
import torch
import argparse
import torch.nn as nn
import torchvision.transforms as tfs
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from torchvision.datasets.cifar import CIFAR10

from model import ViT


def main():
    # Argparser
    parer = argparse.ArgumentParser()
    parer.add_argument('--epoch', type=int, default=50)
    parer.add_argument('--batch_size', type=int, default=128)
    parer.add_argument('--lr', type=float, default=0.001)
    parer.add_argument('--step_size', type=int, default=100)
    parer.add_argument('--root', type=str, default='./CIFAR10')
    parer.add_argument('--name', type=str, default='vit_cifar10')
    parer.add_argument('--rank', type=int, default=0)
    ops = parer.parse_args()

    # Device
    device = torch.device('cuda:{}'.format(0) if torch.cuda.is_available() else 'cpu')

    # Dataset
    transform_cifar = tfs.Compose([
        tfs.RandomCrop(32, padding=4),
        tfs.RandomHorizontalFlip(),
        tfs.ToTensor(),
        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                      std=(0.2023, 0.1994, 0.2010)),
    ])

    test_transform_cifar = tfs.Compose([tfs.ToTensor(),
                                        tfs.Normalize(mean=(0.4914, 0.4822, 0.4465),
                                                      std=(0.2023, 0.1994, 0.2010)),
                                        ])
    train_set = CIFAR10(root=ops.root,
                        train=True,
                        download=True,
                        transform=transform_cifar)

    test_set = CIFAR10(root=ops.root,
                       train=False,
                       download=True,
                       transform=test_transform_cifar)

    train_loader = DataLoader(dataset=train_set,
                              shuffle=True,
                              batch_size=ops.batch_size)

    test_loader = DataLoader(dataset=test_set,
                             shuffle=False,
                             batch_size=ops.batch_size)

    # Model
    model = ViT().to(device)

    # Criterion
    criterion = nn.CrossEntropyLoss()

    # Optimizer 
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=ops.lr,
                                 weight_decay=5e-5)

    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=ops.epoch, eta_min=1e-5)

    # Loss and Accuracy
    train_losses = []
    test_losses = []
    accuracies = []

    # Train
    print("training...")
    for epoch in range(ops.epoch):

        model.train()
        epoch_loss = 0

        for idx, (img, target) in enumerate(train_loader):
            img = img.to(device)  # [N, 3, 32, 32]
            target = target.to(device)  # [N]
            output = model(img)  # [N, 10]
            loss = criterion(output, target)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item()

        train_losses.append(epoch_loss / len(train_loader))

        # Test
        print(f'Evaluation of epoch [{epoch}]')
        model.eval()
        correct = 0
        losses = 0
        total = 0
        with torch.no_grad():
            for idx, (img, target) in enumerate(test_loader):
                img = img.to(device)  # [N, 3, 32, 32]
                target = target.to(device)  # [N]
                output = model(img)  # [N, 10]
                loss = criterion(output, target)

                output = torch.softmax(output, dim=1)
                _, pred = output.max(-1)
                correct += torch.eq(target, pred).sum().item()
                total += target.size(0)
                losses += loss.item()

        accuracy = correct / total
        losses = losses / len(test_loader)
        accuracies.append(accuracy)
        test_losses.append(losses)

        print(f'Epoch {epoch} test: accuracy: {accuracy * 100:.2f}%, avg_loss: {losses:.4f}')

        scheduler.step()

    # Plot loss
    plt.figure()
    plt.plot(range(ops.epoch), train_losses, label='Train Loss')
    plt.plot(range(ops.epoch), test_losses, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Train and Test Loss')
    plt.legend()
    plt.savefig("ViT_loss.png")

    # Plot accuracy 
    plt.figure()
    plt.plot(range(ops.epoch), accuracies, label='Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Test Accuracy')
    plt.legend()
    plt.savefig("ViT_acc.png")


if __name__ == '__main__':
    main()
