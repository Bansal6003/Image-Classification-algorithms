import mlflow
import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torchvision.models.resnet import ResNeXt101_32X8D_Weights
import torchvision
import os
import matplotlib.pyplot as plt
from random import random, randint
from mlflow import log_metric, log_param, log_params, log_artifacts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
mlflow.autolog()


# Allow PyTorch to leverage cuDNN for optimization
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Define the main training process
def main():
    # Timing the whole code run
    start_time = time.time()

    # Path to the dataset
    train_image_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\cropped_test\output'

    # Define transformations (including grayscale)
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
        transforms.ToTensor()
    ])

    # Load dataset
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    # Split dataset into training and testing subsets
    train_dataset, test_dataset = random_split(good_dataset, [int(0.99 * len(good_dataset)), len(good_dataset) - int(0.99 * len(good_dataset))])

    print("Total number of samples in the original dataset:", len(good_dataset))
    print("Number of samples in the training subset:", len(train_dataset))
    print("Number of samples in the testing subset:", len(test_dataset))

    # Define batch size
    BS = 64  # Larger batch size to fully utilize GPU memory

    # Dataloaders with multiple workers and pinned memory for faster data loading
    train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True, num_workers=4, pin_memory=True)

    # Load the ResNeXt-101 model with correct pre-trained weights
    model = torchvision.models.resnext101_32x8d(weights=ResNeXt101_32X8D_Weights.IMAGENET1K_V1)

    # Modify the first convolutional layer to accept 1 input channel instead of 3
    model.conv1 = nn.Conv2d(in_channels=1, 
                            out_channels=model.conv1.out_channels, 
                            kernel_size=model.conv1.kernel_size, 
                            stride=model.conv1.stride, 
                            padding=model.conv1.padding, 
                            bias=False)

    # Modify the final fully connected layer to output 2 classes
    model.fc = nn.Linear(2048, 5)

    # Move the model to GPU
    model = model.to(device)

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001)

    # Use mixed precision training
    scaler = torch.amp.GradScaler()

    # Early stopping parameters
    # patience = 20
    best_val_loss = float('inf')
    # epochs_no_improve = 0

    # Lists to track loss for plotting
    Loss = []
    Validation_Loss = []

    # Number of epochs to train
    num_epochs = 200

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0

        # Training phase with mixed precision
        for img, label in train_loader:
            img, label = img.to(device), label.to(device)

            optimizer.zero_grad()

            # Mixed precision forward pass
            with torch.cuda.amp.autocast():
                output = model(img)
                loss = criterion(output, label)

            # Mixed precision backward pass
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_loss += loss.item()

        Loss.append(train_loss / len(train_loader.dataset))

        # Validation phase
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for img, label in test_loader:
                img, label = img.to(device), label.to(device)

                with torch.cuda.amp.autocast():  # Use mixed precision in validation too
                    output = model(img)
                    loss = criterion(output, label)

                val_loss += loss.item()

        Validation_Loss.append(val_loss / len(test_loader.dataset))

        # Print losses for this epoch
        if epoch % 1 == 0:
            print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(
                epoch + 1, num_epochs, train_loss / len(train_loader.dataset), val_loss / len(test_loader.dataset)
            ))

        # Early stopping logic
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # epochs_no_improve = 0
            torch.save(model.state_dict(), r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\DATAset_V3\output\ResNext101_Classification_WT_nonWT_V3.pt')  # Save the best model
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve == patience:
        #     print(f'Early stopping triggered after {epoch+1} epochs.')
        #     break

    # Plot training and validation losses
    plt.plot(Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

# Safe entry point for Windows and other environments
if __name__ == '__main__':
    main()
