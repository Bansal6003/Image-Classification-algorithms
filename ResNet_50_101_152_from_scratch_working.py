import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler
from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from torch.utils.data.dataset import Subset
from torch import nn
from torch.utils.data import DataLoader, random_split


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

##############################################################################
#Step 1 Define Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride = 1, downsample = None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size = 1, stride = stride, padding = 0),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 3, stride = 1, padding = 1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU())
        self.conv3 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size = 1, stride = 1, padding = 0),
                        nn.BatchNorm2d(out_channels))
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        
    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out

##############################################################################
#Step 2 Define ResNet block
class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes = 2):
        super(ResNet, self).__init__()
        self.inplanes = 64
        self.conv1 = nn.Sequential(
                        nn.Conv2d(1, 64, kernel_size = 7, stride = 2, padding = 3),
                        nn.BatchNorm2d(64),
                        nn.ReLU())
        self.maxpool = nn.MaxPool2d(kernel_size = 3, stride = 2, padding = 1)
        self.layer0 = self._make_layer(block, 64, layers[0], stride = 1)
        self.layer1 = self._make_layer(block, 128, layers[1], stride = 2)
        self.layer2 = self._make_layer(block, 256, layers[2], stride = 2)
        self.layer3 = self._make_layer(block, 512, layers[3], stride = 2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512, num_classes)
        
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes:
            
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes, kernel_size=1, stride=stride),
                nn.BatchNorm2d(planes),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x

# Timing the whole code run
start_time = time.time()


torch.manual_seed(41)


train_image_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\DATAset_V3\output'

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),  # Ensure the image is grayscale
    transforms.ToTensor()
])

good_dataset = ImageFolder(root=train_image_path, transform=transform)

# x, y = good_dataset[0]
# print("Image Shape:", x.shape)
# print("Label:", y)

# for img, _ in good_dataset:
#     plt.imshow(img.squeeze(0), cmap='gray')
#     plt.show()
#     break

train_dataset, test_dataset = random_split(good_dataset, [int(0.95 * len(good_dataset)), len(good_dataset) - int(0.95 * len(good_dataset))])

print("Total number of samples in the original dataset:", len(good_dataset))
print("Number of samples in the training subset:", len(train_dataset))
print("Number of samples in the testing subset:", len(test_dataset))

BS = 64

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BS,  shuffle=True)

image_batch, label_batch = next(iter(train_loader))

print(f'Shape of input images: {image_batch.shape}')
print(f'Shape of labels: {label_batch.shape}')

# grid = torchvision.utils.make_grid(image_batch[0:4], padding=5, nrow=4)
# plt.imshow(grid.permute(1, 2, 0))
# plt.title('Good Samples')
# plt.show()

# model = ResNet(ResidualBlock, [3, 4, 6, 3]).to(device) #ResNet 50
# model = ResNet(ResidualBlock, [3, 4, 23, 3]).to(device) #ResNet 101
model = ResNet(ResidualBlock, [3, 8, 36, 3]).to(device) #ResNet 152
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0001) 

# Add patience for early stopping
patience = 100  # Number of epochs to wait before stopping if no improvement
best_val_loss = float('inf')  # Initialize the best validation loss as infinity
epochs_no_improve = 0  # Count epochs with no improvement

# Train the model
total_step = len(train_loader)

Loss = []
Validation_Loss = []

num_epochs = 500
for epoch in tqdm(range(num_epochs)):
    model.train()
    train_loss = 0
    for img, label in train_loader:
        img, label = img.to(device), label.to(device)

        output = model(img)
        loss = criterion(output, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
    Loss.append(train_loss / len(train_loader.dataset))

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for img, label in test_loader:
            img, label = img.to(device), label.to(device)
            output = model(img)
            loss = criterion(output, label)
            val_loss += loss.item()
    Validation_Loss.append(val_loss / len(test_loader.dataset))

    if epoch % 1 == 0:
        print('Epoch [{}/{}], Loss: {:.4f}, Validation Loss: {:.4f}'.format(
            epoch + 1, num_epochs, train_loss / len(train_loader.dataset), val_loss / len(test_loader.dataset)
        ))

    # Early stopping logic
    if val_loss < best_val_loss:
        best_val_loss = val_loss  # Update the best validation loss
        epochs_no_improve = 0  # Reset the counter
        torch.save(model.state_dict(), r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\DATAset_V3\ResNet152_WT_vs_nonWT.pt')  # Save the best model
    else:
        epochs_no_improve += 1  # Increment the counter

    if epochs_no_improve == patience:
        print(f'Early stopping triggered after {epoch+1} epochs.')
        break

# Plot the losses
plt.plot(Loss, label='Training Loss')
plt.plot(Validation_Loss, label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.show()

# torch.save(model.state_dict(), r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\ResNet152_WT_vs_nonWT.pt')
# model.eval()

#############################################################################
#Testing Block
# with torch.no_grad():
#     correct = 0
#     total = 0
#     for images, labels in test_loader:
#         images = images.to(device)
#         labels = labels.to(device)
#         outputs = model(images)
#         _, predicted = torch.max(outputs.data, 1)
#         total += labels.size(0)
#         correct += (predicted == labels).sum().item()
#         del images, labels, outputs

#     print('Accuracy of the network on the {} test images: {} %'.format(10000, 100 * correct / total))   


# Calculate and print elapsed time
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Total execution time: {elapsed_time:.2f} seconds")
























