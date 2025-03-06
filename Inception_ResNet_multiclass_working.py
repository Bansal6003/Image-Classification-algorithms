import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
from torchvision.models import vit_b_16, ViT_B_16_Weights
import torchvision
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import seaborn as sns
from sklearn.model_selection import KFold
from torchvision import models
import timm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Allow PyTorch to leverage cuDNN for optimization
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

mlflow.set_experiment("Inception_ResNet_isolated_images_V11")
mlflow.set_tracking_uri("http://127.0.0.1:5000")

class CustomInceptionV3(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomInceptionV3, self).__init__()
        # Create the InceptionResnetV2 model
        self.inception = timm.create_model('inception_resnet_v2', pretrained=True)
        
        # Modify the final fully connected layer to output the correct number of classes
        self.inception.classif = nn.Linear(self.inception.classif.in_features, num_classes)

    def forward(self, x):
        return self.inception(x)

# Define the main training process
def main():
    # Timing the whole code run
    start_time = time.time()

    # Path to the dataset
    train_image_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V4\isolated\output'

    # Define transformations (including grayscale)
    transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.Grayscale(num_output_channels=3),  # Ensure the image is grayscale
        transforms.ToTensor()
    ])

    # Load dataset
    good_dataset = ImageFolder(root=train_image_path, transform=transform)

    print("Total number of samples in the original dataset:", len(good_dataset))

    # Define batch size
    BS = 32  # Larger batch size to fully utilize GPU memory

    # Create the custom model
    model = CustomInceptionV3(num_classes=6)

    # Move the model to GPU
    model = model.to(device)

    lr = 0.0001 #0.000001

    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr, weight_decay=1e-4)

    # Use mixed precision training
    scaler = torch.cuda.amp.GradScaler()

    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 20  # How many epochs to wait before stopping when no improvement
    patience_counter = 0  # Counter to track number of epochs without improvement

    # Lists to track loss for plotting
    Loss = []
    Validation_Loss = []
    all_preds = []
    all_labels = []

    # Number of epochs to train
    num_epochs = 300

    # Initialize MLflow and start the experiment
    mlflow.start_run()

    # Log parameters
    mlflow.log_param("learning_rate", lr)
    mlflow.log_param("batch_size", BS)
    mlflow.log_param("num_epochs", num_epochs)

    # K-Fold Cross Validation setup
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(np.arange(len(good_dataset)))):
        print(f'Fold {fold+1}/{kf.n_splits}')

        # Split data
        train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
        test_sampler = torch.utils.data.SubsetRandomSampler(test_idx)

        train_loader = DataLoader(good_dataset, batch_size=BS, sampler=train_sampler)
        test_loader = DataLoader(good_dataset, batch_size=BS, sampler=test_sampler)

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
            preds = []
            labels = []
            
            with torch.no_grad():
                for img, label in test_loader:
                    img, label = img.to(device), label.to(device)
            
                    with torch.cuda.amp.autocast():
                        output = model(img)
                         # Extract the main output for loss computation
                        loss = criterion(output, label)
                      # Ensure label is compatible with output shape
                        preds.append(torch.argmax(output, dim=1).cpu().numpy())  # Take argmax for class predictions
                        labels.append(label.cpu().numpy())


                    val_loss += loss.item()

            all_preds.extend(np.concatenate(preds))
            all_labels.extend(np.concatenate(labels))

            Validation_Loss.append(val_loss / len(test_loader.dataset))

            # Log training and validation loss to MLflow
            mlflow.log_metric("train_loss", train_loss / len(train_loader.dataset), step=epoch)
            mlflow.log_metric("validation_loss", val_loss / len(test_loader.dataset), step=epoch)

            # Print losses for this epoch
            if epoch % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader.dataset):.4f}, Validation Loss: {val_loss / len(test_loader.dataset):.4f}')

            # Early stopping logic
            min_delta = 1e-4  # Minimum improvement threshold
            if (best_val_loss - val_loss) > min_delta:
                best_val_loss = val_loss
                patience_counter = 0 # Reset patience counter
                # Log the model as an artifact
                model_save_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V4\Inception_ResNet_V11_isolated_images_6_classes.pt'
                torch.save(model.state_dict(), model_save_path)
                mlflow.log_artifact(model_save_path)
            else:
                patience_counter += 1  # Increment patience counter
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break  # Stop training if patience is exceeded

    # Plot training and validation losses
    plt.plot(Loss, label='Training Loss')
    plt.plot(Validation_Loss, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig("loss_plot.png")

    # Log the loss plot as an artifact in MLflow
    mlflow.log_artifact("loss_plot.png")

    # Classification report and confusion matrix
    class_report = classification_report(all_labels, all_preds, target_names=good_dataset.classes)
    print("Classification Report:\n", class_report)

    # Log classification report to a text file in MLflow
    with open("classification_report.txt", "w") as f:
        f.write(class_report)
    mlflow.log_artifact("classification_report.txt")

    # Confusion matrix
    conf_matrix = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(conf_matrix, display_labels=good_dataset.classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")

    # Precision-Recall curve
    precisions, recalls, thresholds = precision_recall_curve(all_labels, all_preds)
    plt.figure()
    plt.plot(recalls, precisions, marker='.', label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.savefig("precision_recall_curve.png")
    mlflow.log_artifact("precision_recall_curve.png")

    # End the MLflow run
mlflow.end_run()


# Safe entry point for Windows and other environments
if __name__ == '__main__':
    main()
