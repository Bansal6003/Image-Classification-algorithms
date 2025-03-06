import time
import numpy as np
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split
from torchvision.datasets import ImageFolder
from torch.cuda.amp import GradScaler, autocast
import os
import matplotlib.pyplot as plt
import mlflow
import mlflow.pytorch
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay, precision_recall_curve
import seaborn as sns
import timm

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# Allow PyTorch to leverage cuDNN for optimization
torch.backends.cudnn.benchmark = True

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class CustomEfficientNetV2(nn.Module):
    def __init__(self, num_classes=6):
        super(CustomEfficientNetV2, self).__init__()
        # Create the EfficientNetV2-S model
        self.efficientnet = timm.create_model('tf_efficientnetv2_s', pretrained=True)
        
        # Modify the classifier head
        num_features = self.efficientnet.classifier.in_features
        self.efficientnet.classifier = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, num_classes)
        )

    def forward(self, x):
        return self.efficientnet(x)

def main():
    # Timing the whole code run
    start_time = time.time()

    # MLflow setup
    mlflow.set_experiment("Efficient_NetV2-s_5epoch_test")
    mlflow.set_tracking_uri("http://127.0.0.1:5000")
    
    # Create or get the registered model
    model_name = "EfficientNetV2"
    try:
        # Just check if model exists
        client = mlflow.tracking.MlflowClient()
        try:
            client.get_registered_model(model_name)
            print(f"Found registered model: {model_name}")
        except:
            print(f"Creating new registered model: {model_name}")
            client.create_registered_model(model_name)
    except Exception as e:
        print(f"Error with model registration: {e}")

    # Start MLflow run
    with mlflow.start_run() as run:
    
        # Path to the dataset
        train_image_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V5\isolated\output_uncropped_V2'
    
        # Define transformations
        transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
        # Load dataset
        good_dataset = ImageFolder(root=train_image_path, transform=transform)
        print("Total number of samples in the original dataset:", len(good_dataset))
    
        # Hyperparameters
        BS = 16
        lr = 1e-4
        weight_decay = 1e-3
        T_max = 200
        eta_min = 1e-7
        num_epochs = 5
        patience = 10
    
        # Create the model
        model = CustomEfficientNetV2(num_classes=6)
        model = model.to(device)
    
        # Loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max, eta_min)
    
        # Mixed precision training
        scaler = torch.amp.GradScaler('cuda')
    
        # Early stopping setup
        best_val_loss = float('inf')
        patience_counter = 0
    
        # Tracking metrics
        Loss = []
        Validation_Loss = []
        Accuracy = []
        all_preds = []
        all_labels = []
    
        # Start MLflow run
        # with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "learning_rate": lr,
            "batch_size": BS,
            "num_epochs": num_epochs,
            "weight_decay": weight_decay,
            "patience": patience,
            "T_max": T_max,
            "eta_min": eta_min
        })

        # Split dataset
        dataset_classes = good_dataset.classes
        print("Classes found:", dataset_classes)
        
        # Split dataset
        train_size = int(0.8 * len(good_dataset))  
        test_size = len(good_dataset) - train_size
        train_dataset, test_dataset = random_split(good_dataset, [train_size, test_size])
        
        
        
        # Create DataLoaders
        train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

        # Training loop
        for epoch in range(num_epochs):
            model.train()
            train_loss = 0
            correct = 0
            total = 0

            # Training phase
            for img, label in train_loader:
                img, label = img.to(device), label.to(device)
                
                optimizer.zero_grad()
                
                with torch.amp.autocast('cuda'):
                    output = model(img)
                    loss = criterion(output, label)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                total += label.size(0)
                correct += predicted.eq(label).sum().item()

            scheduler.step()
            epoch_accuracy = 100. * correct / total
            Accuracy.append(epoch_accuracy)
            Loss.append(train_loss / len(train_loader))

            # Validation phase
            model.eval()
            val_loss = 0
            preds = []
            labels = []
            
            with torch.no_grad():
                for img, label in test_loader:
                    img, label = img.to(device), label.to(device)
                    
                    with torch.amp.autocast('cuda'):
                        output = model(img)
                        loss = criterion(output, label)
                        
                        preds.append(torch.argmax(output, dim=1).cpu().numpy())
                        labels.append(label.cpu().numpy())
                    
                    val_loss += loss.item()

            all_preds.extend(np.concatenate(preds))
            all_labels.extend(np.concatenate(labels))
            
            val_loss = val_loss / len(test_loader)
            Validation_Loss.append(val_loss)

            # Log metrics
            mlflow.log_metrics({
                "train_loss": train_loss / len(train_loader),
                "validation_loss": val_loss,
                "accuracy": epoch_accuracy
            }, step=epoch)

            # Print progress
            if epoch % 1 == 0:
                print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {train_loss / len(train_loader):.4f}, '
                      f'Val Loss: {val_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%')

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                model_save_path = r'D:\Behavioral genetics_V1\Metamorph_scans\WT_vs_nonWT_image_Classification\Dataset\MultiClass_Dataset_3dpf\Multiimage_dataset_V5\Efficient_NetV2-s_5epoch_test.pt'
                torch.save(model.state_dict(), model_save_path)
                
                # Log model to MLflow
                mlflow.pytorch.log_model(
                    model,
                    "model",
                    registered_model_name=model_name
                )
                mlflow.log_artifact(model_save_path)
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

        # Plot and save metrics
        plt.figure(figsize=(12, 4))
        plt.subplot(1, 2, 1)
        plt.plot(Loss, label='Training Loss')
        plt.plot(Validation_Loss, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        
        plt.subplot(1, 2, 2)
        plt.plot(Accuracy, label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy (%)')
        plt.legend()
        
        plt.tight_layout()
        plt.savefig("training_metrics.png")
        mlflow.log_artifact("training_metrics.png")

        # Classification report
        
        class_report = classification_report(all_labels, all_preds, target_names=dataset_classes)
        print("Classification Report:\n", class_report)
        with open("classification_report.txt", "w") as f:
            f.write(class_report)
        mlflow.log_artifact("classification_report.txt")

        # Confusion matrix
        # Confusion matrix
        conf_matrix = confusion_matrix(all_labels, all_preds)
        plt.figure(figsize=(10, 8))
        sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues',
            xticklabels=dataset_classes,
            yticklabels=dataset_classes)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.tight_layout()
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        
    # # End MLflow run
    # mlflow.end_run()

    print(f"Total training time: {(time.time() - start_time) / 3600:.2f} hours")

if __name__ == '__main__':
    main()