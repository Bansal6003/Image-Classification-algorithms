import torch
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import ImageFolder
import pytorch_lightning as pl
from torchvision.models import efficientnet_v2_s

from dataclasses import dataclass
from torch.nn import nn
from typing import List, Dict, Any, Optional
import yaml
import logging
import hydra
from omegaconf import DictConfig
import wandb
from pathlib import Path
import mlflow.pytorch
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import WandbLogger, MLFlowLogger
import torch.nn.functional as F
from torch.utils.data import DataLoader
import timm

@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int
    learning_rate: float
    num_epochs: int
    weight_decay: float
    patience: int
    model_name: str
    num_classes: int
    image_size: int
    data_path: str
    experiment_name: str
    
    
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
    
class EfficientNetLightningModule(LightningModule):
    """PyTorch Lightning module for EfficientNet."""
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.save_hyperparameters()
        self.config = config
        self.model = CustomEfficientNetV2(num_classes=config.num_classes)
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        # Log metrics
        self.log('train_loss', loss, on_step=True, on_epoch=True)
        self.log('train_acc', acc, on_step=True, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.criterion(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        
        # Log metrics
        self.log('val_loss', loss, on_epoch=True)
        self.log('val_acc', acc, on_epoch=True)
        return {'val_loss': loss, 'val_acc': acc}
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.config.learning_rate,
            weight_decay=self.config.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.config.num_epochs,
            eta_min=1e-7
        )
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss'
            }
        }

class DataModule(pl.LightningDataModule):
    def __init__(self, config: TrainingConfig):
        super().__init__()
        self.config = config
        self.transform = transforms.Compose([
            transforms.Resize((config.image_size, config.image_size)),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
    def setup(self, stage=None):
        dataset = ImageFolder(root=self.config.data_path, transform=self.transform)
        train_size = int(0.8 * len(dataset))
        val_size = len(dataset) - train_size
        self.train_dataset, self.val_dataset = random_split(
            dataset, [train_size, val_size]
        )
        
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,
            pin_memory=True
        )
        
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=4,
            pin_memory=True
        )

@hydra.main(config_path="configs", config_name="config")
def main(cfg: DictConfig):
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    # Initialize config
    config = TrainingConfig(**cfg.training)
    
    # Initialize wandb
    wandb.init(
        project="larva-classification",
        name=config.experiment_name,
        config=cfg
    )
    
    # Set up MLflow
    mlflow.set_experiment(config.experiment_name)
    mlflow.pytorch.autolog()
    
    # Initialize loggers
    wandb_logger = WandbLogger(project="larva-classification")
    mlf_logger = MLFlowLogger(experiment_name=config.experiment_name)
    
    # Initialize callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath='checkpoints',
        filename='{epoch}-{val_loss:.2f}',
        monitor='val_loss',
        mode='min',
        save_top_k=3
    )
    
    early_stopping = EarlyStopping(
        monitor='val_loss',
        patience=config.patience,
        mode='min'
    )
    
    # Initialize model and data module
    model = EfficientNetLightningModule(config)
    data_module = DataModule(config)
    
    # Initialize trainer
    trainer = Trainer(
        max_epochs=config.num_epochs,
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        devices=1,
        logger=[wandb_logger, mlf_logger],
        callbacks=[checkpoint_callback, early_stopping],
        precision=16,  # Mixed precision training
        deterministic=True
    )
    
    # Train model
    trainer.fit(model, data_module)
    
    # Save model artifacts
    model_path = Path("models") / f"{config.experiment_name}.pt"
    model_path.parent.mkdir(exist_ok=True)
    torch.save(model.state_dict(), model_path)
    
    # Log final metrics and artifacts
    wandb.log({
        "best_val_loss": trainer.callback_metrics["val_loss"].item(),
        "best_val_acc": trainer.callback_metrics["val_acc"].item()
    })
    
    wandb.finish()
    
if __name__ == "__main__":
    main()
