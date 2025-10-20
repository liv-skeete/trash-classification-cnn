# -*- coding: utf-8 -*-
"""Training script for trash classification model."""

import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.cuda import amp
from torch.optim.swa_utils import AveragedModel
import torchvision.datasets as datasets
from sklearn.metrics import classification_report
import kagglehub

from model import create_model
from utils import (
    MappedToThreeBins, 
    FocalLoss, 
    prepare_dataset, 
    get_transforms, 
    calculate_class_weights,
    BIN_LABELS
)


def download_dataset():
    """Download the dataset from Kaggle."""
    print("Downloading dataset...")
    path = kagglehub.dataset_download("alistairking/recyclable-and-household-waste-classification")
    print("Path to dataset files:", path)
    return path


def train(model, loader, optimizer, criterion, scheduler, scaler, device):
    """Training function."""
    model.train()
    correct, total, loss_sum = 0, 0, 0

    for images, labels in loader:
        # Move data to GPU
        images, labels = images.to(device), labels.to(device)

        # Reset gradients
        optimizer.zero_grad()

        # Forward pass with mixed precision
        with amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels)

        # Backpropagation with scaling
        scaler.scale(loss).backward()

        # Update weights
        scaler.step(optimizer)
        scaler.update()

        # Update learning rate scheduler
        scheduler.step()

        # Calculate accuracy
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        # Accumulate loss
        loss_sum += loss.item() * images.size(0)

    return loss_sum / total, correct / total


def evaluate(model, loader, criterion, device):
    """Evaluation function."""
    model.eval()
    correct, total, loss_sum = 0, 0, 0
    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss_sum += loss.item() * images.size(0)
            correct += (outputs.argmax(1) == labels).sum().item()
            total += labels.size(0)
            y_true.extend(labels.cpu().numpy())
            y_pred.extend(outputs.argmax(1).cpu().numpy())
    return loss_sum / total, correct / total, y_true, y_pred


def main(args):
    """Main training function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Download dataset
    dataset_path = download_dataset()
    data_dir = f'{dataset_path}/images/images'
    
    # Prepare dataset
    target_dir = 'data/flattened_dataset'
    prepare_dataset(data_dir, target_dir, args.split_ratio)

    # Get transforms
    train_transform, val_transform = get_transforms(args.image_size)

    # Get original dataset
    raw_train_dataset = datasets.ImageFolder(root=os.path.join(target_dir, 'train'), transform=train_transform)
    raw_val_dataset = datasets.ImageFolder(root=os.path.join(target_dir, 'val'), transform=val_transform)

    # Map labels to 3-class bin
    train_dataset = MappedToThreeBins(raw_train_dataset, raw_train_dataset.classes)
    val_dataset = MappedToThreeBins(raw_val_dataset, raw_val_dataset.classes)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Create model
    model = create_model(args.model_name, args.num_classes, pretrained=True)
    model = model.to(device)

    # Create optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=1e-5)
    
    # Add a learning rate scheduler
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.learning_rate * 10,  # 10x base LR
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        anneal_strategy='cos'
    )

    # We will also add early stopping and model checkpointing
    best_val_acc = 0.0
    patience_counter = 0

    # Calculate class weights
    class_weights = calculate_class_weights(train_dataset, args.num_classes)
    class_weights = class_weights.to(device)

    # Create criterion
    criterion = FocalLoss(alpha=0.25, gamma=2.0, weight=class_weights).to(device)

    # Clear cache
    torch.cuda.empty_cache()

    # Create a GradScaler for AMP
    scaler = amp.GradScaler()

    # Training loop
    print("Starting training...")
    for epoch in range(args.epochs):
        train_loss, train_acc = train(model, train_loader, optimizer, criterion, scheduler, scaler, device)
        val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
        print(f"Epoch {epoch+1}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

        # Check for improvement in validation accuracy
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            # Save the best model
            torch.save(model.state_dict(), args.model_save_path)
            print(f"Saved new best model with accuracy: {val_acc:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break

    # Load best model and evaluate
    model.load_state_dict(torch.load(args.model_save_path, map_location=device))
    _, _, y_true, y_pred = evaluate(model, val_loader, criterion, device)
    
    print("\nClassification Report:\n")
    print(classification_report(y_true, y_pred, target_names=BIN_LABELS))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train trash classification model')
    parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_s', 
                        help='Model name (default: tf_efficientnetv2_s)')
    parser.add_argument('--num-classes', type=int, default=3, 
                        help='Number of classes (default: 3)')
    parser.add_argument('--image-size', type=int, default=256, 
                        help='Image size (default: 256)')
    parser.add_argument('--batch-size', type=int, default=64, 
                        help='Batch size (default: 64)')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs (default: 20)')
    parser.add_argument('--learning-rate', type=float, default=1e-4, 
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--split-ratio', type=float, default=0.8, 
                        help='Train/val split ratio (default: 0.8)')
    parser.add_argument('--patience', type=int, default=3, 
                        help='Early stopping patience (default: 3)')
    parser.add_argument('--model-save-path', type=str, default='models/best_model.pth', 
                        help='Path to save best model (default: models/best_model.pth)')
    
    args = parser.parse_args()
    
    # Create necessary directories
    os.makedirs('models', exist_ok=True)
    os.makedirs('data', exist_ok=True)
    
    main(args)