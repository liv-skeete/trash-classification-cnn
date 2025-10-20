# -*- coding: utf-8 -*-
"""Utility functions for trash classification project."""

import os
import glob
import shutil
import random
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda import amp
import numpy as np
from PIL import Image
import torchvision.transforms as transforms


# Define category mappings
RECYCLABLE_SET = {
    'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_soda_bottles', 'plastic_water_bottles', 'steel_food_cans'
}

COMPOSTABLE_SET = {
    'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags'
}

LANDFILL_SET = {
    'aerosol_cans', 'clothing', 'disposable_plastic_cutlery', 'paper_cups',
    'plastic_shopping_bags', 'plastic_straws', 'plastic_trash_bags',
    'shoes', 'styrofoam_cups', 'styrofoam_food_containers'
}

BIN_LABELS = ['Recyclable', 'Compostable', 'Landfill']


class MappedToThreeBins(torch.utils.data.Dataset):
    """Dataset wrapper that maps original labels to 3-bin categories."""
    
    def __init__(self, dataset, class_names):
        self.dataset = dataset
        self.class_names = class_names
        self.label_map = self._build_label_map()

    def _build_label_map(self):
        mapping = {}
        for i, name in enumerate(self.class_names):
            if name in RECYCLABLE_SET:
                mapping[i] = 0
            elif name in COMPOSTABLE_SET:
                mapping[i] = 1
            elif name in LANDFILL_SET:
                mapping[i] = 2
            else:
                raise ValueError(f"Unknown category: {name}")
        return mapping

    def __getitem__(self, index):
        img, label = self.dataset[index]
        return img, self.label_map[label]

    def __len__(self):
        return len(self.dataset)


class FocalLoss(nn.Module):
    """Focal Loss implementation for handling class imbalance."""
    
    def __init__(self, alpha=0.25, gamma=2.0, weight=None, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss

        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        return focal_loss


def prepare_dataset(data_dir, target_dir, split_ratio=0.8):
    """Flatten dataset structure and split into train/val sets."""
    train_dir = os.path.join(target_dir, 'train')
    val_dir = os.path.join(target_dir, 'val')
    
    # Clean any previous runs
    shutil.rmtree(target_dir, ignore_errors=True)

    # Create directories
    os.makedirs(train_dir)
    os.makedirs(val_dir)

    classes = os.listdir(data_dir)
    for cls in classes:
        class_path = os.path.join(data_dir, cls)
        if not os.path.isdir(class_path): 
            continue

        # Gather all image files in nested dirs
        all_images = glob.glob(os.path.join(class_path, '**', '*.*'), recursive=True)
        all_images = [f for f in all_images if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

        random.shuffle(all_images)
        split_idx = int(len(all_images) * split_ratio)
        train_images = all_images[:split_idx]
        val_images = all_images[split_idx:]

        # Copy to new structure
        for phase, images in zip(['train', 'val'], [train_images, val_images]):
            class_target_dir = os.path.join(target_dir, phase, cls)
            os.makedirs(class_target_dir, exist_ok=True)
            for img_path in images:
                shutil.copy(img_path, class_target_dir)

    print("✅ Dataset reorganized for PyTorch ImageFolder")


def get_transforms(image_size=256):
    """Get training and validation transforms."""
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.5)
    ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def calculate_class_weights(train_dataset, num_classes):
    """Calculate class weights for handling class imbalance."""
    class_counts = Counter([label for _, label in train_dataset])
    total_samples = sum(class_counts.values())
    class_weights = torch.tensor([
        total_samples / class_counts[i] for i in range(num_classes)
    ]).float()
    return class_weights


def preprocess_image(image_path, device, image_size=256):
    """Preprocess a single image for prediction."""
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                            [0.229, 0.224, 0.225])
    ])

    img = Image.open(image_path).convert("RGB")
    img_tensor = transform(img).unsqueeze(0).to(device)  # add batch dim
    return img_tensor