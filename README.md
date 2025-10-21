# Trash Classification CNN

CNN-based waste classifier using EfficientNetV2 to sort trash into recyclable, compostable, and landfill bins. Trained with focal loss and mixed precision for 30-class → 3-bin classification.

## Overview

Our planet generates over 2 billion tons of waste annually, with a significant portion mismanaged due to contamination in recycling streams. Incorrect sorting at the household level leads to entire batches of recyclables being sent to landfills. This project addresses this critical environmental challenge by employing deep learning to automatically classify waste items into the correct disposal bins, thereby reducing contamination and improving recycling efficiency.

## Key Features

* **EfficientNetV2-S backbone (pretrained)** - Leverages state-of-the-art CNN architecture with ImageNet pretraining for optimal feature extraction
* **Focal loss for class imbalance handling** - Addresses the inherent imbalance in waste categories with a modulated loss function that focuses learning on hard examples
* **Mixed precision training (torch.cuda.amp)** - Reduces memory usage and accelerates training by utilizing both FP16 and FP32 numerical formats
* **OneCycleLR learning rate scheduling** - Dynamically adjusts learning rate for faster convergence and improved performance
* **Early stopping and model checkpointing** - Prevents overfitting and saves the best performing model during training
* **30-class to 3-bin mapping strategy** - Consolidates fine-grained waste categories into actionable disposal bins
* **Interactive prediction interface** - Real-time classification of waste items through a user-friendly interface

## Architecture

The model uses a mapping strategy that consolidates 30 specific waste categories into 3 functional disposal bins:

### Recyclable Bin (16 classes)
```python
recyclable_set = {
    'aluminum_food_cans', 'aluminum_soda_cans', 'cardboard_boxes',
    'cardboard_packaging', 'glass_beverage_bottles', 'glass_cosmetic_containers',
    'glass_food_jars', 'magazines', 'newspaper', 'office_paper',
    'plastic_cup_lids', 'plastic_detergent_bottles', 'plastic_food_containers',
    'plastic_soda_bottles', 'plastic_water_bottles', 'steel_food_cans'
}
```

### Compostable Bin (4 classes)
```python
compostable_set = {
    'coffee_grounds', 'eggshells', 'food_waste', 'tea_bags'
}
```

### Landfill Bin (10 classes)
```python
landfill_set = {
    'aerosol_cans', 'clothing', 'disposable_plastic_cutlery', 'paper_cups',
    'plastic_shopping_bags', 'plastic_straws', 'plastic_trash_bags',
    'shoes', 'styrofoam_cups', 'styrofoam_food_containers'
}
```

## Technical Implementation

### Focal Loss Implementation
To address class imbalance inherent in waste datasets, we implement Focal Loss which modulates the cross-entropy loss:
```python
class FocalLoss(nn.Module):
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
```

### Mixed Precision Training Setup
We utilize PyTorch's Automatic Mixed Precision (AMP) for efficient training:
```python
# Create a GradScaler for AMP
scaler = amp.GradScaler()

# Training with mixed precision
with amp.autocast():
    outputs = model(images)
    loss = criterion(outputs, labels)

# Backpropagation with scaling
scaler.scale(loss).backward()
scaler.step(optimizer)
scaler.update()
```

### Model Architecture
The model leverages EfficientNetV2-S with a custom classifier head:
```python
model = timm.create_model('tf_efficientnetv2_s', pretrained=True, num_classes=NUM_CLASSES)
model.classifier = nn.Linear(model.classifier.in_features, NUM_CLASSES)
```

## Usage/Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/trash-classification-cnn.git
   cd trash-classification-cnn
   ```

2. Install requirements:
   ```bash
   pip install -r requirements.txt
   ```

3. Download the dataset:
   ```bash
   # Configure Kaggle API credentials
   # Dataset will be automatically downloaded during first run
   ```

4. Train the model:
   ```bash
   python src/train.py
   ```

5. Run predictions:
   ```bash
   python src/predict.py
   ```

## Dataset

This project uses the [Recyclable and Household Waste Classification](https://www.kaggle.com/datasets/alistairking/recyclable-and-household-waste-classification) dataset from Kaggle, which contains over 15,000 images across 30 waste categories.

## Authors
- Liv Skeete | liv@di.st
- Daniel Zhang
- Levi Reese  
- Chana Fink

*UCLA COSMOS 2025*

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project demonstrates the application of deep learning to environmental challenges, showcasing how technology can contribute to sustainable waste management practices.*
