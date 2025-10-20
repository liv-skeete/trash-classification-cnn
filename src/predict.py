# -*- coding: utf-8 -*-
"""Prediction interface for trash classification."""

import argparse
import torch
import matplotlib.pyplot as plt
from PIL import Image
import os

from model import load_model
from utils import preprocess_image, BIN_LABELS


def predict_image(model, image_path, device):
    """Predict the class of a single image."""
    # Preprocess image
    img_tensor = preprocess_image(image_path, device)
    
    # Predict
    with torch.no_grad():
        output = model(img_tensor)
        pred_index = output.argmax(dim=1).item()
        pred_class = BIN_LABELS[pred_index]
        
    return pred_index, pred_class


def main(args):
    """Main prediction function."""
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load model
    print("Loading model...")
    model = load_model(args.model_path, args.model_name, args.num_classes, device)
    model = model.to(device)
    model.eval()
    
    # Process image
    if args.image_path and os.path.exists(args.image_path):
        pred_index, pred_class = predict_image(model, args.image_path, device)
        
        print(f"Predicted class: {pred_class} (index: {pred_index})")
        
        # Display result
        if args.show_image:
            img = Image.open(args.image_path).convert("RGB")
            plt.imshow(img)
            plt.axis('off')
            plt.title(f"Bin: {pred_class}")
            plt.show()
    else:
        print("Please provide a valid image path.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Predict trash category')
    parser.add_argument('--model-path', type=str, required=True, 
                        help='Path to trained model')
    parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_s', 
                        help='Model name (default: tf_efficientnetv2_s)')
    parser.add_argument('--num-classes', type=int, default=3, 
                        help='Number of classes (default: 3)')
    parser.add_argument('--image-path', type=str, 
                        help='Path to image for prediction')
    parser.add_argument('--show-image', action='store_true',
                        help='Show image with prediction result')
    
    args = parser.parse_args()
    main(args)