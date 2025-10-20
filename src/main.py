# -*- coding: utf-8 -*-
"""Main interface for trash classification project."""

import argparse
import os


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Trash Classification CNN')
    subparsers = parser.add_subparsers(dest='mode', help='Mode to run')

    # Training subparser
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_s', 
                              help='Model name (default: tf_efficientnetv2_s)')
    train_parser.add_argument('--num-classes', type=int, default=3, 
                              help='Number of classes (default: 3)')
    train_parser.add_argument('--image-size', type=int, default=256, 
                              help='Image size (default: 256)')
    train_parser.add_argument('--batch-size', type=int, default=64, 
                              help='Batch size (default: 64)')
    train_parser.add_argument('--epochs', type=int, default=20, 
                              help='Number of epochs (default: 20)')
    train_parser.add_argument('--learning-rate', type=float, default=1e-4, 
                              help='Learning rate (default: 1e-4)')
    train_parser.add_argument('--split-ratio', type=float, default=0.8, 
                              help='Train/val split ratio (default: 0.8)')
    train_parser.add_argument('--patience', type=int, default=3, 
                              help='Early stopping patience (default: 3)')
    train_parser.add_argument('--model-save-path', type=str, default='models/best_model.pth', 
                              help='Path to save best model (default: models/best_model.pth)')

    # Prediction subparser
    predict_parser = subparsers.add_parser('predict', help='Predict using trained model')
    predict_parser.add_argument('--model-path', type=str, required=True, 
                                help='Path to trained model')
    predict_parser.add_argument('--model-name', type=str, default='tf_efficientnetv2_s', 
                                help='Model name (default: tf_efficientnetv2_s)')
    predict_parser.add_argument('--num-classes', type=int, default=3, 
                                help='Number of classes (default: 3)')
    predict_parser.add_argument('--image-path', type=str, 
                                help='Path to image for prediction')
    predict_parser.add_argument('--show-image', action='store_true',
                                help='Show image with prediction result')

    args = parser.parse_args()

    if args.mode == 'train':
        # Import train module and run
        from train import main as train_main
        train_main(args)
    elif args.mode == 'predict':
        # Import predict module and run
        from predict import main as predict_main
        predict_main(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()