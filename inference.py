"""
Inference script that takes an image as input and prints the list of attributes present.
As required by the assignment.
"""

import argparse
import torch
from PIL import Image
import numpy as np

from utils.data import get_transforms
from models.model import create_model


def predict_single_image(image_path, model_path, model_type='resnet50', threshold=0.5):
    """
    Predict attributes for a single image.
    
    Args:
        image_path: Path to input image
        model_path: Path to trained model checkpoint
        model_type: Model architecture type
        threshold: Classification threshold (default: 0.5)
    
    Returns:
        List of attribute indices that are present (1-indexed)
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model = create_model(model_type=model_type, num_labels=4, pretrained=False)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Load and preprocess image
    transform = get_transforms(augment=False)
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Predict
    with torch.no_grad():
        output = model(image_tensor)
        probabilities = torch.sigmoid(output).cpu().numpy()[0]
    
    # Apply threshold to get binary predictions
    predictions = (probabilities >= threshold).astype(int)
    
    # Get list of present attributes (1-indexed as per assignment table)
    present_attributes = []
    for i, pred in enumerate(predictions):
        if pred == 1:
            present_attributes.append(i + 1)  # 1-indexed: Attr1, Attr2, Attr3, Attr4
    
    return present_attributes, probabilities


def main():
    parser = argparse.ArgumentParser(
        description='Inference code that takes an image and prints list of attributes present'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b1'],
                       help='Model architecture type')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold (default: 0.5)')
    parser.add_argument('--verbose', action='store_true',
                       help='Print probabilities as well')
    
    args = parser.parse_args()
    
    # Run inference
    print(f"Loading model from: {args.model}")
    print(f"Processing image: {args.image}")
    print()
    
    present_attributes, probabilities = predict_single_image(
        args.image, args.model, args.model_type, args.threshold
    )
    
    # Print results as required
    print("="*60)
    print("INFERENCE RESULTS")
    print("="*60)
    print(f"\nList of attributes present: {present_attributes}")
    
    if args.verbose:
        print(f"\nDetailed predictions:")
        for i, prob in enumerate(probabilities):
            status = "✓ Present" if prob >= args.threshold else "✗ Absent"
            print(f"  Attribute {i+1}: {prob:.4f} - {status}")
    
    print("\n" + "="*60)
    
    # Return for programmatic use
    return present_attributes


if __name__ == '__main__':
    main()
