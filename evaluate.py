"""
Evaluation and inference script for multi-label image classification.
"""

import os
import argparse
import json
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    hamming_loss, jaccard_score, roc_auc_score
)
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns

from utils.data import load_dataset, create_dataloaders
from models.model import create_model


def compute_metrics(predictions, targets, masks, threshold=0.5):
    """
    Compute various metrics for multi-label classification.
    Only considers available labels (based on mask).
    
    Args:
        predictions: Model predictions (probabilities after sigmoid)
        targets: Ground truth labels
        masks: Binary mask for available labels
        threshold: Classification threshold
    
    Returns:
        Dictionary of metrics
    """
    # Apply threshold to get binary predictions
    pred_binary = (predictions >= threshold).astype(int)
    
    # Flatten arrays and apply mask
    targets_flat = targets.flatten()
    pred_flat = pred_binary.flatten()
    masks_flat = masks.flatten().astype(bool)
    
    # Filter out missing labels
    targets_available = targets_flat[masks_flat]
    pred_available = pred_flat[masks_flat]
    pred_probs_available = predictions.flatten()[masks_flat]
    
    # Compute metrics
    metrics = {}
    
    # Subset accuracy (exact match)
    metrics['subset_accuracy'] = accuracy_score(targets, pred_binary)
    
    # Sample-wise metrics (averaged across samples)
    sample_f1 = []
    sample_precision = []
    sample_recall = []
    
    for i in range(len(targets)):
        mask_idx = masks[i].astype(bool)
        if mask_idx.sum() > 0:
            target_sample = targets[i][mask_idx]
            pred_sample = pred_binary[i][mask_idx]
            
            if target_sample.sum() > 0 or pred_sample.sum() > 0:
                sample_f1.append(f1_score(target_sample, pred_sample, zero_division=0))
                sample_precision.append(precision_score(target_sample, pred_sample, zero_division=0))
                sample_recall.append(recall_score(target_sample, pred_sample, zero_division=0))
    
    metrics['sample_f1'] = np.mean(sample_f1) if sample_f1 else 0.0
    metrics['sample_precision'] = np.mean(sample_precision) if sample_precision else 0.0
    metrics['sample_recall'] = np.mean(sample_recall) if sample_recall else 0.0
    
    # Label-wise metrics (macro average)
    label_metrics = {
        'precision': [],
        'recall': [],
        'f1': [],
        'accuracy': []
    }
    
    for label_idx in range(targets.shape[1]):
        mask_label = masks[:, label_idx].astype(bool)
        if mask_label.sum() > 0:
            target_label = targets[:, label_idx][mask_label]
            pred_label = pred_binary[:, label_idx][mask_label]
            
            label_metrics['accuracy'].append(accuracy_score(target_label, pred_label))
            label_metrics['precision'].append(precision_score(target_label, pred_label, zero_division=0))
            label_metrics['recall'].append(recall_score(target_label, pred_label, zero_division=0))
            label_metrics['f1'].append(f1_score(target_label, pred_label, zero_division=0))
    
    metrics['macro_accuracy'] = np.mean(label_metrics['accuracy']) if label_metrics['accuracy'] else 0.0
    metrics['macro_precision'] = np.mean(label_metrics['precision']) if label_metrics['precision'] else 0.0
    metrics['macro_recall'] = np.mean(label_metrics['recall']) if label_metrics['recall'] else 0.0
    metrics['macro_f1'] = np.mean(label_metrics['f1']) if label_metrics['f1'] else 0.0
    
    # Hamming loss (fraction of wrong labels)
    metrics['hamming_loss'] = hamming_loss(targets_available, pred_available)
    
    # Jaccard score (IoU)
    metrics['jaccard_score'] = jaccard_score(targets_available, pred_available, average='samples', zero_division=0)
    
    # AUC-ROC (if possible)
    try:
        if len(np.unique(targets_available)) > 1:
            metrics['roc_auc'] = roc_auc_score(targets_available, pred_probs_available)
        else:
            metrics['roc_auc'] = None
    except:
        metrics['roc_auc'] = None
    
    return metrics, label_metrics


def evaluate_model(model, dataloader, device, threshold=0.5):
    """
    Evaluate model on a dataset.
    """
    model.eval()
    
    all_predictions = []
    all_targets = []
    all_masks = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc='Evaluating'):
            images = batch['image'].to(device)
            labels = batch['labels']
            masks = batch['mask']
            
            # Forward pass
            outputs = model(images)
            
            # Apply sigmoid to get probabilities
            probs = torch.sigmoid(outputs).cpu().numpy()
            
            all_predictions.append(probs)
            all_targets.append(labels.numpy())
            all_masks.append(masks.numpy())
    
    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    masks = np.concatenate(all_masks, axis=0)
    
    # Compute metrics
    metrics, label_metrics = compute_metrics(predictions, targets, masks, threshold)
    
    return metrics, label_metrics, predictions, targets, masks


def plot_metrics(label_metrics, output_path):
    """
    Plot per-label metrics.
    """
    labels = [f'Label {i+1}' for i in range(len(label_metrics['f1']))]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Per-Label Metrics', fontsize=16, fontweight='bold')
    
    # Precision
    axes[0, 0].bar(labels, label_metrics['precision'], color='skyblue')
    axes[0, 0].set_title('Precision')
    axes[0, 0].set_ylim([0, 1])
    axes[0, 0].grid(axis='y', alpha=0.3)
    
    # Recall
    axes[0, 1].bar(labels, label_metrics['recall'], color='lightcoral')
    axes[0, 1].set_title('Recall')
    axes[0, 1].set_ylim([0, 1])
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # F1 Score
    axes[1, 0].bar(labels, label_metrics['f1'], color='lightgreen')
    axes[1, 0].set_title('F1 Score')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Accuracy
    axes[1, 1].bar(labels, label_metrics['accuracy'], color='lightyellow')
    axes[1, 1].set_title('Accuracy')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Metrics plot saved to {output_path}")


def save_predictions(filenames, predictions, targets, masks, output_path):
    """
    Save predictions to a CSV file.
    """
    results = []
    
    for i, filename in enumerate(filenames):
        row = {
            'filename': filename,
            'pred_label_1': predictions[i, 0],
            'pred_label_2': predictions[i, 1],
            'pred_label_3': predictions[i, 2],
            'pred_label_4': predictions[i, 3],
            'true_label_1': targets[i, 0] if masks[i, 0] else 'NA',
            'true_label_2': targets[i, 1] if masks[i, 1] else 'NA',
            'true_label_3': targets[i, 2] if masks[i, 2] else 'NA',
            'true_label_4': targets[i, 3] if masks[i, 3] else 'NA',
        }
        results.append(row)
    
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    print(f"Predictions saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate multi-label image classification model')
    
    # Model parameters
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101',
                               'efficientnet_b0', 'efficientnet_b1'],
                       help='Model architecture')
    
    # Data parameters
    parser.add_argument('--labels_path', type=str, default='dataset/labels.txt',
                       help='Path to labels file')
    parser.add_argument('--image_dir', type=str, default='dataset/images',
                       help='Directory containing images')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'val', 'test'],
                       help='Which split to evaluate')
    
    # Evaluation parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for evaluation')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--threshold', type=float, default=0.5,
                       help='Classification threshold')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='results',
                       help='Directory to save results')
    parser.add_argument('--save_predictions', action='store_true',
                       help='Save predictions to CSV')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_splits = load_dataset(
        labels_path=args.labels_path,
        train_size=0.7,
        val_size=0.15,
        random_state=42
    )
    
    # Create dataloader
    print(f"\nCreating {args.split} dataloader...")
    dataloaders = create_dataloaders(
        dataset_splits,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        image_dir=args.image_dir
    )
    
    # Create model
    print(f"\nCreating {args.model_type} model...")
    model = create_model(
        model_type=args.model_type,
        num_labels=4,
        pretrained=False
    )
    
    # Load checkpoint
    print(f"\nLoading checkpoint from {args.model_path}...")
    checkpoint = torch.load(args.model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    
    # Evaluate
    print(f"\nEvaluating on {args.split} set...")
    metrics, label_metrics, predictions, targets, masks = evaluate_model(
        model, dataloaders[args.split], device, args.threshold
    )
    
    # Print metrics
    print("\n" + "="*50)
    print("Evaluation Results")
    print("="*50)
    print(f"\nOverall Metrics:")
    print(f"  Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"  Hamming Loss: {metrics['hamming_loss']:.4f}")
    print(f"  Jaccard Score: {metrics['jaccard_score']:.4f}")
    if metrics['roc_auc'] is not None:
        print(f"  ROC-AUC: {metrics['roc_auc']:.4f}")
    
    print(f"\nSample-wise Metrics (averaged across samples):")
    print(f"  Precision: {metrics['sample_precision']:.4f}")
    print(f"  Recall: {metrics['sample_recall']:.4f}")
    print(f"  F1 Score: {metrics['sample_f1']:.4f}")
    
    print(f"\nLabel-wise Metrics (macro average):")
    print(f"  Accuracy: {metrics['macro_accuracy']:.4f}")
    print(f"  Precision: {metrics['macro_precision']:.4f}")
    print(f"  Recall: {metrics['macro_recall']:.4f}")
    print(f"  F1 Score: {metrics['macro_f1']:.4f}")
    
    print(f"\nPer-Label Metrics:")
    for i in range(len(label_metrics['f1'])):
        print(f"  Label {i+1}:")
        print(f"    Accuracy: {label_metrics['accuracy'][i]:.4f}")
        print(f"    Precision: {label_metrics['precision'][i]:.4f}")
        print(f"    Recall: {label_metrics['recall'][i]:.4f}")
        print(f"    F1 Score: {label_metrics['f1'][i]:.4f}")
    
    # Save metrics to JSON
    metrics_path = os.path.join(args.output_dir, f'{args.split}_metrics.json')
    with open(metrics_path, 'w') as f:
        # Convert to serializable format
        metrics_save = {k: (v if v is not None else 'N/A') for k, v in metrics.items()}
        json.dump(metrics_save, f, indent=4)
    print(f"\nMetrics saved to {metrics_path}")
    
    # Plot metrics
    plot_path = os.path.join(args.output_dir, f'{args.split}_metrics.png')
    plot_metrics(label_metrics, plot_path)
    
    # Save predictions if requested
    if args.save_predictions:
        pred_path = os.path.join(args.output_dir, f'{args.split}_predictions.csv')
        filenames = dataset_splits[args.split]['filenames']
        save_predictions(filenames, predictions, targets, masks, pred_path)
    
    print("\n" + "="*50)
    print("Evaluation completed!")
    print("="*50)


if __name__ == '__main__':
    main()
