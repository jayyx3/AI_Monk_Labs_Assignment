"""
Training script for multi-label image classification.
"""

import os
import json
import argparse
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from utils.data import load_dataset, create_dataloaders
from models.model import create_model, MultiLabelLoss


def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    """
    Train the model for one epoch.
    """
    model.train()
    running_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} - Training')
    for batch_idx, batch in enumerate(pbar):
        images = batch['image'].to(device)
        labels = batch['labels'].to(device)
        masks = batch['mask'].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels, masks)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        avg_loss = running_loss / (batch_idx + 1)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}'})
    
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """
    Validate the model.
    """
    model.eval()
    running_loss = 0.0
    
    with torch.no_grad():
        pbar = tqdm(dataloader, desc='Validation')
        for batch_idx, batch in enumerate(pbar):
            images = batch['image'].to(device)
            labels = batch['labels'].to(device)
            masks = batch['mask'].to(device)
            
            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels, masks)
            
            # Statistics
            running_loss += loss.item()
            avg_loss = running_loss / (batch_idx + 1)
            pbar.set_postfix({'val_loss': f'{avg_loss:.4f}'})
    
    return running_loss / len(dataloader)


def train_model(args):
    """
    Main training function.
    """
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_dir = os.path.join(args.output_dir, f'{args.model_type}_{timestamp}')
    os.makedirs(output_dir, exist_ok=True)
    
    # Save training arguments
    with open(os.path.join(output_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)
    
    # Load dataset
    print("\nLoading dataset...")
    dataset_splits = load_dataset(
        labels_path=args.labels_path,
        train_size=args.train_size,
        val_size=args.val_size,
        random_state=args.seed
    )
    
    # Create dataloaders
    print("\nCreating dataloaders...")
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
        pretrained=args.pretrained
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss function and optimizer
    criterion = MultiLabelLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")
    
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(1, args.epochs + 1):
        # Train
        train_loss = train_one_epoch(
            model, dataloaders['train'], criterion, optimizer, device, epoch
        )
        train_losses.append(train_loss)
        
        # Validate
        val_loss = validate(model, dataloaders['val'], criterion, device)
        val_losses.append(val_loss)
        
        # Learning rate scheduling
        scheduler.step(val_loss)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Print epoch summary
        print(f"\nEpoch {epoch}/{args.epochs}:")
        print(f"  Train Loss: {train_loss:.4f}")
        print(f"  Val Loss: {val_loss:.4f}")
        print(f"  Learning Rate: {current_lr:.6f}")
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            print(f"  New best validation loss! Saving model...")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, 'best_model.pth'))
        
        # Save checkpoint
        if epoch % args.save_every == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, os.path.join(output_dir, f'checkpoint_epoch_{epoch}.pth'))
        
        print()
    
    # Save final model
    torch.save({
        'epoch': args.epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_losses[-1],
        'val_loss': val_losses[-1],
    }, os.path.join(output_dir, 'final_model.pth'))
    
    # Save training history
    history = {
        'train_losses': train_losses,
        'val_losses': val_losses
    }
    with open(os.path.join(output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f, indent=4)
    
    # Generate the required loss plot
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    iterations = list(range(1, len(train_losses) + 1))
    plt.plot(iterations, train_losses, marker='o', linewidth=2, markersize=6, color='blue')
    plt.xlabel('iteration_number', fontsize=12)
    plt.ylabel('training_loss', fontsize=12)
    plt.title('AimonK_multilabel_problem', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_loss_plot.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*50)
    print("Training completed!")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to: {output_dir}")
    print(f"Training loss plot saved to: {os.path.join(output_dir, 'training_loss_plot.png')}")
    print("="*50)
    
    return model, history, output_dir


def main():
    parser = argparse.ArgumentParser(description='Train multi-label image classification model')
    
    # Data parameters
    parser.add_argument('--labels_path', type=str, default='dataset/labels.txt',
                       help='Path to labels file')
    parser.add_argument('--image_dir', type=str, default='dataset/images',
                       help='Directory containing images')
    parser.add_argument('--train_size', type=float, default=0.7,
                       help='Proportion of data for training')
    parser.add_argument('--val_size', type=float, default=0.15,
                       help='Proportion of data for validation')
    
    # Model parameters
    parser.add_argument('--model_type', type=str, default='resnet50',
                       choices=['resnet18', 'resnet34', 'resnet50', 'resnet101', 
                               'efficientnet_b0', 'efficientnet_b1'],
                       help='Model architecture')
    parser.add_argument('--pretrained', action='store_true', default=True,
                       help='Use pretrained weights')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32,
                       help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of training epochs')
    parser.add_argument('--learning_rate', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                       help='Weight decay for regularization')
    parser.add_argument('--num_workers', type=int, default=4,
                       help='Number of data loading workers')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed')
    
    # Output parameters
    parser.add_argument('--output_dir', type=str, default='checkpoints',
                       help='Directory to save model checkpoints')
    parser.add_argument('--save_every', type=int, default=10,
                       help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
    
    # Train model
    train_model(args)


if __name__ == '__main__':
    main()
