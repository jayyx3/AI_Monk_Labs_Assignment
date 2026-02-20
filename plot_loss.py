"""
Script to plot training loss curve as required by assignment.
Generates plot with specific formatting:
- xlabel='iteration_number'
- ylabel='training_loss'  
- title='AimonK_multilabel_problem'
"""

import json
import argparse
import matplotlib.pyplot as plt
import numpy as np


def plot_training_loss(history_path, output_path='training_loss_plot.png'):
    """
    Plot training loss curve from training history.
    
    Args:
        history_path: Path to training_history.json
        output_path: Path to save the plot
    """
    # Load training history
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    
    # Create iteration numbers (per epoch)
    iterations = list(range(1, len(train_losses) + 1))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(iterations, train_losses, marker='o', linewidth=2, markersize=6, color='blue')
    
    # Set labels as required
    plt.xlabel('iteration_number', fontsize=12)
    plt.ylabel('training_loss', fontsize=12)
    plt.title('AimonK_multilabel_problem', fontsize=14, fontweight='bold')
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3)
    
    # Save the plot
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Training loss plot saved to: {output_path}")
    
    # Also display
    plt.show()


def plot_detailed_loss(history_path, output_path='detailed_loss_plot.png'):
    """
    Plot both training and validation loss.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)
    
    train_losses = history['train_losses']
    val_losses = history.get('val_losses', [])
    
    iterations = list(range(1, len(train_losses) + 1))
    
    plt.figure(figsize=(12, 6))
    plt.plot(iterations, train_losses, marker='o', label='Training Loss', linewidth=2)
    
    if val_losses:
        plt.plot(iterations, val_losses, marker='s', label='Validation Loss', linewidth=2)
    
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Detailed loss plot saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Plot training loss curves')
    parser.add_argument('--history', type=str, required=True,
                       help='Path to training_history.json file')
    parser.add_argument('--output', type=str, default='training_loss_plot.png',
                       help='Output path for the plot')
    parser.add_argument('--detailed', action='store_true',
                       help='Also create detailed plot with train and val loss')
    
    args = parser.parse_args()
    
    # Create the required plot
    plot_training_loss(args.history, args.output)
    
    # Optionally create detailed plot
    if args.detailed:
        detailed_path = args.output.replace('.png', '_detailed.png')
        plot_detailed_loss(args.history, detailed_path)


if __name__ == '__main__':
    main()
