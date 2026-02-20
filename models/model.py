"""
Model architectures for multi-label image classification.
"""

import torch
import torch.nn as nn
import torchvision.models as models


class MultiLabelResNet(nn.Module):
    """
    ResNet-based model for multi-label classification.
    Uses transfer learning with pretrained ResNet.
    """
    
    def __init__(self, num_labels=4, pretrained=True, model_name='resnet50'):
        """
        Args:
            num_labels: Number of output labels
            pretrained: Whether to use pretrained weights
            model_name: ResNet variant ('resnet18', 'resnet34', 'resnet50', 'resnet101')
        """
        super(MultiLabelResNet, self).__init__()
        
        # Load pretrained ResNet
        if model_name == 'resnet18':
            self.backbone = models.resnet18(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet34':
            self.backbone = models.resnet34(pretrained=pretrained)
            num_features = 512
        elif model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            num_features = 2048
        elif model_name == 'resnet101':
            self.backbone = models.resnet101(pretrained=pretrained)
            num_features = 2048
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Remove the final fully connected layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-1])
        
        # Add custom classifier for multi-label classification
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, x):
        features = self.backbone(x)
        output = self.classifier(features)
        return output


class MultiLabelEfficientNet(nn.Module):
    """
    EfficientNet-based model for multi-label classification.
    """
    
    def __init__(self, num_labels=4, pretrained=True, model_name='efficientnet_b0'):
        """
        Args:
            num_labels: Number of output labels
            pretrained: Whether to use pretrained weights
            model_name: EfficientNet variant
        """
        super(MultiLabelEfficientNet, self).__init__()
        
        # Load pretrained EfficientNet
        if model_name == 'efficientnet_b0':
            self.backbone = models.efficientnet_b0(pretrained=pretrained)
            num_features = 1280
        elif model_name == 'efficientnet_b1':
            self.backbone = models.efficientnet_b1(pretrained=pretrained)
            num_features = 1280
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # Replace classifier
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_labels)
        )
        
    def forward(self, x):
        return self.backbone(x)


class MultiLabelLoss(nn.Module):
    """
    Custom loss function for multi-label classification with missing labels.
    Only computes loss for available labels (non-NaN).
    """
    
    def __init__(self):
        super(MultiLabelLoss, self).__init__()
        self.bce_loss = nn.BCEWithLogitsLoss(reduction='none')
    
    def forward(self, predictions, targets, mask):
        """
        Args:
            predictions: Model predictions (logits) of shape (batch_size, num_labels)
            targets: Ground truth labels of shape (batch_size, num_labels)
            mask: Binary mask indicating available labels (batch_size, num_labels)
                  1 for available, 0 for missing
        
        Returns:
            Averaged loss over available labels only
        """
        # Compute BCE loss for all predictions
        loss = self.bce_loss(predictions, targets)
        
        # Apply mask to only consider available labels
        masked_loss = loss * mask
        
        # Average over available labels only
        num_available_labels = mask.sum()
        
        if num_available_labels > 0:
            return masked_loss.sum() / num_available_labels
        else:
            return masked_loss.sum()  # Should not happen in practice


def create_model(model_type='resnet50', num_labels=4, pretrained=True):
    """
    Factory function to create models.
    
    Args:
        model_type: Type of model ('resnet18', 'resnet34', 'resnet50', 'resnet101', 
                    'efficientnet_b0', 'efficientnet_b1')
        num_labels: Number of output labels
        pretrained: Whether to use pretrained weights
    
    Returns:
        PyTorch model
    """
    if 'resnet' in model_type:
        model = MultiLabelResNet(
            num_labels=num_labels,
            pretrained=pretrained,
            model_name=model_type
        )
    elif 'efficientnet' in model_type:
        model = MultiLabelEfficientNet(
            num_labels=num_labels,
            pretrained=pretrained,
            model_name=model_type
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


if __name__ == '__main__':
    # Test model creation
    model = create_model('resnet50', num_labels=4)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Test forward pass
    dummy_input = torch.randn(4, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
    
    # Test loss function
    criterion = MultiLabelLoss()
    targets = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1]], dtype=torch.float32)
    mask = torch.tensor([[1, 1, 0, 1], [1, 0, 1, 1]], dtype=torch.float32)
    predictions = torch.randn(2, 4)
    
    loss = criterion(predictions, targets, mask)
    print(f"Test loss: {loss.item():.4f}")
