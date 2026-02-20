# Multi-Label Image Classification with Missing Labels

A robust PyTorch implementation for multi-label image classification that gracefully handles missing labels (NA values) in the training data.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train model (produces model weights + loss plot automatically)
python train.py --epochs 30 --batch_size 32

# 3. Run inference on single image (prints attributes)
python inference.py --image dataset/images/image_0.jpg --model checkpoints/resnet50_*/best_model.pth --model_type resnet50

# 4. Evaluate on test set
python evaluate.py --model_path checkpoints/resnet50_*/best_model.pth --model_type resnet50 --split test

# 5. Plot training loss (optional - already generated during training)
python plot_loss.py --loss_file checkpoints/resnet50_*/training_losses.json --output loss_plot.png
```

## 📋 Problem Statement

This project tackles multi-label image classification where:
- Each image can have multiple labels (binary classification for each label)
- Some labels may be missing (NA) for certain images
- The model must learn to predict all labels while only training on available ones

## 🌟 Key Features

- **Missing Label Handling**: Custom loss function that only computes gradients for available labels
- **Transfer Learning**: Leverages pretrained models (ResNet, EfficientNet) for better performance
- **Class Imbalance Handling**: BCE loss naturally handles imbalanced datasets
- **Comprehensive Metrics**: Evaluates performance using multiple metrics suitable for multi-label classification
- **Data Augmentation**: Applies various augmentation techniques to improve generalization
- **Flexible Architecture**: Easy to switch between different backbones

## 🔬 Technical Approach

### 1. Handling Missing Values (NA)

**Problem**: Some images have missing attribute labels (marked as "NA").

**Solution**: Implemented a mask-based approach:
- Each sample has a binary mask indicating which labels are available
- Custom `MultiLabelLoss` only computes loss for available labels:
  ```python
  loss = BCE(predictions, targets) * mask
  loss = loss.sum() / mask.sum()  # Average only over available labels
  ```
- This prevents the model from learning incorrect patterns from missing data
- Gradients are only backpropagated for known labels

### 2. Handling Class Imbalance

**Problem**: Dataset may have different numbers of positive/negative examples per attribute.

**Solutions Implemented**:
1. **Binary Cross-Entropy Loss**: Naturally handles imbalanced data by computing loss per sample
2. **Data Augmentation**: Increases effective dataset size and helps with rare classes
3. **Per-Label Metrics**: Track performance individually for each attribute
4. **Dropout Regularization**: Prevents overfitting to majority class

### 3. Data Preprocessing & Augmentations

**Preprocessing Pipeline**:
```python
# Training (with augmentation)
- Resize to 224x224
- Random horizontal flip (p=0.5)
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation ±20%)
- Convert to tensor
- Normalize with ImageNet statistics

# Validation/Test (without augmentation)
- Resize to 224x224
- Convert to tensor
- Normalize with ImageNet statistics
```

**Why These Augmentations?**:
- **Horizontal flip**: Images may appear flipped in real-world scenarios
- **Rotation**: Handles slight camera angle variations
- **Color jitter**: Accounts for different lighting conditions
- **ImageNet normalization**: Required for transfer learning from pretrained models

**Data Split Strategy**:
- Training: 70% (for learning)
- Validation: 15% (for hyperparameter tuning)
- Test: 15% (for final evaluation)

## 📁 Project Structure

```
.
├── dataset/
│   ├── images/           # Image files (975 images)
│   └── labels.txt        # Labels file with format: filename label1 label2 label3 label4
├── models/
│   ├── __init__.py
│   └── model.py          # Model architectures and custom loss function
├── utils/
│   ├── __init__.py
│   └── data.py           # Data loading, preprocessing, and augmentations
├── checkpoints/          # Saved model checkpoints (created during training)
├── results/              # Evaluation results (created during evaluation)
├── train.py              # Training script (generates model + loss plot)
├── inference.py          # Inference script for single image prediction
├── evaluate.py           # Evaluation script with comprehensive metrics
├── plot_loss.py          # Loss visualization script
├── demo.ipynb            # Interactive Jupyter notebook demo
├── requirements.txt      # Python dependencies
├── .gitignore            # Git ignore rules
└── README.md             # This file
```

## 🚀 Getting Started

### Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)

### Installation

1. Clone this repository:
```bash
git clone <repository-url>
cd "AiMonk Labs Project Assignment"
```

2. Create a virtual environment (optional but recommended):
```bash
python -m venv venv
# On Windows
venv\Scripts\activate
# On Linux/Mac
source venv/bin/activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

### Dataset Format

The labels file (`dataset/labels.txt`) should have the following format:
```
image_0.jpg 1 NA 0 1
image_1.jpg NA 0 0 0
image_2.jpg 1 1 0 0
...
```

Where:
- First column: image filename
- Columns 2-5: Binary labels (0 or 1) or NA for missing

## 🎯 Training

### Basic Training

Train with default parameters:
```bash
python train.py
```

### Advanced Training Options

```bash
python train.py \
    --model_type resnet50 \
    --batch_size 32 \
    --epochs 30 \
    --learning_rate 0.001 \
    --train_size 0.7 \
    --val_size 0.15 \
    --output_dir checkpoints
```

### Available Arguments

- `--model_type`: Model architecture (resnet18, resnet34, resnet50, resnet101, efficientnet_b0, efficientnet_b1)
- `--batch_size`: Batch size for training (default: 32)
- `--epochs`: Number of training epochs (default: 30)
- `--learning_rate`: Learning rate (default: 0.001)
- `--weight_decay`: L2 regularization weight (default: 1e-4)
- `--train_size`: Proportion of data for training (default: 0.7)
- `--val_size`: Proportion of data for validation (default: 0.15)
- `--num_workers`: Number of data loading workers (default: 4)
- `--output_dir`: Directory to save checkpoints (default: checkpoints)

## � Inference

Run inference on a single image:
```bash
python inference.py \
    --image dataset/images/image_0.jpg \
    --model checkpoints/resnet50_*/best_model.pth \
    --model_type resnet50
```

**Output Format**:
```
Loading model from: checkpoints/resnet50_20260221_120000/best_model.pth
Processing image: dataset/images/image_0.jpg
============================================================
INFERENCE RESULTS
============================================================
List of attributes present: [1, 3, 4]
============================================================
```

### Inference Arguments

- `--image`: Path to input image (required)
- `--model`: Path to trained model checkpoint (required)
- `--model_type`: Model architecture (default: resnet50)
- `--threshold`: Classification threshold (default: 0.5)
- `--verbose`: Show detailed output with probabilities

**With verbose output**:
```bash
python inference.py --image dataset/images/test.jpg --model model.pth --verbose
```
## 📉 Loss Visualization

The training script automatically generates a loss plot. You can also manually create/customize the plot:

```bash
python plot_loss.py \
    --loss_file checkpoints/resnet50_*/training_losses.json \
    --output custom_loss_plot.png
```

**Generated plot format**:
- X-axis: iteration_number
- Y-axis: training_loss
- Title: AimonK_multilabel_problem

The plot is automatically saved during training in the checkpoint directory.
## �📊 Evaluation

Evaluate a trained model:
```bash
python evaluate.py \
    --model_path checkpoints/resnet50_20260221_120000/best_model.pth \
    --model_type resnet50 \
    --split test \
    --save_predictions
```

### Evaluation Metrics

The evaluation script computes:

1. **Overall Metrics**:
   - Subset Accuracy: Percentage of samples with all labels correct
   - Hamming Loss: Fraction of wrong labels
   - Jaccard Score: Intersection over Union
   - ROC-AUC: Area under ROC curve

2. **Sample-wise Metrics** (averaged across samples):
   - Precision, Recall, F1 Score

3. **Label-wise Metrics** (macro average across labels):
   - Accuracy, Precision, Recall, F1 Score

4. **Per-Label Metrics**:
   - Individual metrics for each of the 4 labels

## 🔧 Technical Details

### Custom Loss Function

The `MultiLabelLoss` class implements a masked Binary Cross-Entropy loss:
```python
loss = BCE(predictions, targets) * mask
loss = loss.sum() / mask.sum()
```

This ensures gradients are only computed for available labels, preventing the model from learning incorrect patterns from missing data.

### Data Augmentation

Training augmentations include:
- Random horizontal flip
- Random rotation (±15°)
- Color jitter (brightness, contrast, saturation)
- Normalization with ImageNet statistics

### Model Architecture

Models use transfer learning with pretrained weights:
1. **Backbone**: Pretrained ResNet or EfficientNet (frozen or fine-tuned)
2. **Classifier**: Custom head with dropout for regularization
3. **Output**: 4 sigmoid outputs (one per label)

## 📈 Results

Example results on test set:

```
Overall Metrics:
  Subset Accuracy: 0.XXX
  Hamming Loss: 0.XXX
  Jaccard Score: 0.XXX

Label-wise Metrics (macro average):
  Accuracy: 0.XXX
  Precision: 0.XXX
  Recall: 0.XXX
  F1 Score: 0.XXX
```

*(Run training and evaluation to see actual results)*

## 🎓 Key Implementation Highlights

1. **Missing Value Handling**: 
   - Uses mask tensors to track available vs missing labels
   - Loss computation only considers available labels
   - Evaluation metrics properly handle missing ground truth

2. **Data Split Strategy**:
   - 70% training, 15% validation, 15% test (default)
   - Stratified to maintain label distribution
   - Reproducible with fixed random seed

3. **Training Features**:
   - Learning rate scheduling (ReduceLROnPlateau)
   - Early stopping based on validation loss
   - Checkpoint saving and resumption
   - Progress bars with tqdm

4. **Evaluation Features**:
   - Multiple metrics for comprehensive assessment
   - Per-label analysis
   - Visualization of metrics
   - Export predictions to CSV

## 🐛 Troubleshooting

### CUDA Out of Memory
- Reduce `--batch_size`
- Use a smaller model (e.g., resnet18 instead of resnet50)

### Slow Training
- Increase `--num_workers` (but not more than CPU cores)
- Ensure data is on SSD, not HDD
- Use mixed precision training (requires code modification)

### Poor Performance
- Train for more epochs
- Try different learning rates
- Increase data augmentation
- Use a larger model

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@software{multilabel_classification_2026,
  author = {Jay Joshi},
  title = {Multi-Label Image Classification with Missing Labels},
  year = {2026},
  url = {https://github.com/jayyx3}
}
```

## 📄 License

This project is licensed under the MIT License.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

**Note**: This project was developed as part of the ML Engineer technical assessment for AiMonk Labs.

## 📧 Contact

**Jay Joshi**

- 📧 Email: joshijayy421@gmail.com
- 📱 Phone: +91 8875549960
- 💼 LinkedIn: [https://www.linkedin.com/in/jay-joshi](https://www.linkedin.com/in/jay-joshi)
- 🐙 GitHub: [https://github.com/jayyx3](https://github.com/jayyx3)
- 🌐 Portfolio: [https://jay-portfolio-ten-tawny.vercel.app/](https://jay-portfolio-ten-tawny.vercel.app/)

---

*Developed with ❤️ for AiMonk Labs Technical Assessment*
