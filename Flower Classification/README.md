# Flower Classification with CNN 🌸

A deep learning project for classifying 102 different flower species using Convolutional Neural Networks (CNN) built with PyTorch.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Setup & Installation](#setup--installation)
- [Usage](#usage)
- [Training](#training)
- [Results](#results)
- [File Structure](#file-structure)
- [Requirements](#requirements)
- [Contributing](#contributing)
- [License](#license)

## 🌟 Overview

This project implements a Convolutional Neural Network to classify flower images into 102 different categories. The model is trained on the Oxford 102 Category Flower Dataset and achieves competitive accuracy through careful architecture design and optimization.

### Key Features
- **Multi-class classification**: 102 flower species
- **CNN Architecture**: Custom 3-layer convolutional network
- **Data augmentation**: Resize, normalization, and tensor conversion
- **GPU Support**: CUDA-enabled training for faster computation
- **Comprehensive evaluation**: Per-class and overall accuracy metrics

## 📊 Dataset

The project uses the **Oxford 102 Category Flower Dataset**, which contains:
- **102 flower categories** commonly occurring in the United Kingdom
- **8,189 images** total
- **Training/Validation/Test splits** pre-defined
- **High-resolution images** with varying orientations and scales

### Dataset Structure
```
102flowers/
├── jpg/                    # Image files (image_00001.jpg to image_08189.jpg)
├── imagelabels.mat        # Labels for each image (1-102)
└── setid.mat             # Train/validation/test split indices
```

### Getting the Dataset
The dataset files should be placed in the project directory:
1. Download the Oxford 102 flowers dataset
2. Extract images to `./102flowers/jpg/`
3. Place `imagelabels.mat` and `setid.mat` in the root directory

## 🏗️ Model Architecture

### CNN Structure
```
Input (3 x 224 x 224)
    ↓
Conv2d(3→32) + ReLU + MaxPool2d(2x2)     # 32 x 112 x 112
    ↓
Conv2d(32→64) + ReLU + MaxPool2d(2x2)    # 64 x 56 x 56
    ↓
Conv2d(64→128) + ReLU + MaxPool2d(2x2)   # 128 x 28 x 28
    ↓
Flatten → Linear(100352→512) + ReLU       # 512
    ↓
Linear(512→102)                           # 102 classes
```

### Model Specifications
- **Total Parameters**: ~51.5M
- **Input Size**: 224×224×3 (RGB images)
- **Output**: 102 classes (flower categories)
- **Activation**: ReLU
- **Loss Function**: CrossEntropyLoss
- **Optimizer**: SGD (lr=0.001, momentum=0.9)

## 🚀 Setup & Installation

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (optional, but recommended)
- Git

### Installation Steps

1. **Clone the repository**
   ```bash
   git clone https://github.com/doroteaMonaco/Flower-Predictor.git
   cd Flower-Predictor
   ```

2. **Install dependencies**
   ```bash
   pip install torch torchvision torchaudio
   pip install scipy matplotlib pillow numpy jupyter
   ```

3. **For CUDA support** (if you have a compatible GPU):
   ```bash
   # Check PyTorch installation with CUDA
   python -c "import torch; print('CUDA available:', torch.cuda.is_available())"
   ```

4. **Setup dataset** (see [Dataset section](#dataset))

## 💻 Usage

### Jupyter Notebook
The main implementation is in `Flowers.ipynb`. Open it with:
```bash
jupyter notebook Flowers.ipynb
```

### Key Code Sections
1. **Data Loading & Preprocessing**
   - Load MATLAB files with labels and splits
   - Create custom Dataset class
   - Apply transformations (resize, normalize)

2. **Model Definition**
   - 3-layer CNN architecture
   - Forward pass implementation

3. **Training Loop**
   - 20 epochs with loss monitoring
   - GPU acceleration (if available)
   - Model checkpointing

4. **Evaluation**
   - Per-class accuracy calculation
   - Overall model performance
   - Confusion matrix analysis

### Running Training
```python
# In the notebook or Python script
epochs = 20
model = net.cuda() if torch.cuda.is_available() else net

for epoch in range(epochs):
    # Training loop implementation
    # See Flowers.ipynb for complete code
```

## 📈 Training

### Training Configuration
- **Epochs**: 20
- **Batch Size**: 16
- **Learning Rate**: 0.001
- **Momentum**: 0.9
- **Device**: CUDA (if available) or CPU

### Data Preprocessing
```python
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
```

### Training Tips
- **GPU Recommended**: Training on CPU will be significantly slower
- **Memory Requirements**: ~2GB GPU memory for batch size 16
- **Training Time**: ~30-60 minutes on modern GPU, several hours on CPU

## 📊 Results

### Performance Metrics
- **Overall Accuracy**: [To be updated after training]
- **Training Loss**: Monitored per epoch
- **Per-class Accuracy**: Detailed breakdown for all 102 flower categories

### Model Evaluation
The model provides:
- Total accuracy across all test samples
- Individual class accuracies for detailed analysis
- Confusion matrix for error analysis

## 📁 File Structure

```
Flower-Predictor/
├── README.md                 # This file
├── Flowers.ipynb            # Main Jupyter notebook
├── .gitignore              # Git ignore file
├── 102flowers/             # Dataset directory
│   └── jpg/               # Flower images
├── imagelabels.mat        # Flower category labels
├── setid.mat              # Train/test/validation splits
└── model.pth              # Saved model weights (after training)
```

## 📦 Requirements

### Python Packages
```
torch >= 1.9.0
torchvision >= 0.10.0
scipy >= 1.7.0
matplotlib >= 3.3.0
Pillow >= 8.0.0
numpy >= 1.21.0
jupyter >= 1.0.0
```

### System Requirements
- **RAM**: 8GB+ recommended
- **Storage**: 2GB for dataset and dependencies
- **GPU**: CUDA-compatible (optional but recommended)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. Areas for improvement:
- Data augmentation techniques
- Model architecture optimizations
- Transfer learning implementations
- Hyperparameter tuning
- Additional evaluation metrics

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Oxford Visual Geometry Group** for the 102 Category Flower Dataset
- **PyTorch Team** for the deep learning framework
- **Community contributors** and researchers in computer vision

## 📞 Contact

**Author**: Dorotea Monaco  
**GitHub**: [@doroteaMonaco](https://github.com/doroteaMonaco)  
**Project Link**: [https://github.com/doroteaMonaco/Flower-Predictor](https://github.com/doroteaMonaco/Flower-Predictor)

---

*Built with ❤️ and PyTorch*