# CIFAR-10 Enhanced Neural Network Classification 🖼️

An advanced implementation of a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch, featuring improved architecture design and training strategies that achieve **78.51% test accuracy**.

## 🎯 Project Overview

This project implements an enhanced CNN architecture for multi-class image classification on the CIFAR-10 dataset, demonstrating significant performance improvements through modern deep learning techniques including BatchNormalization, Dropout regularization, and optimized training procedures.

## 📊 Key Results

- **Test Accuracy**: **78.51%** (significant improvement from baseline ~54%)
- **Training Epochs**: 30 epochs with stable convergence
- **Architecture**: Enhanced 3-layer CNN with modern techniques
- **Dataset**: 60,000 32×32 color images across 10 classes

## 🏗️ Architecture Improvements

### Enhanced CNN Design:
```python
class Net(nn.Module):
    - Conv1: 3→32 channels (3×3, padding=1) + BatchNorm + ReLU + MaxPool
    - Conv2: 32→64 channels (3×3, padding=1) + BatchNorm + ReLU + MaxPool  
    - Conv3: 64→128 channels (3×3, padding=1) + BatchNorm + ReLU + MaxPool
    - FC1: 128×4×4 → 512 (with Dropout 0.5)
    - FC2: 512 → 10 classes
```

### Key Improvements from Baseline:
1. **BatchNormalization**: Added after each convolutional layer for stable training
2. **Increased Channels**: Progressive channel expansion (32→64→128)
3. **Dropout Regularization**: 0.5 dropout before final classification
4. **Larger Batch Size**: Increased from 4 to 32 for better gradient estimates
5. **Extended Training**: 30 epochs vs. previous 20 epochs
6. **Proper Padding**: Maintains spatial dimensions through convolutions

## 🚀 Technical Implementation

### Data Preprocessing:
- **Training Augmentation**: RandomHorizontalFlip, RandomCrop with padding
- **Normalization**: CIFAR-10 specific mean/std normalization
- **Batch Size**: 32 for efficient GPU utilization

### Training Configuration:
- **Optimizer**: SGD with momentum=0.9, lr=0.001
- **Loss Function**: CrossEntropyLoss
- **Device**: CUDA GPU acceleration
- **Epochs**: 30 with loss monitoring

### Model Features:
- **GPU Acceleration**: Full CUDA support for training and inference
- **Model Persistence**: Saved weights for reproducibility
- **Performance Analysis**: Per-class accuracy breakdown

## 📈 Performance Analysis

### Overall Results:
- **Previous Baseline**: ~54% accuracy
- **Current Model**: **78.51% accuracy**
- **Improvement**: **+24.51 percentage points**

### Training Characteristics:
- **Stable Convergence**: Consistent loss reduction across epochs
- **No Overfitting**: Proper regularization with BatchNorm + Dropout
- **Efficient Training**: ~30 epochs for optimal performance

## 🎯 CIFAR-10 Classes:
1. Airplane ✈️
2. Automobile 🚗  
3. Bird 🐦
4. Cat 🐱
5. Deer 🦌
6. Dog 🐕
7. Frog 🐸
8. Horse 🐎
9. Ship 🚢
10. Truck 🚛

## 💻 Requirements

```bash
torch>=1.9.0
torchvision>=0.10.0
matplotlib>=3.3.0
numpy>=1.21.0
```

## 🚀 Usage

### Training the Model:
```bash
# Navigate to project directory
cd "Deep Learning/CIFAR10"

# Run the complete pipeline
jupyter notebook CIFAR10_neural_network.ipynb
```

### Key Notebook Sections:
1. **Data Loading**: Automatic CIFAR-10 download and preprocessing
2. **Architecture Definition**: Enhanced CNN with BatchNorm + Dropout
3. **Training Loop**: 30 epochs with GPU acceleration
4. **Model Evaluation**: Comprehensive testing and per-class analysis
5. **Model Persistence**: Save/load trained weights

## 📁 Project Structure

```
CIFAR10/
├── README.md                        # This comprehensive documentation
├── CIFAR10_neural_network.ipynb     # Main implementation notebook
├── cifar_net.pth                   # Trained model weights (78.51% accuracy)
└── data/                           # CIFAR-10 dataset (auto-downloaded)
    └── cifar-10-batches-py/        # Extracted dataset files
```

## 🎓 Learning Outcomes

This project demonstrates:

### Technical Skills:
- **CNN Architecture Design**: Modern layer composition with normalization
- **PyTorch Proficiency**: End-to-end deep learning pipeline
- **GPU Programming**: CUDA acceleration for training
- **Model Optimization**: Hyperparameter tuning and architecture improvement

### Deep Learning Concepts:
- **Batch Normalization**: Stabilizing training dynamics
- **Dropout Regularization**: Preventing overfitting
- **Data Augmentation**: Improving generalization
- **Transfer of Knowledge**: Scaling from simple to complex architectures

### Best Practices:
- **Reproducible Research**: Model saving and comprehensive documentation
- **Performance Monitoring**: Training loss tracking and evaluation metrics
- **Code Organization**: Clean, well-documented implementation

## 🚀 Future Enhancements

### Potential Improvements:
- **ResNet Architecture**: Residual connections for deeper networks
- **Learning Rate Scheduling**: Adaptive learning rate strategies  
- **Advanced Augmentation**: Mixup, CutMix, or AutoAugment
- **Ensemble Methods**: Combining multiple models
- **Transfer Learning**: Pre-trained model fine-tuning

### Target Performance:
- **80-85%**: With ResNet-style architectures
- **85-90%**: With advanced augmentation techniques
- **90%+**: With ensemble methods and modern architectures

## 📊 Comparison with Other Approaches

| Method | Accuracy | Notes |
|--------|----------|--------|
| **Basic CNN** | ~54% | Original simple architecture |
| **Enhanced CNN** | **78.51%** | **Current implementation** |
| ResNet-18 | ~85% | Residual connections |
| Transfer Learning | ~90%+ | Pre-trained models |

## 🙏 Acknowledgments

- **CIFAR-10 Dataset**: Canadian Institute for Advanced Research
- **PyTorch Framework**: Meta AI Research
- **Educational Context**: Advanced Machine Learning course

---

**Built with 🧠 intelligence and 🔬 scientific rigor**

**Author**: Dorotea Monaco | **Institution**: Politecnico di Torino  
**Achievement**: 78.51% accuracy on CIFAR-10 classification
