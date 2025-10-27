# Hyperparameter Optimization for MNIST Classification

## 📋 Project Description

This project implements and compares different approaches for MNIST dataset classification using PyTorch, focusing on hyperparameter optimization. It includes implementations of:

- **Multi-Layer Perceptron (MLP)** with 2 hidden layers
- **Convolutional Neural Network (CNN)** with BatchNorm and Dropout
- **Random Search** for hyperparameter optimization
- **Early Stopping** to prevent overfitting

## 🛠️ Technologies Used

- **PyTorch** - Deep learning framework
- **torchvision** - MNIST dataset and transformations
- **scikit-learn** - Train/validation split
- **matplotlib** - Results visualization
- **numpy** - Numerical operations

### Hardware Compatibility
- **CPU**: Full support
- **CUDA GPU**: Automatic acceleration if available (NVIDIA)

## 📁 Project Structure

```
LAB2/
├── Hyperparameter_Optimization.ipynb    # Main notebook
├── README.md                           # This file
└── drive/MyDrive/MNIST/               # MNIST dataset
    └── raw/
        ├── train-images-idx3-ubyte
        ├── train-labels-idx1-ubyte
        ├── t10k-images-idx3-ubyte
        └── t10k-labels-idx1-ubyte
```

## 🚀 How to Use

### 1. Dependencies Installation

```bash
pip install torch torchvision matplotlib scikit-learn numpy
```

**Optional - For Intel acceleration:**
```bash
pip install openvino
```

### 2. Execution

1. Open `Hyperparameter_Optimization.ipynb` in Jupyter or VS Code
2. Execute cells sequentially
3. The MNIST dataset will be downloaded automatically

## 🧠 Implemented Models

### 1. Multi-Layer Perceptron (MLP)
- **Architecture**: 784 → 256 → 128 → 10
- **Activation function**: ReLU
- **Optimizers tested**: SGD with different learning rates (1.0, 0.1, 0.01)

### 2. Convolutional Neural Network (CNN)
- **Conv1**: 1 → n1 channels (kernel 3x3, padding 1)
- **Conv2**: n1 → n2 channels (kernel 3x3, padding 1)
- **BatchNorm2d** after each convolution
- **Dropout** (0.25) for regularization
- **MaxPooling2d** (2x2)
- **FC Layer**: (n2×7×7) → 10

## 🎯 Hyperparameter Optimization

### Optimized Parameters
- **n1, n2**: Number of CNN channels [32, 64, 128, 256]
- **Learning Rate**: [0.1, 0.01, 0.001, 0.0001]
- **Weight Decay**: [0, 0.0001, 0.001, 0.01]
- **Batch Size**: [64, 128, 256, 512]
- **Optimizer**: ['SGD', 'Adam', 'RMSprop']
- **Momentum**: [0.95, 0.9, 0.8] (SGD/RMSprop only)

### Search Strategy
- **Random Search** with 10 iterations
- **Early Stopping** (patience=5) to prevent overfitting
- **Automatic saving** of best model per epoch

## 📊 Results

### Learning Rates Comparison (MLP)
- **LR = 1.0**: Gradient explosion 🔴
- **LR = 0.1**: Stable convergence ✅
- **LR = 0.01**: Overfitting detected 🟡

### CNN Performance
- **CNN vs MLP**: CNN significantly superior
- **Regularization**: BatchNorm + Dropout prevent overfitting
- **Convergence**: Faster and more stable than MLP

### Final Test Set Accuracy
| Model | Parameters | Test Accuracy |
|-------|------------|---------------|
| **Model 1** | n1=32, n2=128, lr=0.001, RMSprop | **99.08%** 🏆 |
| **Model 2** | n1=32, n2=128, lr=0.001, RMSprop | **98.79%** |
| **Model 3** | n1=32, n2=128, lr=0.001, RMSprop | **98.89%** |

> **🎯 Best Result**: Model 1 with **99.08%** accuracy - excellent performance for MNIST!

## 📈 Notebook Features

### 1. Data Preparation
- Automatic MNIST download
- Training/validation split (90%/10%)
- Normalization: μ=0.5, σ=0.5

### 2. Training Pipeline
- **Device detection**: Automatic CUDA/CPU
- **Progress monitoring**: Loss and accuracy per epoch
- **Visualization**: Real-time training/validation plots

### 3. Model Evaluation
- **Test accuracy** for all models
- **Performance comparison** between configurations
- **Model saving** of trained models (.pth)

## 🔧 Advanced Configuration

### Early Stopping
```python
patience = 5  # Epochs without improvement before stopping
```

### Device Selection
```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

### Hyperparameter Search
```python
Niteration = 10  # Number of configurations to test
EPOCHS = 10     # Epochs per configuration
```

## 📝 Technical Notes

### Preprocessing
- Conversion to PyTorch tensor
- Normalization: `(pixel - 0.5) / 0.5` → range [-1, 1]

### Regularization
- **BatchNorm**: Stabilizes training and accelerates convergence
- **Dropout**: Reduces overfitting during training
- **Weight Decay**: L2 regularization in optimizers

### Optimization
- **SGD**: Momentum 0.9 to accelerate convergence
- **Adam**: Automatic learning rate adaptation
- **RMSprop**: Good compromise for CNN

## 👥 Authors

Project developed for the **Machine Learning for Visual and Multimedia** course - Politecnico di Torino

## 📚 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)
- [Hyperparameter Optimization Techniques](https://arxiv.org/abs/1502.02127)