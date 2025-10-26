# Predictors Projects Collection 🤖

A comprehensive collection of machine learning and deep learning projects focused on predictive modeling across different domains: computer vision, medical diagnosis, and multi-class classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Projects](#projects)
  - [CIFAR-10 Neural Network](#cifar-10-neural-network-classification)
  - [Diabetes Predictor](#diabetes-predictor)
  - [Flower Classification](#flower-classification-with-cnn)
- [Technologies Used](#technologies-used)
- [Getting Started](#getting-started)
- [Project Comparison](#project-comparison)
- [Installation](#installation)
- [Usage](#usage)
- [Results Summary](#results-summary)
- [Repository Structure](#repository-structure)
- [Contributing](#contributing)

## 🌟 Overview

This repository contains three distinct machine learning projects that demonstrate various approaches to predictive modeling:

1. **Computer Vision**: Image classification using Convolutional Neural Networks
2. **Medical Prediction**: Healthcare analytics with traditional ML algorithms
3. **Multi-class Classification**: Large-scale flower species recognition

Each project showcases different aspects of machine learning pipeline development, from data preprocessing to model evaluation, covering both traditional ML and deep learning approaches.

## 🚀 Projects

### 🖼️ CIFAR-10 Neural Network Classification

**Domain**: Computer Vision | **Type**: Multi-class Classification | **Framework**: PyTorch

A Convolutional Neural Network implementation for classifying images from the CIFAR-10 dataset into 10 distinct categories (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck).

#### Key Features:
- **Dataset**: 60,000 32×32 color images in 10 classes
- **Architecture**: Custom CNN with multiple convolutional layers
- **Performance**: ~54% accuracy on test set
- **Training**: Stochastic Gradient Descent optimization
- **Output**: Saved model weights (`cifar_net.pth`)

#### Technical Highlights:
- Automatic dataset downloading and preprocessing
- GPU acceleration support
- Real-time training loss monitoring
- Comprehensive performance analysis across classes

**📁 Location**: `./CIFAR10/`  
**📓 Main File**: `CIFAR10_neural_network.ipynb`

---

### 🏥 Diabetes Predictor

**Domain**: Healthcare Analytics | **Type**: Binary Classification | **Framework**: Scikit-learn + XGBoost

A comprehensive machine learning pipeline for predicting diabetes risk using medical and demographic data from the Pima Indians Diabetes Database.

#### Key Features:
- **Dataset**: 768 patient records with 8 medical features
- **Best Model**: XGBoost with 76% recall, 72% F1-score
- **Pipeline**: End-to-end ML workflow with preprocessing
- **Clinical Focus**: High recall to minimize missed diagnoses
- **Class Imbalance**: Handled with SMOTE and scale_pos_weight

#### Technical Highlights:
- **Advanced Preprocessing**: Missing value imputation, robust scaling
- **Feature Engineering**: Glucose/Insulin ratio creation
- **Model Comparison**: Logistic Regression variants vs. XGBoost
- **Medical Validation**: Clinically relevant evaluation metrics
- **Comprehensive EDA**: Detailed data exploration and visualization

#### Models Trained:
| Model | Recall | F1-Score | AUC | Notes |
|-------|--------|----------|-----|-------|
| Logistic Regression | 0.55 | 0.65 | 0.82 | Baseline |
| Polynomial Features | 0.60 | 0.68 | 0.83 | Non-linear |
| **XGBoost** | **0.76** | **0.72** | **0.85** | **Best** |

**📁 Location**: `./DiabetPredictor/`  
**📓 Main File**: `DiabetPredictor.ipynb`

---

### 🌸 Flower Classification with CNN

**Domain**: Computer Vision | **Type**: Multi-class Classification (102 classes) | **Framework**: PyTorch

A deep learning project for classifying flower images into 102 different species using the Oxford 102 Category Flower Dataset.

#### Key Features:
- **Dataset**: 8,189 high-resolution flower images
- **Classes**: 102 different flower species
- **Architecture**: 3-layer CNN with ~51.5M parameters
- **Input**: 224×224×3 RGB images
- **Preprocessing**: Resizing, normalization, tensor conversion

#### Technical Highlights:
- **Custom Dataset Class**: Handles MATLAB label files
- **Large-scale Classification**: 102-way classification problem
- **GPU Optimization**: CUDA acceleration for training
- **Data Loading**: Efficient batch processing
- **Pre-defined Splits**: Uses official train/validation/test splits

#### Model Architecture:
```
Input (224×224×3) → Conv+ReLU+MaxPool → Conv+ReLU+MaxPool → 
Conv+ReLU+MaxPool → Flatten → FC(512) → FC(102 classes)
```

**📁 Location**: `./Flower Classification/`  
**📓 Main File**: `Flowers.ipynb`

## 🛠️ Technologies Used

### Machine Learning Frameworks:
- **PyTorch**: Deep learning models (CIFAR-10, Flower Classification)
- **Scikit-learn**: Traditional ML algorithms (Diabetes Predictor)
- **XGBoost**: Gradient boosting for tabular data

### Data Processing:
- **Pandas**: Data manipulation and analysis
- **NumPy**: Numerical computing
- **SciPy**: Scientific computing (MATLAB file handling)
- **imbalanced-learn**: Class imbalance handling (SMOTE)

### Visualization:
- **Matplotlib**: Plotting and visualization
- **Seaborn**: Statistical data visualization

### Development Environment:
- **Jupyter Notebook**: Interactive development
- **Python 3.8+**: Core programming language
- **CUDA**: GPU acceleration (optional)

## 🚀 Getting Started

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Jupyter Notebook
- GPU with CUDA support (recommended for deep learning projects)

### Quick Start
1. **Clone the repository**
   ```bash
   git clone https://github.com/doroteaMonaco/Predictors-Projects.git
   cd Predictors-Projects
   ```

2. **Choose a project and navigate to its directory**
   ```bash
   cd CIFAR10                    # For CIFAR-10 classification
   cd DiabetPredictor           # For diabetes prediction
   cd "Flower Classification"   # For flower classification
   ```

3. **Open the Jupyter notebook**
   ```bash
   jupyter notebook
   ```

## 📊 Project Comparison

| Aspect | CIFAR-10 | Diabetes Predictor | Flower Classification |
|--------|----------|-------------------|----------------------|
| **Domain** | Computer Vision | Healthcare | Computer Vision |
| **Data Type** | Images (32×32) | Tabular | Images (224×224) |
| **Classes** | 10 | 2 (Binary) | 102 |
| **Samples** | 60,000 | 768 | 8,189 |
| **Algorithm** | CNN | XGBoost | CNN |
| **Framework** | PyTorch | Scikit-learn | PyTorch |
| **Accuracy** | ~54% | 76% Recall | In Progress |
| **Focus** | Multi-class | Medical/Recall | Large-scale |
| **Complexity** | Medium | High Pipeline | High Architecture |

## 💻 Installation

### Core Dependencies
```bash
# Essential packages for all projects
pip install jupyter pandas numpy matplotlib seaborn

# For PyTorch projects (CIFAR-10, Flower Classification)
pip install torch torchvision torchaudio

# For traditional ML (Diabetes Predictor)
pip install scikit-learn xgboost imbalanced-learn

# For flower classification MATLAB files
pip install scipy

# For visualization
pip install pillow
```

### GPU Support (Optional but Recommended)
```bash
# For CUDA-enabled PyTorch (check CUDA version first)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### Verify Installation
```python
import torch
import sklearn
import xgboost
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("Scikit-learn version:", sklearn.__version__)
```

## 📈 Usage

### Running Individual Projects

#### CIFAR-10 Classification:
```bash
cd CIFAR10
jupyter notebook CIFAR10_neural_network.ipynb
# Dataset downloads automatically
```

#### Diabetes Predictor:
```bash
cd DiabetPredictor
jupyter notebook DiabetPredictor.ipynb
# Uses included diabetes.csv dataset
```

#### Flower Classification:
```bash
cd "Flower Classification"
jupyter notebook Flowers.ipynb
# Requires manual dataset setup (see project README)
```

### Training Tips:
- **GPU Recommended**: Deep learning projects benefit significantly from GPU acceleration
- **Memory Requirements**: Ensure sufficient RAM (8GB+) and GPU memory (2GB+)
- **Training Time**: Varies from minutes (diabetes) to hours (deep learning)

## 📊 Results Summary

### Performance Overview:

| Project | Metric | Value | Significance |
|---------|--------|-------|--------------|
| **CIFAR-10** | Test Accuracy | 54% | Good for simple CNN |
| **Diabetes** | Recall | 76% | High medical relevance |
| **Diabetes** | F1-Score | 72% | Balanced performance |
| **Flower** | Architecture | 51.5M params | Large-scale classification |

### Key Achievements:
- **CIFAR-10**: Successful CNN implementation with automatic data handling
- **Diabetes**: Clinically relevant model with 76% recall (38% improvement over baseline)
- **Flower**: Complex 102-class classification with sophisticated preprocessing

## 📁 Repository Structure

```
Predictors-Projects/
├── README.md                           # This comprehensive overview
├── .gitignore                         # Git ignore patterns
│
├── CIFAR10/                           # Computer Vision - 10 classes
│   ├── README.md                      # Project-specific documentation
│   ├── CIFAR10_neural_network.ipynb   # Main implementation
│   ├── cifar_net.pth                 # Saved model weights
│   └── data/                          # CIFAR-10 dataset (auto-downloaded)
│       └── cifar-10-batches-py/       # Extracted dataset files
│
├── DiabetPredictor/                   # Healthcare Analytics
│   ├── README.md                      # Detailed project documentation
│   ├── DiabetPredictor.ipynb          # Complete ML pipeline
│   └── data_lab9/                     # Dataset directory
│       └── diabetes.csv               # Pima Indians Diabetes Database
│
└── Flower Classification/             # Computer Vision - 102 classes
    ├── README.md                      # Project documentation
    ├── Flowers.ipynb                  # Deep learning implementation
    ├── imagelabels.mat               # Flower category labels
    ├── setid.mat                     # Train/test/validation splits
    └── 102flowers/                   # Flower images dataset
        └── jpg/                      # 8,189 flower images
```

## 🎯 Learning Outcomes

This collection demonstrates:

### Technical Skills:
- **Deep Learning**: CNN architecture design and training
- **Traditional ML**: Feature engineering and model selection
- **Data Preprocessing**: Handling missing values, scaling, imbalance
- **Model Evaluation**: Appropriate metrics for different domains
- **Framework Proficiency**: PyTorch and Scikit-learn expertise

### Domain Knowledge:
- **Computer Vision**: Image classification challenges and solutions
- **Healthcare Analytics**: Medical data characteristics and evaluation priorities
- **Multi-class Problems**: Scaling to large number of categories

### Best Practices:
- **Reproducible Research**: Comprehensive documentation and code organization
- **Evaluation Focus**: Domain-appropriate metrics (accuracy vs. recall)
- **Pipeline Development**: End-to-end ML workflow implementation

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

### Enhancement Opportunities:
- **Transfer Learning**: Pre-trained models for image classification
- **Hyperparameter Tuning**: Systematic optimization
- **Cross-Validation**: Robust evaluation strategies
- **Ensemble Methods**: Combining multiple models
- **Deployment**: Model serving and API development

### How to Contribute:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 🙏 Acknowledgments

### Datasets:
- **CIFAR-10**: Canadian Institute for Advanced Research
- **Pima Indians Diabetes**: UCI Machine Learning Repository
- **Oxford 102 Flowers**: Visual Geometry Group, University of Oxford

### Frameworks:
- **PyTorch Team**: Deep learning framework
- **Scikit-learn Contributors**: Machine learning library
- **XGBoost Developers**: Gradient boosting framework

### Educational Support:
- **Politecnico di Torino**: Academic context and guidance
- **Open Source Community**: Libraries and tools that made this possible

## 📞 Contact & Links

**Author**: Dorotea Monaco  
**Institution**: Politecnico di Torino  
**GitHub**: [@doroteaMonaco](https://github.com/doroteaMonaco)  
**Repository**: [Predictors-Projects](https://github.com/doroteaMonaco/Predictors-Projects)

### Project-Specific Links:
- [CIFAR-10 Implementation](./CIFAR10/)
- [Diabetes Predictor](./DiabetPredictor/)
- [Flower Classification](./Flower%20Classification/)

---

*Built with 🧠 intelligence, ❤️ passion, and 🔬 scientific rigor*

**Last Updated**: October 2025