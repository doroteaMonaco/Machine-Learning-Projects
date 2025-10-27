# CIFAR-10 Neural Network Classification

This repository contains an Advanced Machine Learning homework assignment implementing a Convolutional Neural Network (CNN) for image classification on the CIFAR-10 dataset using PyTorch. The project covers:

- Data loading and preprocessing of the CIFAR-10 dataset
- Definition of a simple CNN architecture
- Model training with stochastic gradient descent
- Testing and evaluation of the trained model
- Performance analysis across different classes

The trained model achieves approximately 54% accuracy on the test set.

## Requirements

- Python 3.x
- PyTorch
- Torchvision
- Matplotlib
- NumPy

## Usage

1. Clone the repository.
2. Run the Jupyter notebook `CIFAR10_neural_network.ipynb` to train and test the model.
3. The dataset will be automatically downloaded if not present.

## Files

- `CIFAR10_neural_network.ipynb`: Main notebook with the implementation
- `cifar_net.pth`: Trained model weights (generated after running the notebook)
- `data/`: Directory for CIFAR-10 dataset (ignored by git, downloaded automatically)
