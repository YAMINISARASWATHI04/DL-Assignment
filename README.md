# DL-Assignment
# Fashion MNIST Classification with Flexible Neural Networks

## Overview
This project trains a feedforward neural network (FNN) on the Fashion MNIST dataset using different hidden layer configurations and activation functions (ReLU and Sigmoid). The model is evaluated based on training and validation accuracy, and key observations are made about model performance.

## Requirements
Ensure you have the following dependencies installed:
```bash
pip install torch torchvision matplotlib numpy kagglehub
```

## Dataset
The dataset is automatically downloaded using KaggleHub:
- Training data: 60,000 images
- Test data: 10,000 images

## Model Training
The training script supports:
- Different hidden layer configurations ([128,64], [256,128,64], [512,256,128,64])
- Activation functions: ReLU and Sigmoid
- Adam optimizer with backpropagation
- Xavier weight initialization

### To train the model, run:
```bash
python train.py
```
The script will train the model with different configurations and track accuracy over epochs.

## Evaluation
The model is evaluated on both the validation and test datasets.
- Training and validation accuracy are plotted over epochs.
- The best configuration is selected based on validation accuracy.
- Test set evaluation is performed on the best model.

### To evaluate the model, run:
```bash
python evaluate.py
```

## Results & Inferences
Observations:
- Deeper networks generally perform better but may overfit.
- ReLU activation performs better than Sigmoid due to better gradient flow.
- Sigmoid activation struggles in deeper networks due to vanishing gradients.
- The optimal model balances depth and generalization ability.

## Coding Style & Clarity
- Functions are modularized for training and evaluation.
- Code follows best practices with clear documentation and comments.
- Training and evaluation logic is separated for better maintainability.
