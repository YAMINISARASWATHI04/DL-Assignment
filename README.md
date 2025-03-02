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
## Conclusion
Based on extensive experimentation on the Fashion MNIST dataset, I identified the best-performing hyperparameter configurations. These configurations demonstrated strong generalization and are recommended for use on new datasets:
## 1
Hidden Layers: [128, 64] | Activation: ReLU | Optimizer: RMSprop
Test Accuracy: 87.00%
RMSprop adapts learning rates effectively, making it suitable for different datasets.4
## 2
Hidden Layers: [128, 64, 32] | Activation: ReLU | Optimizer: Nesterov Momentum
Test Accuracy: 87.11%
Nesterov Momentum enhances stability and improves convergence.
## 3
Hidden Layers: [64, 32] | Activation: ReLU | Optimizer: Adam
Test Accuracy: 86.85%
Adam provides a balance between speed and accuracy with efficient gradient updates.
## Key Learnings
ReLU Activation significantly outperforms Sigmoid due to the vanishing gradient problem.
Optimizers Matter: RMSprop, Adam, and Nesterov Momentum showed superior performance.
Network Depth: Models with 2-3 hidden layers performed best, while deeper architectures provided diminishing returns.
Batch Size & Learning Rate: A batch size of 32 with lr = 0.001 worked well across all configurations.
These findings can guide model selection for similar classification tasks, improving both performance and efficiency.


