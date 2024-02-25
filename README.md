# Deep Neural Network for Binary Classification

This repository contains a Python implementation of a deep neural network for binary classification tasks. The neural network is capable of learning complex patterns in data and making predictions on unseen examples.

## Features
- Flexible architecture: Allows customization of the number of layers and nodes in each layer.
- Supports both sigmoid and hyperbolic tangent (tanh) activation functions for hidden layers.
- Utilizes He initialization for weight initialization and zero initialization for bias.
- Implements forward propagation, backward propagation, and gradient descent for training.
- Computes the categorical cross-entropy cost function for evaluation.
- Provides methods for training the model, evaluating performance, and saving/loading trained models.

## Usage
1. Instantiate the `DeepNeuralNetwork` class with the desired architecture and activation function.
2. Train the model using the `train` method by providing input features and corresponding labels.
3. Evaluate the trained model's performance using the `evaluate` method.
4. Save the trained model to a file using the `save` method for future use.

## Dependencies
- `numpy`: For numerical operations and array manipulations.
- `matplotlib`: For plotting training costs and visualizations.

## Example
```python
from deep_neural_network import DeepNeuralNetwork

# Instantiate the model with desired architecture and activation function
model = DeepNeuralNetwork(nx=10, layers=[16, 8, 1], activation='sig')

# Train the model
model.train(X_train, Y_train, iterations=5000, alpha=0.05, verbose=True)

# Evaluate the model
predictions, cost = model.evaluate(X_test, Y_test)
print("Test Cost:", cost)

# Save the trained model
model.save("trained_model.pkl")

