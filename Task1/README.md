# Task 1. Image classification + OOP

This project implements a image classification system for the MNIST dataset using three different algorithmic approaches: Random Forest, Feed-Forward Neural Network, and Convolutional Neural Network (CNN).

The solution is built using Object-Oriented Programming principles:

1. MnistClassifierInterface - An abstract base class that defines the contract for all classification models. It contains two mandatory methods:
   * train(train_loader): Handles the model training logic.
   * predict(image): Returns the predicted class for a given input.

2. Concrete Implementations

    * RandomForest Classifier (rf).
    * Feed-Forward Neural Network (nn) - a multi-layer perceptron (MLP) built with PyTorch.
    * Convolutional Neural Network (cnn) - a PyTorch implementation using convolutional layers (Conv2d).

3. The Wrapper Class: MnistClassifier. This is the main entry point for the user. It acts as a Strategy Pattern wrapper. It takes an algorithm parameter ("rf", "nn", or "cnn") and internally initializes the corresponding class, hiding the complexity of the underlying model from the user.

## How to use
```python
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from MnistClassifier import MnistClassifier

# Import data
mnist = fetch_openml('mnist_784')
X, y = mnist["data"], mnist["target"]
y = y.astype(int)
# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

# 1. Initialize with a specific algorithm ('cnn', 'rf', or 'nn')
clf = MnistClassifier(algorithm='cnn')

# 2. Train the model
clf.train(X_train, y_train)

# 3. Get predictions
prediction = clf.predict(sample_image)
print(f"Predicted Digit: {prediction}")
```
