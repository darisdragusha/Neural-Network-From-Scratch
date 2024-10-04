# Neural Network from Scratch

## Description

This project implements a custom neural network from scratch to classify images from the Fashion-MNIST dataset. The neural network is built using only NumPy  for mathematical operations and includes features such as adaptive learning rates, batch processing, and various activation functions.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Architecture](#architecture)
- [Performance](#performance)
- [Contributing](#contributing)
- [License](#license)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/Fashion-MNIST-Neural-Network.git
   cd Fashion-MNIST-Neural-Network
   ```

2. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

## Usage

To train the model and evaluate its performance:

1. Run the training script:
   ```
   python train.py
   ```

2. The script will train the model and save it as "trained_model.npz".

3. After training, the script will automatically evaluate the model on the test set and print the results.

## Features

- Custom implementation of a neural network using NumPy
- Xavier initialization for weights
- Leaky ReLU activation for hidden layers
- Softmax activation for the output layer
- Cross-entropy loss function
- Adaptive learning rate
- Mini-batch gradient descent
- Model saving and loading functionality

## Architecture

The neural network consists of:
- Input layer: 784 neurons (28x28 flattened images)
- Hidden layer: 64 neurons with Leaky ReLU activation
- Output layer: 10 neurons with Softmax activation

## Performance

The model's performance on the Fashion-MNIST test set will be displayed after training. Typical results might include:

Test Loss: X.XXXX, Test Accuracy: XX.XX%