# Neural Network Digits Classifier

This repository contains my implementation of a neural network for classifying handwritten digits, built as part of my learning journey through neural networks and deep learning.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Learning Resources](#learning-resources)
3. [Installation](#installation)
4. [Usage](#usage)
5. [Model Overview](#model-overview)
6. [Results](#results)

## Project Overview

The goal of this project is to build a simple neural network from scratch, capable of classifying handwritten digits (0-9) from the popular MNIST dataset. The project was inspired by a couple of excellent resources that guided me through the theoretical and practical aspects of neural networks.

### Features:
- Implementation of a basic feedforward neural network.
- Backpropagation for training.
- The network is trained to classify digits from the MNIST dataset.
- No external deep learning libraries (like TensorFlow or PyTorch) were used — the implementation was done from scratch using `NumPy`.

## Learning Resources

I used the following resources to guide me through understanding and building neural networks:

1. **Book**: [Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/) by Michael Nielsen
   - This book was instrumental in understanding the mathematics and concepts behind neural networks and their implementation from scratch.

2. **Video Series**: [3Blue1Brown's YouTube Playlist on Neural Networks](https://www.youtube.com/watch?v=aircAruvnKk&list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi&index=1)
   - This series visually explains neural networks and backpropagation in an intuitive way, helping me understand the logic and flow of a neural network.

## Installation

To run this project locally, you'll need Python and a few essential libraries:

### Prerequisites
- Python 3.x
- `numpy`
- `matplotlib` (optional, for visualizing results)

### Installation
1. Clone the repository:
    ```bash
    git clone https://github.com/wojciechtrapkowski/digits_classifier
    cd neural-network-digits-classifier
    ```

2. Install the required libraries:
    ```bash
    pip install numpy matplotlib
    ```

### Usage

1. Run program:
    ```python
    python3 main.py
    ```

## Model Overview

### Architecture:
- **Input Layer**: 784 neurons (28x28 pixel input images).
- **Hidden Layer**: Configurable, but typically a layer with 30 neurons.
- **Output Layer**: 10 neurons (one for each digit 0-9).

### Training:
- The network is trained using **stochastic gradient descent** with **backpropagation**.
- Activation function used: **Sigmoid** for neurons.

## Results

After training for several epochs, the model achieves approximately 95% accuracy on the MNIST test set.