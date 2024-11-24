# Multilayer Perceptron
Implementing a multilayer perceptron from scratch with numpy as an introduction to artificial neural networks
## Overview
This project aims to build a modular multilayer perceptron from scratch to learn basics of neural networks with only [Numpy](https://numpy.org/) library. Using a dataset that describes the characteristics of a cell nucleus of breast
mass extracted with fine-needle aspiration, the model is able to learn and predict whether a cell is tumoral or not with an accuracy higher than 90%

## Data Visualization
Before designing a good model, we need to understand the dataset by visualizing it using graphs. This is accomplished using the **visualize.py** program.

![pairplot](https://github.com/user-attachments/assets/72c627d3-8ef3-4978-a2ed-6d254f06eb77)
</br>
(This pairplot does not represent the entire dataset because of its size)

## Training Part
We need to divide the dataset into two parts before training: one for training and one for validation, in order to assess the model's performance in a robust way. 
The training curves are displayed after training to visualize the model's ability to make correct predictions.
</br>
![Learning_curves](https://github.com/user-attachments/assets/7f1639d8-ac58-4321-af41-16a82784a994)

### Features Available
To make the implementation modular and robust, there are several things that can be useful:
- Layer Type:
    - Dense Layer
    - Activation Layer
    - Softmax Layer
    - Dropout Layer
- Activations Functions:
    - Sigmoid
    - ReLU (Rectified Linear Unit)
    - Tanh
    - ELU (Exponential Linear Unit)
- Optimizers:
    - SGD (Stochastic Gradient Descent)
    - SGDM (Stochastic Gradient Descent with Momentum)
    - ADAM (Adaptive Moment Estimation)
    - RMSProp (Root Mean Squared Propagation)

## Predictions Part
After training a model, we can use it to make predictions on unseen datas. To see the model's performance, we refer to the confusion matrix
</br>
![Confusion](https://github.com/user-attachments/assets/03e3c4ed-e27d-4db9-bc8b-a518f6e0b19d)

## Benchmarking 
In order to compare different architectures, we can use the **benchmark.py** program and find the most suitable one. 

![Bench](https://github.com/user-attachments/assets/542947c5-d5df-491e-9ea1-55bdce0fca1d)
