from NeuralNetwork import NeuralNetwork
from Layer import DenseLayer, ActivationLayer, SoftmaxLayer
import numpy as np

np.random.seed(4242)

FULL_RELU = NeuralNetwork(
    [
        DenseLayer(30, 30),
        ActivationLayer("relu"),
        DenseLayer(30, 20),
        ActivationLayer("relu"),
        DenseLayer(20, 10),
        ActivationLayer("relu"),
        DenseLayer(10, 2),
        SoftmaxLayer(),
    ]
)


SUBJECT = NeuralNetwork(
    [
        DenseLayer(30, 24),
        ActivationLayer("sigmoid"),
        DenseLayer(24, 24),
        ActivationLayer("sigmoid"),
        DenseLayer(24, 24),
        ActivationLayer("sigmoid"),
        DenseLayer(24, 2),
        SoftmaxLayer(),
    ]
)


MIX = NeuralNetwork(
    [
        DenseLayer(30, 10),
        ActivationLayer("elu"),
        DenseLayer(10, 6),
        ActivationLayer("tanh"),
        DenseLayer(6, 6),
        ActivationLayer("relu"),
        DenseLayer(6, 10, "xavier"),
        ActivationLayer("tanh"),
        DenseLayer(10, 2, "xavier"),
        SoftmaxLayer(),
    ]
)


RELU_SIG_TANH = NeuralNetwork(
    [
        DenseLayer(30, 30),
        ActivationLayer("relu"),
        DenseLayer(30, 20),
        ActivationLayer("sigmoid"),
        DenseLayer(20, 10),
        ActivationLayer("tanh"),
        DenseLayer(10, 2),
        SoftmaxLayer(),
    ]
)
