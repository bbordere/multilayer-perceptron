import numpy as np
from activation_functions import get_activation
from Optimizer import *


def init_weights(
    input_size: int, output_size: int, method: str = "heUniform"
) -> np.ndarray:
    match method:
        case "heUniform":
            limit = np.sqrt(6 / input_size)
            return np.random.uniform(-limit, limit, (input_size, output_size))
        case "xavier":
            limit = np.sqrt(6 / (input_size + output_size))
            return np.random.uniform(-limit, limit, (input_size, output_size))
        case "zeros":
            return np.zeros((input_size, output_size))
        case "ones":
            return np.ones((input_size, output_size))
        case "random":
            return np.random.randn(input_size, output_size)
        case _:
            return np.random.randn(input_size, output_size)


class AbstractLayer:
    def __init__(self, input_size: int, output_size: int) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.input = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def backward(self, out_grad: np.ndarray, lr: float) -> np.ndarray:
        raise NotImplementedError


class DenseLayer(AbstractLayer):
    def __init__(
        self, input_size: int, output_size: int, weights_init: str = "heUniform"
    ) -> None:
        super().__init__(input_size, output_size)
        self.w = init_weights(input_size, output_size, weights_init)
        self.b = np.zeros(output_size)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return np.dot(x, self.w) + self.b

    def backward(
        self, out_grad: np.ndarray, lr: float, optimizer: Optimizer
    ) -> np.ndarray:
        input_grad = np.dot(out_grad, self.w.T)

        weights_grad = np.dot(self.input.T, out_grad)
        bias_grad = 1 / len(out_grad) * np.sum(out_grad, axis=0)
        assert weights_grad.shape == self.w.shape
        assert bias_grad.shape == self.b.shape

        optimizer.optimize(self.w, weights_grad)
        optimizer.optimize(self.b, bias_grad)

        return input_grad


class ActivationLayer:
    def __init__(self, activation: str) -> None:
        self.act_func, self.act_func_prime = get_activation(activation)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return self.act_func(x)

    def backward(
        self, out_grad: np.ndarray, lr: float, optimizer: Optimizer
    ) -> np.ndarray:
        return out_grad * self.act_func_prime(self.input)


class SoftmaxLayer:
    def forward(self, x):
        e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return e_x / np.sum(e_x, axis=-1, keepdims=True)

    def backward(self, grad_output, lr: float, optimizer: Optimizer):
        return grad_output


class DropoutLayer:
    def __init__(self, dropout_rate):
        self.dropout_rate = dropout_rate
        self.mask = None

    def forward(self, inputs, training=True):
        if training:
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=inputs.shape)
            return inputs * self.mask / (1 - self.dropout_rate)
        return inputs

    def backward(self, out_grad: np.ndarray, lr: float, optimizer: Optimizer):
        return out_grad * self.mask / (1 - self.dropout_rate)
