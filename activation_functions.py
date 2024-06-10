import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    # return 1 / (1 + np.exp(-x))
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))


def d_sigmoid(x: np.ndarray) -> np.ndarray:
    z = sigmoid(x)
    return z * (1 - z)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def d_relu(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(x.dtype)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def d_tanh(x: np.ndarray) -> np.ndarray:
    return 1 - (tanh(x) ** 2)


def elu(x: np.ndarray) -> np.ndarray:
    # return x if (x > 0).astype(x.dtype) else np.exp(x) - 1
    return np.where(x > 0, x, np.exp(x) - 1)


def d_elu(x: np.ndarray) -> np.ndarray:
    # return 1 if (x > 0).astype(x.dtype) else np.exp(x)
    return np.where(x > 0, 1, np.exp(x))


def get_activation(name: str) -> tuple:
    match name:
        case "sigmoid":
            return (sigmoid, d_sigmoid)
        case "relu":
            return (relu, d_relu)
        case "tanh":
            return (tanh, d_tanh)
        case "elu":
            return (elu, d_elu)
        case _:
            return (sigmoid, d_sigmoid)
