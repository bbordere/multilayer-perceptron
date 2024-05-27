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


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e_x / np.sum(e_x, axis=-1, keepdims=True)
    # return np.exp(x) / np.sum(np.exp(x), axis=0)


def d_softmax(x: np.ndarray) -> np.ndarray:
    # z = softmax(x)
    # return z * (1 - z)
    return x


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
        case "softmax":
            return (softmax, d_softmax)
        case "relu":
            return (relu, d_relu)
        case "tanh":
            return (tanh, d_tanh)
        case "elu":
            return (elu, d_elu)
        case _:
            return (sigmoid, d_sigmoid)
