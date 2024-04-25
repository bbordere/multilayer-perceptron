import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-x))


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
    z = softmax(x)
    return z * (1 - z)


def get_activation(name: str) -> tuple:
    match name:
        case "sigmoid":
            return (sigmoid, d_sigmoid)
        case "softmax":
            return (softmax, d_softmax)
        case "relu":
            return (relu, d_relu)
        case _:
            return (sigmoid, d_sigmoid)
