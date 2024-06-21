import numpy as np


class Optimizer:
    def __init__(self, lr: float = 0.01) -> None:
        self.lr = lr
        self.name = "SGD"

    def optimize(self, weights, gradients):
        weights -= self.lr * gradients

    def __str__(self) -> str:
        return self.name


class SGDMOptimizer(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9):
        super().__init__(lr)
        self.beta = beta
        self.velocities = {}
        self.name = "SGDM"

    def optimize(self, weights, gradients):
        weights_id = id(weights)
        if weights_id not in self.velocities:
            self.velocities[weights_id] = np.zeros_like(weights)

        v = self.beta * self.velocities[weights_id] + (1 - self.beta) * gradients
        weights -= self.lr * v
        self.velocities[weights_id] = v


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        lr: float = 0.001,
        beta1: float = 0.9,
        beta2: float = 0.999,
        epsilon: float = 1e-8,
    ):
        super().__init__(lr)
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.m = {}
        self.v = {}
        self.t = 0
        self.name = "ADAM"

    def optimize(self, weights, gradients):
        weights_id = id(weights)
        if weights_id not in self.m:
            self.m[weights_id] = np.zeros_like(weights)
        if weights_id not in self.v:
            self.v[weights_id] = np.zeros_like(weights)

        self.t += 1

        self.m[weights_id] = (
            self.beta1 * self.m[weights_id] + (1 - self.beta1) * gradients
        )
        self.v[weights_id] = self.beta2 * self.v[weights_id] + (1 - self.beta2) * (
            gradients**2
        )

        m_hat = self.m[weights_id] / (1 - self.beta1**self.t)
        v_hat = self.v[weights_id] / (1 - self.beta2**self.t)

        weights -= self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon)


class RMSPropOptimizer(Optimizer):
    def __init__(self, lr: float = 0.001, beta: float = 0.9, epsilon: float = 1e-8):
        super().__init__(lr)
        self.beta = beta
        self.epsilon = epsilon
        self.s = {}
        self.name = "RMSProp"

    def optimize(self, weights, gradients):
        weights_id = id(weights)
        if weights_id not in self.s:
            self.s[weights_id] = np.zeros_like(weights)

        self.s[weights_id] = self.beta * self.s[weights_id] + (1 - self.beta) * (
            gradients**2
        )

        weights -= self.lr * gradients / (np.sqrt(self.s[weights_id]) + self.epsilon)
