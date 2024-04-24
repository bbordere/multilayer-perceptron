import numpy as np
from activation_functions import get_activation


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


class Layer:
    def __init__(
        self,
        input_size: int,
        output_size: int,
        activation: str = "sigmoid",
        weights_init: str = "heUniform",
    ) -> None:
        self.input_size = input_size
        self.output_size = output_size
        self.w = init_weights(output_size, input_size, weights_init)
        self.b = init_weights(output_size, 1, "zeros")
        self.activation_name = activation
        self.act_func, self.act_func_prime = get_activation(self.activation_name)

    def __str__(self) -> str:
        return f"Layer: shape={self.w.shape[1], self.w.shape[0]}, activation={self.activation_name}"

    def __repr__(self) -> str:
        return str(self)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        z = np.dot(self.w, x) + self.b
        self.output = self.act_func(z)
        return self.output

    def backward(self, gradient: np.ndarray) -> np.ndarray:
        pass


if __name__ == "__main__":
    l = Layer(10, 10)
    data = np.linspace(-100, 100, 10)
    print(l.forward(data))
    # print(l.backward())
