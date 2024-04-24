from Layer import Layer
import numpy as np


class NeuralNetwork:
    def __init__(self, layers: list[Layer]) -> None:
        self.layers = layers
        for i in range(1, len(self.layers)):
            assert (
                self.layers[i].input_size == self.layers[i - 1].output_size
            ), f"incompatible shape betweem layers {i - 1} and {i}"

    def __str__(self) -> str:
        res = "Network:\n"
        for l in self.layers:
            res += "\t" + str(l) + "\n"
        return res

    def __repr__(self) -> str:
        return str(self)

    def forward(self, x: np.ndarray) -> np.ndarray:
        out = x
        for l in self.layers:
            out = l.forward(out)
        return out.T

    def backward(self, predict: np.ndarray, target: np.ndarray) -> None:
        delta = predict - target
        for l in reversed(self.layers):
            delta = l.backward(delta)


if __name__ == "__main__":
    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]]).T
    y = np.array([[0], [1], [1], [0]])
    net = NeuralNetwork([Layer(2, 50), Layer(50, 20), Layer(20, 2)])
    # for _ in range(1000000):
    predict = net.forward(x)
    print(predict)
    # net.backward(predict, y)
    # net.update()
    # print(predict)
