from Layer import *
import numpy as np
from alive_progress import alive_bar


class NeuralNetwork:
    def __init__(self, layers: list[AbstractLayer]) -> None:
        self.layers = layers

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
        return out

    def backward(self, predict: np.ndarray, target: np.ndarray) -> None:
        grad = predict - target
        for l in reversed(self.layers):
            grad = l.backward(grad)


def BCE(y_true, y_pred):
    return np.mean(-y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred))


if __name__ == "__main__":

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    net = NeuralNetwork(
        [
            DenseLayer(2, 50),
            ActivationLayer("sigmoid"),
            DenseLayer(50, 20),
            ActivationLayer("relu"),
            DenseLayer(20, 2),
            ActivationLayer("softmax"),
        ]
    )
    predict = []
    with alive_bar(200) as bar:
        for epoch in range(200):
            predict = net.forward(x)
            # print(BCE(y, predict))
            net.backward(predict, y)
            bar()
    print(predict)
