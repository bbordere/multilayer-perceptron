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

    def fit(self, x: np.ndarray, y: np.ndarray, epochs=1000) -> None:
        predict = []
        with alive_bar(epochs) as bar:
            for epoch in range(epochs):
                predict = self.forward(x)
                self.backward(predict, y)
                bar()

    def predict(self, x: np.ndarray) -> np.array:
        raw_predict = self.forward(x)
        return np.argmax(raw_predict, axis=1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        predict = self.predict(x)
        print(BCE(y, predict))
        return sum(predict == y) / len(y)


def BCE(y_true, y_pred):

    target = y_true
    output = y_pred

    output = np.clip(output, 1e-7, 1.0 - 1e-7)
    output = -target * np.log(output) - (1.0 - target) * np.log(1.0 - output)
    return np.mean(output, axis=-1)


# y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)
# term_0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
# term_1 = y_true * np.log(y_pred + 1e-7)
# return -np.mean(term_0 + term_1, axis=0)


if __name__ == "__main__":

    x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[1, 0], [0, 1], [0, 1], [1, 0]])
    net = NeuralNetwork(
        [
            DenseLayer(2, 50),
            ActivationLayer("sigmoid"),
            DenseLayer(50, 20),
            ActivationLayer("sigmoid"),
            DenseLayer(20, 2),
            ActivationLayer("softmax"),
        ]
    )
    predict = []
    with alive_bar(200) as bar:
        for epoch in range(200):
            predict = net.forward(x)
            print(BCE(y, predict))
            net.backward(predict, y)
            bar()
    print(predict)
