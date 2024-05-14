from Layer import *
import numpy as np
from alive_progress import alive_bar
from utils import *


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
            grad = l.backward(grad, self.lr)

    def fit(
        self,
        train: np.ndarray,
        test: np.ndarray,
        epochs: int = 1000,
        lr: float = 0.1,
        batch_size=64,
    ) -> None:
        predict = []
        self.lr = lr
        x, y = train
        y = one_hot(y, 2)
        with alive_bar(epochs) as bar:
            for _ in range(epochs):
                for batch in get_batches((x, y), batch_size):
                    X, Y = batch
                    predict = self.forward(X)
                    print(BCE(Y, predict).mean())
                    self.backward(predict, Y)
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
