import sklearn.metrics
from Layer import *
import numpy as np
from alive_progress import alive_bar
from utils import *
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

import sklearn


class NeuralNetwork:
    def __init__(self, layers: list[AbstractLayer]) -> None:
        self.layers = layers
        self.metrics = {
            "val_loss": [],
            "train_loss": [],
            "train_acc": [],
            "val_acc": [],
            "train_f1": [],
            "test_f1": [],
            # "test_precision": [],
        }

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

    def compute_metrics(self, x_train, y_train, x_test, y_test):
        train_pred = self.predict(x_train)
        test_pred = self.predict(x_test)
        self.metrics["train_loss"].append(BCE(y_train, train_pred).mean())
        self.metrics["val_loss"].append(BCE(y_test, test_pred).mean())
        self.metrics["train_acc"].append(sum(train_pred == y_train) / len(y_train))
        self.metrics["val_acc"].append(sum(test_pred == y_test) / len(y_test))
        self.metrics["train_f1"].append(sklearn.metrics.f1_score(y_train, train_pred))
        self.metrics["test_f1"].append(sklearn.metrics.f1_score(y_test, test_pred))
        # self.metrics["test_precision"].append(
        #     sklearn.metrics.precision_score(y_test, test_pred)
        # )

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
        self.metrics["epoch"] = range(0, epochs)
        x_train, y_train = train
        x_test, y_test = test
        y_train = one_hot(y_train, 2)
        with alive_bar(epochs) as bar:
            for _ in range(epochs):
                for batch in get_batches((x_train, y_train), batch_size):
                    X, Y = batch
                    predict = self.forward(X)
                    self.backward(predict, Y)
                bar()
                self.compute_metrics(train[0], train[1], x_test, y_test)
                print(
                    f"loss: {self.metrics['train_loss'][-1]} - val_loss: {self.metrics['val_loss'][-1]}",
                )

    def predict(self, x: np.ndarray) -> np.array:
        raw_predict = self.forward(x)
        return np.argmax(raw_predict, axis=1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        predict = self.predict(x)
        return sum(predict == y) / len(y)

    def plot_metrics(self):
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3)

        fig.suptitle("Learning Curves")
        data = pd.DataFrame.from_dict(self.metrics)
        ax1.set_yticks(np.arange(0, 50, 0.5))
        ax2.set_yticks(np.arange(0, 1.01, 0.02))
        ax1.grid(True)
        ax2.grid(True)
        ax3.grid(True)

        sns.lineplot(
            x="epoch", y="train_loss", data=data, ax=ax1, label="Training loss"
        )
        sns.lineplot(
            x="epoch", y="val_loss", data=data, ax=ax1, label="Validation loss"
        )
        sns.lineplot(
            x="epoch", y="train_acc", data=data, ax=ax2, label="Training accuracy"
        )
        sns.lineplot(
            x="epoch", y="val_acc", data=data, ax=ax2, label="Validation accuracy"
        )
        sns.lineplot(
            x="epoch", y="train_f1", data=data, ax=ax3, label="Training F1 Score"
        )
        sns.lineplot(
            x="epoch", y="test_f1", data=data, ax=ax3, label="Validation F1 Score"
        )
        # sns.lineplot(
        #     x="epoch",
        #     y="test_precision",
        #     data=data,
        #     ax=ax3,
        #     label="Validation F1 Score",
        # )

        print(self.metrics["test_f1"][-1])

        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
        ax2.set_xlabel("Epoch")
        ax2.set_ylabel("Accuracy")
        plt.show()


if __name__ == "__main__":
    pass
