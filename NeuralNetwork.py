import sklearn.metrics
from Layer import *
import numpy as np
from alive_progress import alive_bar
from utils import *
from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import copy
import joblib

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
            "val_f1": [],
            "train_recall": [],
            "val_recall": [],
            "train_precision": [],
            "val_precision": [],
            "train_precision": [],
            "val_precision": [],
            "train_auc": [],
            "val_auc": [],
        }
        self.best_loss = np.inf
        self.early_counter = 0
        self.optimizer = None

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
            grad = l.backward(grad, self.lr, self.optimizer)

    def compute_metrics(self, x_train, y_train, x_test, y_test):
        train_pred = self.predict(x_train)
        val_pred = self.predict(x_test)

        self.metrics["train_loss"].append(
            BCE(one_hot(y_train, 2), self.forward(x_train))
        )
        self.metrics["val_loss"].append(BCE(one_hot(y_test, 2), self.forward(x_test)))

        self.metrics["train_acc"].append(sum(train_pred == y_train) / len(y_train))
        self.metrics["val_acc"].append(sum(val_pred == y_test) / len(y_test))

        self.metrics["train_f1"].append(sklearn.metrics.f1_score(y_train, train_pred))
        self.metrics["val_f1"].append(sklearn.metrics.f1_score(y_test, val_pred))

        self.metrics["train_recall"].append(
            sklearn.metrics.recall_score(y_train, train_pred)
        )
        self.metrics["val_recall"].append(
            sklearn.metrics.recall_score(y_test, val_pred)
        )

        self.metrics["train_precision"].append(
            sklearn.metrics.precision_score(y_train, train_pred, zero_division=0)
        )
        self.metrics["val_precision"].append(
            sklearn.metrics.precision_score(y_test, val_pred, zero_division=0)
        )

        self.metrics["train_auc"].append(
            sklearn.metrics.roc_auc_score(y_train, train_pred)
        )
        self.metrics["val_auc"].append(sklearn.metrics.roc_auc_score(y_test, val_pred))

    def early_stop_check(
        self, metric: str = "val_loss", eps: float = 1e-3, limit: int = 10
    ) -> bool:
        if self.metrics[metric][-1] < self.best_loss - eps:
            self.best_loss = self.metrics[metric][-1]
            self.early_counter = 0
            self.copy = copy.deepcopy(self.layers)
        else:
            self.early_counter += 1
        return self.early_counter == limit

    def fit(
        self,
        train: tuple[np.ndarray, np.ndarray],
        test: tuple[np.ndarray, np.ndarray],
        epochs: int = 1000,
        lr: float = 0.01,
        batch_size=64,
        early_stop: bool = True,
        optimizer: Optimizer = Optimizer(),
        verbose: bool = True,
    ) -> None:
        predict = []
        self.lr = lr
        self.optimizer = optimizer
        self.optimizer.lr = self.lr
        self.copy = self.layers

        x_train, y_train = train
        x_test, y_test = test
        y_train = one_hot(y_train, 2)

        self.compute_metrics(train[0], train[1], x_test, y_test)

        with alive_bar(epochs) as bar:
            for epoch in range(epochs):
                for batch in get_batches((x_train, y_train), batch_size):
                    X, Y = batch
                    predict = self.forward(X)
                    self.backward(predict, Y)
                self.compute_metrics(train[0], train[1], x_test, y_test)
                if verbose:
                    print(
                        f"loss: {self.metrics['train_loss'][-1]:.4f} - val_loss: {self.metrics['val_loss'][-1]:.4f}",
                    )
                bar()
                if early_stop and self.early_stop_check(
                    metric="val_loss", eps=1e-4, limit=10
                ):
                    print("Early Stoppping !")
                    self.layers = self.copy
                    break
        self.metrics["epoch"] = range(0, epoch + 2)

    def save(self, name: str) -> None:
        print(f"Saving model into 'models/{name}.joblib'...")
        with open(f"models/{name}.joblib", "wb") as f:
            joblib.dump(self, f)

    def predict(self, x: np.ndarray) -> np.array:
        raw_predict = self.forward(x)
        return np.argmax(raw_predict, axis=1)

    def score(self, x: np.ndarray, y: np.ndarray) -> float:
        predict = self.predict(x)
        return sum(predict == y) / len(y)

    def plot_metrics(self):
        fig, axes = plt.subplots(2, 3)

        fig.suptitle("Learning Curves")
        data = pd.DataFrame.from_dict(self.metrics)

        axes[0, 0].set_ylim(
            (0, max(max(self.metrics["val_loss"]), max(self.metrics["train_loss"])))
        )

        sns.lineplot(
            x="epoch", y="train_loss", data=data, ax=axes[0, 0], label="Training Loss"
        )
        sns.lineplot(
            x="epoch", y="val_loss", data=data, ax=axes[0, 0], label="Validation Loss"
        )
        sns.lineplot(
            x="epoch",
            y="train_acc",
            data=data,
            ax=axes[0, 1],
            label="Training Accuracy",
        )
        sns.lineplot(
            x="epoch",
            y="val_acc",
            data=data,
            ax=axes[0, 1],
            label="Validation Accuracy",
        )
        sns.lineplot(
            x="epoch", y="train_f1", data=data, ax=axes[0, 2], label="Training F1 Score"
        )
        sns.lineplot(
            x="epoch", y="val_f1", data=data, ax=axes[0, 2], label="Validation F1 Score"
        )

        sns.lineplot(
            x="epoch",
            y="train_recall",
            data=data,
            ax=axes[1, 0],
            label="Training Recall Score",
        )
        sns.lineplot(
            x="epoch",
            y="val_recall",
            data=data,
            ax=axes[1, 0],
            label="Validation Recall Score",
        )

        sns.lineplot(
            x="epoch",
            y="train_precision",
            data=data,
            ax=axes[1, 1],
            label="Training Precision Score",
        )
        sns.lineplot(
            x="epoch",
            y="val_precision",
            data=data,
            ax=axes[1, 1],
            label="Validation Precision Score",
        )

        sns.lineplot(
            x="epoch",
            y="train_auc",
            data=data,
            ax=axes[1, 2],
            label="Training AUC Score",
        )
        sns.lineplot(
            x="epoch",
            y="val_auc",
            data=data,
            ax=axes[1, 2],
            label="Validation AUC Score",
        )

        for a in axes:
            for i in range(len(a)):
                a[i].grid(True)
                a[i].set_xlabel("Epochs")
                a[i].xaxis.set_major_locator(MaxNLocator(integer=True))

        axes[0, 0].set_ylabel("Loss")
        axes[0, 1].set_ylabel("Accuracy")
        axes[0, 2].set_ylabel("F1 Score")
        axes[1, 0].set_ylabel("Recall Score")
        axes[1, 1].set_ylabel("Precision Score")
        axes[1, 2].set_ylabel("AUC Score")

        plt.show()


if __name__ == "__main__":
    pass
