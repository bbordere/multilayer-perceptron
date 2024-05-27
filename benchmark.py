from NeuralNetwork import NeuralNetwork
from Layer import *
from Extractor import Extractor
import joblib
import argparse
from utils import *
import sklearn
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns

np.random.seed(4242)


def plots_optimizers(models: list[NeuralNetwork]) -> None:
    fig, axes = plt.subplots(1, 2)
    for i in range(len(models)):
        data = pd.DataFrame.from_dict(models[i].metrics)
        sns.lineplot(
            x="epoch",
            y="val_loss",
            data=data,
            ax=axes[0],
            label=str(models[i].optimizer),
        )
        sns.lineplot(
            x="epoch",
            y="val_acc",
            data=data,
            ax=axes[1],
            label=str(models[i].optimizer),
        )

    plt.show()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="benchmark multiple models",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("train_path", help="train dataset path", type=str)
    parser.add_argument("valid_path", help="validation dataset path", type=str)
    args = parser.parse_args()

    extractor = Extractor(args.train_path, header="")
    x_train, y_train = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    extractor = Extractor(args.valid_path, header="")
    x_valid, y_valid = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    x_train = x_train.values
    x_valid = x_valid.values

    model = NeuralNetwork(
        [
            DenseLayer(x_train.shape[1], 30),
            ActivationLayer("relu"),
            DenseLayer(30, 20),
            ActivationLayer("relu"),
            DenseLayer(20, 10),
            ActivationLayer("relu"),
            DenseLayer(10, 2),
            SoftmaxLayer(),
        ]
    )

    model1 = copy.deepcopy(model)
    model2 = copy.deepcopy(model)
    model3 = copy.deepcopy(model)
    model4 = copy.deepcopy(model)

    LR = 0.001
    BS = 32
    EPOCHS = 200

    model1.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        verbose=False,
    )

    model2.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=SGDMOptimizer(),
        verbose=False,
    )

    model3.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=AdamOptimizer(),
        verbose=False,
    )

    model4.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=RMSPropOptimizer(),
        verbose=False,
    )

    print(
        model1.score(x_valid, y_valid),
        model1.metrics["val_loss"][-1],
        model1.metrics["epoch"][-1],
    )
    print(
        model2.score(x_valid, y_valid),
        model2.metrics["val_loss"][-1],
        model2.metrics["epoch"][-1],
    )
    print(
        model3.score(x_valid, y_valid),
        model3.metrics["val_loss"][-1],
        model3.metrics["epoch"][-1],
    )
    print(
        model4.score(x_valid, y_valid),
        model4.metrics["val_loss"][-1],
        model4.metrics["epoch"][-1],
    )

    plots_optimizers([model1, model2, model3, model4])


if __name__ == "__main__":
    main()
