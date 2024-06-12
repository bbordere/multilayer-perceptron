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

np.random.seed(42)


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
    extractor.keep_range_columns((1, 32))
    x_train, y_train = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    extractor = Extractor(args.valid_path, header="")
    extractor.keep_range_columns((1, 32))
    x_valid, y_valid = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    x_train = x_train.values
    y_train = y_train.values
    x_valid = x_valid.values
    y_valid = y_valid.values

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

    model1: NeuralNetwork = copy.deepcopy(model)
    model2: NeuralNetwork = copy.deepcopy(model)
    model3: NeuralNetwork = copy.deepcopy(model)
    model4: NeuralNetwork = copy.deepcopy(model)

    LR = 0.001
    BS = 16
    EPOCHS = 500

    model1.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        verbose=False,
        compute_all=True,
        # early_stop=False,
    )

    model2.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=SGDMOptimizer(),
        verbose=False,
        compute_all=True,
        # early_stop=False,
    )

    model3.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=AdamOptimizer(),
        verbose=False,
        compute_all=True,
        # early_stop=False,
    )

    model4.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=EPOCHS,
        lr=LR,
        batch_size=BS,
        optimizer=RMSPropOptimizer(),
        verbose=False,
        compute_all=True,
        # early_stop=False,
    )

    print(
        model1.optimizer.name,
        model1.score(x_valid, y_valid),
        model1.metrics["val_loss"][-11],
        model1.metrics["epoch"][-11],
    )
    print(
        model2.optimizer.name,
        model2.score(x_valid, y_valid),
        model2.metrics["val_loss"][-11],
        model2.metrics["epoch"][-11],
    )
    print(
        model3.optimizer.name,
        model3.score(x_valid, y_valid),
        model3.metrics["val_loss"][-11],
        model3.metrics["epoch"][-11],
    )
    print(
        model4.optimizer.name,
        model4.score(x_valid, y_valid),
        model4.metrics["val_loss"][-11],
        model4.metrics["epoch"][-11],
    )

    plots_optimizers([model1, model2, model3, model4])


if __name__ == "__main__":
    main()
