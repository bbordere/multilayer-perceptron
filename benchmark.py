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


def plots_optimizers(models: list[NeuralNetwork]) -> None:
    fig, axes = plt.subplots(1, 2)
    metrics = ["val_acc", "val_loss", "epoch"]
    for i in range(len(models)):
        met = dict((k, models[i].metrics[k]) for k in metrics if k in models[i].metrics)
        data = pd.DataFrame.from_dict(met)
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
    np.random.seed(4242)

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
            ActivationLayer("tanh"),
            DenseLayer(20, 2),
            SoftmaxLayer(),
        ]
        # [
        #     DenseLayer(30, 32),
        #     ActivationLayer("relu"),
        #     DenseLayer(32, 20),
        #     ActivationLayer("sigmoid"),
        #     DenseLayer(20, 10),
        #     ActivationLayer("sigmoid"),
        #     DenseLayer(10, 2),
        #     SoftmaxLayer(),
        # ]
    )

    LR = 0.001
    BS = 64
    EPOCHS = 500

    model1: NeuralNetwork = copy.deepcopy(model)
    model2: NeuralNetwork = copy.deepcopy(model)
    model3: NeuralNetwork = copy.deepcopy(model)
    model4: NeuralNetwork = copy.deepcopy(model)

    models = [model1, model2, model3, model4]
    optimizers = [Optimizer(), SGDMOptimizer(), RMSPropOptimizer(), AdamOptimizer()]

    for i in range(4):
        models[i].fit(
            (x_train, y_train),
            (x_valid, y_valid),
            epochs=EPOCHS,
            lr=LR,
            batch_size=BS,
            verbose=False,
            compute_all=False,
            optimizer=optimizers[i],
        )

    for i in range(4):
        print(
            optimizers[i].name,
            models[i].score(x_valid, y_valid),
            models[i].metrics["val_loss"][-11],
            models[i].metrics["epoch"][-11],
        )
    plots_optimizers([model1, model2, model3, model4])


if __name__ == "__main__":
    main()
