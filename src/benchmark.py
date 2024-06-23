from NeuralNetwork import NeuralNetwork
from Layer import *
from Extractor import Extractor
import argparse
from utils import *
import pandas as pd
import copy
import matplotlib.pyplot as plt
import seaborn as sns
import example_models
from tabulate import tabulate


def plots_optimizers(models: list[NeuralNetwork], names: list[str]) -> None:
    """
    plots validation loss against the epoch number for each model and optimizer

    Args:
        models (list[NeuralNetwork]): list of NeuralNetwork objects
        names (list[str]): list of names corresponding to the models
    """
    fig, axes = plt.subplots(2, 2)
    metrics = ["val_acc", "val_loss", "epoch"]
    for i in range(len(models)):
        met = dict((k, models[i].metrics[k]) for k in metrics if k in models[i].metrics)
        data = pd.DataFrame.from_dict(met)
        sns.lineplot(
            x="epoch",
            y="val_loss",
            data=data,
            ax=axes[i // 8, ((i // 4)) % 2],
            label=str(models[i].optimizer),
        ).set(title=f"Model {names[i // 4]}")
    plt.tight_layout()
    plt.show()


def print_metrics(models: list[NeuralNetwork], names: list[str], datas: tuple) -> None:
    """
    prints the metrics for each model and optimizer combination in a tabular format

    Args:
        models (list[NeuralNetwork]): list of NeuralNetwork objects
        names (list[str]): list of names corresponding to the models
        datas (tuple): tuple containing the training data and labels
    """
    headers = ["Optmizer", "Accuracy", "Val Loss", "Best Epoch"]
    print("----------Results----------")
    for i in range(0, len(models), 4):
        data = [
            [
                model.optimizer.name,
                "{:.6f}".format(model.score(datas[0], datas[1])),
                "{:.6f}".format(model.metrics["val_loss"][-model.patience]),
                model.metrics["epoch"][-model.patience],
            ]
            for model in models[i : i + 4]
        ]
        print(f"Model: {names[i // 4]}")
        print(tabulate(data, headers=headers, tablefmt="pretty"))
        print()


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmark",
        description="benchmark multiple models",
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

    LR = 0.001
    BS = 8
    EPOCHS = 250

    models = [copy.deepcopy(example_models.RELU_SIG_TANH) for _ in range(4)]
    models.extend([copy.deepcopy(example_models.SUBJECT) for _ in range(4)])
    models.extend([copy.deepcopy(example_models.MIX) for _ in range(4)])
    models.extend([copy.deepcopy(example_models.FULL_RELU) for _ in range(4)])

    optimizers = [Optimizer(), SGDMOptimizer(), RMSPropOptimizer(), AdamOptimizer()]
    names = ["relu_sig", "subject", "mix", "full_relu"]

    for i in range(len(models)):
        if i % 4 == 0:
            print(f"Model: {names[i // 4]}")
            print(models[i], end="")
        np.random.seed(4242)
        models[i].fit(
            (x_train, y_train),
            (x_valid, y_valid),
            epochs=EPOCHS,
            lr=LR,
            batch_size=BS,
            verbose=False,
            compute_all=False,
            optimizer=optimizers[i % 4],
            # patience=100,
        )
        if i % 4 == 3:
            print("\n\n")

    print_metrics(models, names, (x_train, y_train))
    plots_optimizers(models, names)


if __name__ == "__main__":
    main()
