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


def plots_optimizers(models: list[NeuralNetwork], names: list[str]) -> None:
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

    # model =

    LR = 0.001
    BS = 16
    EPOCHS = 250

    models = [copy.deepcopy(example_models.RELU_SIG_TANH) for _ in range(4)]
    models.extend([copy.deepcopy(example_models.SUBJECT) for _ in range(4)])
    models.extend([copy.deepcopy(example_models.MIX) for _ in range(4)])
    models.extend([copy.deepcopy(example_models.FULL_RELU) for _ in range(4)])
    optimizers = [Optimizer(), SGDMOptimizer(), RMSPropOptimizer(), AdamOptimizer()]
    names = ["RELU_SIG", "SUBJECT", "MIX", "FULL_RELU"]

    for i in range(len(models)):
        if i % 4 == 0:
            print(names[i // 4])
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
            early_stop=False,
        )

    for i in range(len(models)):
        if i % 4 == 0:
            print(names[i // 4])
        print(
            optimizers[i % 4].name + "->",
            f"Acc: {models[i].score(x_valid, y_valid)}",
            f"Loss: {models[i].metrics['val_loss'][-models[i].patience]}",
            f"Epoch: {models[i].metrics['epoch'][-models[i].patience]}",
        )
    plots_optimizers(models, names)


if __name__ == "__main__":
    main()
