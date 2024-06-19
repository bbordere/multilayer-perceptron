from NeuralNetwork import NeuralNetwork
from Layer import *
import argparse
from utils import *

np.random.seed(4242)
# 15 6


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="train",
        description="train multilayer perceptron",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("train_path", help="train dataset path", type=str)
    parser.add_argument(
        "-np",
        "--noplot",
        help="disable the plot the learning curves",
        action="store_true",
    )
    parser.add_argument(
        "-n", "--name", help="name the model", type=str, default="model"
    )
    args = parser.parse_args()

    x_train, y_train, x_valid, y_valid = data_process(args.train_path, 0.25)

    net = NeuralNetwork(
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
        # [
        #     DenseLayer(x_train.shape[1], 30),
        #     ActivationLayer("relu"),
        #     DenseLayer(30, 10),
        #     ActivationLayer("relu"),
        #     DenseLayer(10, 2),
        #     SoftmaxLayer(),
        # ]
        # [
        #     DenseLayer(x_train.shape[1], 10),
        #     ActivationLayer("elu"),
        #     DenseLayer(10, 6),
        #     ActivationLayer("tanh"),
        #     DenseLayer(6, 6),
        #     ActivationLayer("relu"),
        #     DenseLayer(6, 10, "xavier"),
        #     ActivationLayer("tanh"),
        #     DenseLayer(10, 2, "xavier"),
        #     SoftmaxLayer(),
        # ]
        # [
        #     DenseLayer(30, 24),
        #     ActivationLayer("sigmoid"),
        #     DenseLayer(24, 24),
        #     ActivationLayer("sigmoid"),
        #     DenseLayer(24, 24),
        #     ActivationLayer("sigmoid"),
        #     DenseLayer(24, 2),
        #     SoftmaxLayer(),
        # ]
    )
    print(net)

    net.fit(
        (x_train, y_train),
        (x_valid, y_valid),
        epochs=500,
        lr=0.001,
        batch_size=16,
        optimizer=AdamOptimizer(),
        # optimizer=SGDMOptimizer(),
        # optimizer=RMSPropOptimizer(),
        # early_stop=False,
        # optimizer=SGDMOptimizer(),
        # optimizer=RMSPropOptimizer(),
        compute_all=not args.noplot,
    ),

    if not args.noplot:
        net.plot_metrics()

    net.save(args.name)


if __name__ == "__main__":
    main()
