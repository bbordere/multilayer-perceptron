import sklearn.metrics
from NeuralNetwork import NeuralNetwork
from Layer import *
from Extractor import Extractor
import joblib
import argparse
import matplotlib.pyplot as plt
import sklearn
import utils


def main() -> None:

    parser = argparse.ArgumentParser(
        prog="predict",
        description="get predictions with trained model",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("path", help="dataset path", type=str)
    parser.add_argument("model", help="trained model path", type=str)
    parser.add_argument(
        "-p",
        "--plot",
        help="plot the confusion matrix",
        action="store_true",
    )
    args = parser.parse_args()

    net: NeuralNetwork = joblib.load(args.model)
    extractor = Extractor(args.path, header=[])
    extractor.keep_range_columns((1, 32))
    x, y = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})
    y = y.to_numpy()
    predict = net.predict(x)

    print("Acc:", net.score(x, y))
    print("Loss:", utils.BCE(utils.one_hot(y, 2), net.forward(x)))

    if not args.plot:
        return
    sklearn.metrics.ConfusionMatrixDisplay.from_predictions(
        y_true=y,
        y_pred=predict,
        display_labels=[0, 1],
        colorbar=False,
        cmap=plt.cm.Blues,
    )
    plt.show()


if __name__ == "__main__":
    main()
