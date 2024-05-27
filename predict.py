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
    net: NeuralNetwork = joblib.load("models/adam.joblib")
    extractor = Extractor("data/data_test.csv", header=[])
    extractor.keep_range_columns((1, 32))
    x, y = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    predict = net.predict(x)

    confusion_matrix = sklearn.metrics.confusion_matrix(y, predict)
    cm_display = sklearn.metrics.ConfusionMatrixDisplay(
        confusion_matrix=confusion_matrix, display_labels=[0, 1]
    )
    cm_display.plot(cmap=plt.cm.Blues)
    print("Acc:", sklearn.metrics.accuracy_score(y, predict))
    print("Loss:", sklearn.metrics.log_loss(utils.one_hot(y, 2), net.forward(x)))
    plt.show()


if __name__ == "__main__":
    main()
