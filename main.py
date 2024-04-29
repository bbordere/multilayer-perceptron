from NeuralNetwork import NeuralNetwork
from Layer import *
import pandas as pd
import numpy as np
from Extractor import Extractor
from alive_progress import alive_bar


extractor = Extractor("data.csv")
extractor.keep_range_columns((1, 22))
x, y = extractor.get_data_training("diagnosis", True)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


train_part = (len(x) // 100) * 80
test_part = len(x) - train_part


x_train, y_train = x[:train_part], y[:train_part]
x_test, y_test = x[test_part:], y[test_part:]


y_train_one = one_hot(y_train, 2)

res = []
np.random.seed(42)

net = NeuralNetwork(
    [
        DenseLayer(x_train.shape[1], 25),
        ActivationLayer("relu"),
        DenseLayer(25, 30),
        ActivationLayer("relu"),
        DenseLayer(30, 2),
        ActivationLayer("sigmoid"),
        DenseLayer(2, 2),
        ActivationLayer("softmax"),
    ]
)
net.fit(x_train, y_train_one, 200)
net.predict(x_test)
print(net.score(x_test, y_test))
