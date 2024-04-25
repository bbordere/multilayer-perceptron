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

for i in range(25):
    net = NeuralNetwork(
        [
            DenseLayer(x_train.shape[1], 2500),
            ActivationLayer("relu"),
            DenseLayer(2500, 1000),
            ActivationLayer("sigmoid"),
            DenseLayer(1000, 1000),
            ActivationLayer("relu"),
            DenseLayer(1000, 2),
            ActivationLayer("softmax"),
        ]
    )
    predict = []
    with alive_bar(1000) as bar:
        for epoch in range(1000):
            predict = net.forward(x_train)
            net.backward(predict, y_train_one)
            bar()

    pred_test = net.forward(x_test)
    labels_pred = []
    for p in pred_test:
        labels_pred.append(int(p[0] < p[1]))

    good = 0
    for i in range(len(labels_pred)):
        good += labels_pred[i] == y_test[i]

    res.append(good / len(labels_pred))

print(np.max(res))
print(np.mean(res))
