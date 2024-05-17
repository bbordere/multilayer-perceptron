from NeuralNetwork import NeuralNetwork
from Layer import *
from Extractor import Extractor

extractor = Extractor("data.csv", 1)
extractor.keep_range_columns((1, 32))
x, y = extractor.get_data_training("diagnosis", replace_params={"B": 0, "M": 1})


PERCENTAGE = 70

train_part = int((PERCENTAGE / 100) * len(x))
test_part = len(x) - train_part


x_train, y_train = x[:train_part], y[:train_part]
x_test, y_test = x[test_part:], y[test_part:]

np.random.seed(42)

net = NeuralNetwork(
    [
        DenseLayer(x_train.shape[1], 16),
        ActivationLayer("relu"),
        DenseLayer(16, 32),
        ActivationLayer("sigmoid"),
        DenseLayer(32, 16),
        ActivationLayer("relu"),
        DenseLayer(16, 2),
        ActivationLayer("softmax"),
    ]
)

net.fit(
    (x_train, y_train),
    (x_test, y_test),
    epochs=50,
    lr=0.02,
    batch_size=256,
)
print(net.score(x_test, y_test))
net.plot_metrics()
