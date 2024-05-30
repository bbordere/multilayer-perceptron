import numpy as np
import sklearn
import sklearn.metrics
from Extractor import Extractor


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], Y[batch_indices]


def BCE(y_true, y_pred, eps: float = 1e-16):
    y_pred = np.clip(y_pred, eps, 1 - eps)
    log_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(log_loss)


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def data_process(
    path: str, test_part: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    assert test_part > 0.05
    assert test_part < 1
    extractor = Extractor(path, header=[])
    extractor.keep_range_columns((1, 32))
    x, y = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})
    limit = int((1 - test_part) * len(x))
    x_train = x[:limit].values
    y_train = y[:limit]
    x_valid = x[limit:].values
    y_valid = y[limit:]

    return x_train, y_train, x_valid, y_valid
