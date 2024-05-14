import numpy as np


def get_batches(dataset, batch_size):
    X, Y = dataset
    n_samples = X.shape[0]
    indices = np.arange(n_samples)
    np.random.shuffle(indices)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        yield X[start:end], Y[start:end]


def one_hot(a, num_classes):
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])
