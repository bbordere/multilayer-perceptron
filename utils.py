import numpy as np
from Extractor import Extractor
from typing import Generator
import pandas as pd

columns_names = [
    "id",
    "diagnosis",
    "radius_mean",
    "texture_mean",
    "perimeter_mean",
    "area_mean",
    "smoothness_mean",
    "compactness_mean",
    "concavity_mean",
    "concave points_mean",
    "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se",
    "texture_se",
    "perimeter_se",
    "area_se",
    "smoothness_se",
    "compactness_se",
    "concavity_se",
    "concave points_se",
    "symmetry_se",
    "fractal_dimension_se",
    "radius_worst",
    "texture_worst",
    "perimeter_worst",
    "area_worst",
    "smoothness_worst",
    "compactness_worst",
    "concavity_worst",
    "concave points_worst",
    "symmetry_worst",
    "fractal_dimension_worst",
]


def get_batches(dataset: tuple[np.ndarray, np.ndarray], batch_size: int) -> Generator:
    """generator function for mini_batch

    Args:
        dataset (tuple[np.ndarray, np.ndarray]): dataset to split into minibatch
        batch_size (int): size of mini_batch

    Yields:
        np.ndarray: mini_batch
    """

    X, Y = dataset
    n_samples = X.shape[0]
    # indices = np.arange(n_samples)
    # np.random.shuffle(indices)
    indices = np.random.permutation(len(X))
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        batch_indices = indices[start:end]
        yield X[batch_indices], Y[batch_indices]


def BCE(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-16) -> float:
    """binary cross entropy loss function

    Args:
        y_true (np.ndarray): target values
        y_pred (np.ndarray): predicted values
        eps (float, optional): epsilon value for clipping. Defaults to 1e-16.

    Returns:
        float: mean loss value for predictions
    """
    y_pred = np.clip(y_pred, eps, 1 - eps)
    log_loss = -(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return np.mean(log_loss)


def one_hot(a: np.ndarray, num_classes: int) -> np.ndarray:
    """onehot encoding

    Args:
        a (np.ndarray): array to encode
        num_classes (int): number of class

    Returns:
        np.ndarray: onehot_encoded array
    """
    return np.squeeze(np.eye(num_classes)[a.reshape(-1)])


def data_process(
    path: str, test_part: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """process data for learnig phase

    Args:
        path (str): path of dataset
        test_part (float): test_part for spliting

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: splited parts
    """
    assert test_part > 0.01
    assert test_part < 1
    extractor = Extractor(path, header=[])
    extractor.keep_range_columns((1, 32))

    x, y = extractor.get_data("diagnosis", replace_params={"B": 0, "M": 1})

    df = pd.concat([x, y], axis=1)
    x_train, y_train, x_valid, y_valid = stratified_train_test_split(
        df, "diagnosis", test_size=test_part
    )
    return x_train.values, y_train.values, x_valid.values, y_valid.values


def compute_cm(pred: np.ndarray, target: np.ndarray) -> dict:
    """compute confusion matrix and other metrics

    Args:
        pred (np.ndarray): predictions of neural network
        target (np.ndarray): target values for predictions

    Returns:
        dict: contains metrics
    """
    np.seterr(divide="ignore", invalid="ignore")
    res = {}
    res["fp"] = np.sum((pred == 1) & (target == 0))
    res["tp"] = np.sum((pred == 1) & (target == 1))

    res["fn"] = np.sum((pred == 0) & (target == 1))
    res["tn"] = np.sum((pred == 0) & (target == 0))

    res["recall"] = res["tp"] / (res["tp"] + res["fn"])
    res["precision"] = res["tp"] / (res["tp"] + res["fp"])
    res["f1"] = 2 / ((1 / res["precision"]) + 1 / res["recall"])
    res["fpr"] = res["fp"] / (res["fp"] + res["tn"])
    return res


def stratified_train_test_split(
    data: pd.DataFrame,
    label: str,
    train_path: str = "data/data_train.csv",
    test_path: str = "data/data_test.csv",
    test_size: float = 0.2,
    store: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """stratified split to ensure same repartition on label

    Args:
        data (pd.DataFrame): data to split
        label (str): label to split
        train_path (str, optional): path to train path if store flag is set. Defaults to "data/data_train.csv".
        test_path (str, optional): path to test path if store flag is set. Defaults to "data/data_test.csv".
        test_size (float, optional): test_part for spliting. Defaults to 0.2.
        store (bool, optional): flag to store splitted part on csv files. Defaults to False.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]: splitted parts
    """
    x = data.drop(columns=[label])
    y = data[label]

    test_counts = (y.value_counts() * test_size).round().astype(int)

    train_id, test_id = [], []

    for label, count in test_counts.items():
        indices = y[y == label].index.to_numpy()
        np.random.shuffle(indices)
        test_id.extend(indices[:count])
        train_id.extend(indices[count:])

    x_train, y_train, x_test, y_test = (
        x.loc[train_id],
        y.loc[train_id],
        x.loc[test_id],
        y.loc[test_id],
    )

    train_df = pd.concat([x_train, y_train], axis=1)[data.columns]
    test_df = pd.concat([x_test, y_test], axis=1)[data.columns]

    if store:
        train_df.to_csv(train_path, header=columns_names)
        test_df.to_csv(test_path, header=columns_names)

    return x_train, y_train, x_test, y_test
