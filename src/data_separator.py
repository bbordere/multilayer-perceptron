from Extractor import Extractor
import argparse
import numpy as np
from utils import *

np.random.seed(4242)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="data_separator",
        description="Split dataset into two parts, one for training and the other for validation",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("path", help="dataset path to split", type=str)
    parser.add_argument(
        "-p",
        "--percentage",
        help="percentage of training partition (default: 80)",
        choices=range(1, 101),
        metavar="{1..100}",
        type=int,
        default=80,
    )
    args = parser.parse_args()

    extractor = Extractor(args.path, names=columns_names)
    extractor.data = extractor.data.sort_values("diagnosis")
    stratified_train_test_split(
        extractor.data, "diagnosis", store=True, test_size=(100 - args.percentage) / 100
    )
    print(
        f"Dataset successfully split with {args.percentage}/{100 - args.percentage} ratio!"
    )


if __name__ == "__main__":
    main()
