from Extractor import Extractor
import argparse
import os
import numpy as np

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
        help="percentage of training partition (default: 60)",
        choices=range(1, 101),
        metavar="{1..100}",
        type=int,
        default=80,
    )
    args = parser.parse_args()

    extractor = Extractor(args.path, names=columns_names)
    extractor.data = extractor.data.sample(frac=1).reset_index(drop=True)

    train_part = int((args.percentage / 100) * len(extractor.data))
    train_cnt = extractor.data[:train_part]
    test_cnt = extractor.data[train_part:]
    dir = os.path.dirname(args.path)
    name = os.path.splitext(os.path.basename(args.path))[0]

    train_cnt.to_csv(f"{dir}/{name}_train.csv", index=False)
    test_cnt.to_csv(f"{dir}/{name}_test.csv", index=False)


if __name__ == "__main__":
    main()
