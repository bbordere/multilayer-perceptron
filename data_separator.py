from Extractor import Extractor
import argparse
import os
import numpy as np
import pandas as pd
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
        help="percentage of training partition (default: 60)",
        choices=range(1, 101),
        metavar="{1..100}",
        type=int,
        default=80,
    )
    args = parser.parse_args()

    extractor = Extractor(args.path, names=columns_names)
    extractor.data = extractor.data.sort_values("diagnosis")

    x_train, y_train, x_test, y_test = stratified_train_test_split(
        extractor.data, "diagnosis", store=True
    )

    # train_cnt.to_csv(f"{dir}/{name}_train.csv", index=False)
    # validation_cnt.to_csv(f"{dir}/{name}_test.csv", index=False)

    # validation_part = int(((100 - args.percentage) / 100) * len(extractor.data))
    # print(validation_part)

    # # n_m = int(validation_part * (60 / 100))
    # n_m = validation_part // 2

    # print(n_m, validation_part - n_m)

    # train_cnt = extractor.data.drop(extractor.data.index[:n_m]).drop(
    #     extractor.data.index[-(validation_part - n_m) :]
    # )
    # validation_cnt = pd.concat(
    #     [extractor.data[:n_m], extractor.data[-(validation_part - n_m) :]]
    # )

    # train_cnt = train_cnt.apply(np.random.permutation, axis=0)
    # validation_cnt = validation_cnt.apply(np.random.permutation, axis=0)

    # dir = os.path.dirname(args.path)
    # name = os.path.splitext(os.path.basename(args.path))[0]

    # train_cnt.to_csv(f"{dir}/{name}_train.csv", index=False)
    # validation_cnt.to_csv(f"{dir}/{name}_test.csv", index=False)


if __name__ == "__main__":
    main()

# from Extractor import Extractor
# import argparse
# import os
# import numpy as np

# columns_names = [
#     "id",
#     "diagnosis",
#     "radius_mean",
#     "texture_mean",
#     "perimeter_mean",
#     "area_mean",
#     "smoothness_mean",
#     "compactness_mean",
#     "concavity_mean",
#     "concave points_mean",
#     "symmetry_mean",
#     "fractal_dimension_mean",
#     "radius_se",
#     "texture_se",
#     "perimeter_se",
#     "area_se",
#     "smoothness_se",
#     "compactness_se",
#     "concavity_se",
#     "concave points_se",
#     "symmetry_se",
#     "fractal_dimension_se",
#     "radius_worst",
#     "texture_worst",
#     "perimeter_worst",
#     "area_worst",
#     "smoothness_worst",
#     "compactness_worst",
#     "concavity_worst",
#     "concave points_worst",
#     "symmetry_worst",
#     "fractal_dimension_worst",
# ]

# np.random.seed(4242)


# def main() -> None:
#     parser = argparse.ArgumentParser(
#         prog="data_separator",
#         description="Split dataset into two parts, one for training and the other for validation",
#         epilog="Text at the bottom of help",
#     )
#     parser.add_argument("path", help="dataset path to split", type=str)
#     parser.add_argument(
#         "-p",
#         "--percentage",
#         help="percentage of training partition (default: 60)",
#         choices=range(1, 101),
#         metavar="{1..100}",
#         type=int,
#         default=80,
#     )
#     args = parser.parse_args()

#     extractor = Extractor(args.path, names=columns_names)
#     extractor.data = extractor.data.sample(frac=1).reset_index(drop=True)

#     train_part = int((args.percentage / 100) * len(extractor.data))
#     train_cnt = extractor.data[:train_part]
#     test_cnt = extractor.data[train_part:]
#     dir = os.path.dirname(args.path)
#     name = os.path.splitext(os.path.basename(args.path))[0]

#     train_cnt.to_csv(f"{dir}/{name}_train.csv", index=False)
#     test_cnt.to_csv(f"{dir}/{name}_test.csv", index=False)


# if __name__ == "__main__":
#     main()
