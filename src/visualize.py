import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os
import argparse
from utils import columns_names
from Extractor import Extractor


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="visualize",
        description="Make histplot and pairplot for data visualization",
        epilog="Text at the bottom of help",
    )
    parser.add_argument("path", help="dataset to visualize", type=str)
    args = parser.parse_args()

    extractor = Extractor(args.path, names=columns_names)

    cols = [
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
    ]

    if not os.path.exists("plots"):
        os.makedirs("plots")

    print("Saving plots/hist.png...")
    plt.figure()
    sns.histplot(
        data=extractor.data,
        x="diagnosis",
        hue="diagnosis",
        palette=["tab:red", "tab:green"],
    )
    plt.savefig("plots/hist.png")

    print("Saving plots/pairplot.png...")
    plt.figure()
    sns.pairplot(
        data=extractor.data[cols], hue="diagnosis", palette=["tab:red", "tab:green"]
    )
    plt.savefig("plots/pairplot.png")
    print("Plots saved!")


if __name__ == "__main__":
    main()
