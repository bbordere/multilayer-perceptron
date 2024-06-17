import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import os


def main() -> None:
    data = pd.read_csv("data/data_train.csv")

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

    # sns.pairplot(data=data[cols], hue="diagnosis", palette=["tab:red", "tab:green"])
    # plt.savefig("plots/pairplot.png")
    sns.histplot(data=data, x="diagnosis", hue="diagnosis")
    plt.savefig("plots/hist.png")


if __name__ == "__main__":
    main()
