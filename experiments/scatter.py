if __name__ == "__main__":
    import argparse
    import os
    import pickle

    import matplotlib.pyplot as plt

    plt.rc("text", usetex=True)

    parser = argparse.ArgumentParser()
    parser.add_argument("--sprop1", type=str)
    parser.add_argument("--icons1", type=str)
    parser.add_argument("--sprop2", type=str)
    parser.add_argument("--icons2", type=str)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--slimit", type=int, default=40)
    args = parser.parse_args()

    # data layout:
    # first column: number of variables
    # second column: time in seconds
    # make a heatmap
    # x-axis: number of variables
    # y-axis: time in seconds classified into 10 buckets (0-100, 100-200, ..., 900-1000)
    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=(6, 3), dpi=300)
    # remove ax2 ytick
    ax2.tick_params(axis="y", which="both", left=False, right=False, labelleft=False)
    ax1.sharey(ax2)

    with open(args.sprop1, "rb") as f:
        data_sprop1 = pickle.load(f)

    with open(args.icons1, "rb") as f:
        data_icons1 = pickle.load(f)

    # prune lines whose 1st column > args.slimit
    # prune lines whose 2nd column > args.limit
    data_sprop1 = data_sprop1[data_sprop1[:, 1] <= args.limit]
    data_sprop1 = data_sprop1[data_sprop1[:, 0] <= args.slimit]
    data_icons1 = data_icons1[data_icons1[:, 1] <= args.limit]
    data_icons1 = data_icons1[data_icons1[:, 0] <= args.slimit]

    path_sprop1 = ax1.scatter(
        data_sprop1[:, 0],
        data_sprop1[:, 1],
        s=5,
        color="green",
        label="Shape Propagation",
    )
    path_icons1 = ax1.scatter(
        data_icons1[:, 0],
        data_icons1[:, 1],
        s=4,
        color="red",
        label="Input Constraints",
    )

    with open(args.sprop2, "rb") as f:
        data_sprop2 = pickle.load(f)

    with open(args.icons2, "rb") as f:
        data_icons2 = pickle.load(f)

    data_sprop2 = data_sprop2[data_sprop2[:, 1] <= args.limit]
    data_sprop2 = data_sprop2[data_sprop2[:, 0] <= args.slimit]
    data_icons2 = data_icons2[data_icons2[:, 1] <= args.limit]
    data_icons2 = data_icons2[data_icons2[:, 0] <= args.slimit]

    path_sprop2 = ax2.scatter(
        data_sprop2[:, 0],
        data_sprop2[:, 1],
        s=5,
        color="green",
        label="Shape Propagation",
    )
    path_icons2 = ax2.scatter(
        data_icons2[:, 0],
        data_icons2[:, 1],
        s=4,
        color="red",
        label="Input Constraints",
    )

    # x y log scale
    ax1.set_yscale("log")

    # xlabel on top.
    ax1.set_title("\\textbf{PyTorch}", loc="left")
    ax2.set_title("\\textbf{TensorFlow}", loc="right")
    # set title for combined plot
    # fig.suptitle("\# Symbols (i.e., $|A\cup I|$)", y=0.0)
    # make title upper?
    ax1.set_ylabel(f"Time (seconds)")
    fig.suptitle("\# Symbols (i.e., $|A\cup I|$)", y=0.05)

    # horitontal line --
    ax1.grid(axis="y", linestyle="--", alpha=0.5)
    ax2.grid(axis="y", linestyle="--", alpha=0.5)

    # add legend and reuse it
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=2,
        fancybox=True,
        shadow=True,
        bbox_to_anchor=(0.53, 0.90),
    )
    path_sprop1.set_alpha(0.1)
    path_icons1.set_alpha(0.1)
    path_sprop2.set_alpha(0.1)
    path_icons2.set_alpha(0.1)

    fig.tight_layout()
    plt.savefig(os.path.join("results", "scatter.png"), bbox_inches="tight")
    plt.savefig(os.path.join("results", "scatter.pdf"), bbox_inches="tight")
