import mlflow
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

plt.style.use("seaborn-v0_8-white")


def load_mlruns(filter_string="", columns=None, compute_runtime=True):
    experiments = mlflow.search_experiments()
    experiment_ids = [experiment.experiment_id for experiment in experiments]
    all_runs = mlflow.search_runs(experiment_ids, filter_string)

    runs = all_runs.copy() if columns is None else all_runs[columns].copy()

    if compute_runtime:
        runs["runtime"] = (
            all_runs["end_time"] - all_runs["start_time"]
        ).dt.total_seconds()

    runs.dropna(inplace=True)
    return runs


def make_winkler_boxenplot(
    df, x_col, distinguish_covered=False, title="", save_path=None
):
    plt.figure()
    if distinguish_covered:
        df["Validation coverage"] = df["metrics.coverage_val"].apply(
            lambda x: "> 0.9" if x > 0.9 else "<= 0.9"
        )
        sns.boxenplot(
            x=x_col,
            y="metrics.winkler_val",
            hue="Validation coverage",
            data=df,
            palette={"> 0.9": "green", "<= 0.9": "red"},
        )
        plt.legend(title="val coverage", loc="upper right", frameon=True)
    else:
        sns.boxenplot(x=x_col, y="metrics.winkler_val", data=df)

    baseline = 4.49
    plt.text(x=-1, y=baseline - 0.025, s="Baseline", color="black")
    plt.axhline(baseline, color="black", linestyle="--")

    plt.xticks(rotation=90)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()


def make_runtime_boxenplot(df, x_col, title="", save_path=None):
    plt.figure()
    sns.boxenplot(x=x_col, y="runtime", data=df)
    plt.title(title)
    plt.xticks(rotation=90)
    plt.yscale("log")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
