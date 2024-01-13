from pathlib import Path

import pandas as pd

from reports.report_utils import (
    load_mlruns,
    make_winkler_boxenplot,
    make_runtime_boxenplot,
)


if __name__ == "__main__":
    base_path = Path("reports/24-01-12_simple")
    commit_hashes = [
        "e5a549864be437363e7a08943253b2674848464f",
        "c494908a28df1f7e99520c957411b2eb23fe82c9",
        "ce856c515965dbd01fd0b7f23d4154e1f36c4f46",
    ]
    runs = pd.DataFrame()
    for hash in commit_hashes:
        runs_commit = load_mlruns(
            filter_string=f"tags.`mlflow.source.git.commit` = '{hash}'",
            columns=[
                "params.model",
                "metrics.winkler_val",
                "metrics.coverage_val",
            ],
        )
        runs = pd.concat([runs, runs_commit], axis=0)

    runs.drop(runs[runs["metrics.winkler_val"] > 4.5].index, inplace=True)
    runs.drop(runs[runs["params.model"] == "DummyModel"].index, inplace=True)

    make_winkler_boxenplot(
        df=runs,
        x_col="params.model",
        distinguish_covered=True,
        title="dataset: simple",
        save_path=Path(base_path, "figures/winkler.png"),
    )

    make_runtime_boxenplot(
        df=runs,
        x_col="params.model",
        title="dataset: simple",
        save_path=Path(base_path, "figures/runtime.png"),
    )
