from pathlib import Path

import pandas as pd

from reports.report_utils import (
    load_mlruns,
    make_winkler_boxenplot,
    make_runtime_boxenplot,
)


if __name__ == "__main__":
    base_path = Path("reports/24-01-13_simple-quantile")
    commit_hash = "cf1ed91245cc919052c5584429220f1588f02bc0"
    runs = load_mlruns(
        filter_string=f"tags.`mlflow.source.git.commit` = '{commit_hash}'",
        columns=[
            "params.model",
            "metrics.winkler_val",
            "metrics.coverage_val",
        ],
    )

    runs.drop(runs[runs["metrics.winkler_val"] > 4.5].index, inplace=True)
    runs.drop(runs[runs["params.model"] == "DummyModel"].index, inplace=True)

    make_winkler_boxenplot(
        df=runs,
        x_col="params.model",
        distinguish_covered=True,
        title="dataset: simple",
        save_path=Path(base_path, "figures/winkler.png"),
    )
