from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import mlflow
import tempfile

MAX_DENSITY = 1


class Analysis:
    "Utility functions for analyzing trained models."

    def __init__(
        self,
        y: np.ndarray,
        y_pred: np.ndarray,
        stage: str,
        model_name: str,
    ) -> None:
        idxs_sorted = np.argsort(y, axis=0).flatten()
        self.y = y[idxs_sorted].flatten()
        self.y_pred = y_pred[idxs_sorted]
        self.is_covered = (self.y_pred[:, 0] <= self.y) & (self.y <= self.y_pred[:, 1])
        self.percent_covered = int(self.is_covered.mean() * 100)
        self.avg_width = (self.y_pred[:, 1] - self.y_pred[:, 0]).mean()

        self.stage = stage
        self.model_name = model_name

        self.bins = 80
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        self.target_color = color_cycle[0]
        self.prediction_color = color_cycle[1]
        self.covered_color = "green"
        self.not_covered_color = "red"

    def full_analysis(self):
        """Run all analysis functions."""
        self.marginals()
        self.target_vs_predicted()

    def marginals(self):
        """Show marginal distributions for target values, prediction interval means, and
        prediction interval widths."""
        y_pred_center = self.y_pred.mean(axis=1)
        y_pred_width = self.y_pred[:, 1] - self.y_pred[:, 0]

        fig, axs = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))

        axs[0].hist(
            self.y,
            bins=self.bins,
            density=True,
            alpha=0.7,
            color=self.target_color,
            label="target",
        )
        self._plot_hist_or_bar(
            axs[0],
            y_pred_center,
            alpha=0.7,
            color=self.prediction_color,
            label="prediction center",
        )
        axs[0].set_xlim((-5.5, 5.5))

        self._plot_partial_histogram(
            axs[1], self.is_covered, self.covered_color, "covered"
        )
        self._plot_partial_histogram(
            axs[1], ~self.is_covered, self.not_covered_color, "not covered"
        )

        self._plot_hist_or_bar(
            axs[2],
            y_pred_width,
            alpha=1,
            color=self.prediction_color,
            label="prediction width",
        )
        axs[2].set_xlim((0, 5))

        for ax in axs:
            ax.legend()
            ax.set_ylim((0, MAX_DENSITY))
        plt.suptitle(f"Marginals. Model: {self.model_name} / Stage: {self.stage}")
        plt.tight_layout()
        self._save_fig(fig, "marginals")

    def target_vs_predicted(self):
        """Show prediction intervals against sorted target values."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = np.where(self.is_covered, self.covered_color, self.not_covered_color)
        ax.vlines(
            self.y,
            self.y_pred[:, 0],
            self.y_pred[:, 1],
            colors=colors,
            alpha=0.1,
            linewidth=0.5,
        )
        ax.scatter(self.y, self.y, color=self.target_color, s=1)
        ax.set_xlim((-5.5, 5.5))
        ax.set_ylim((-5.5, 5.5))

        plt.suptitle(
            f"""Prediction vs target. Model: {self.model_name} / Stage: {self.stage}. 
            {self.percent_covered}% covered; avg width: {self.avg_width:.2f}"""
        )
        self._save_fig(fig, "target-vs-predicted")

    def _plot_hist_or_bar(self, ax, data, alpha, color, label):
        """Like ax.hist, but data that are (close to) point masses is plotted as a single bar."""
        if data.std() > 1e-5:
            ax.hist(
                data,
                bins=self.bins,
                density=True,
                alpha=alpha,
                color=color,
                label=label,
            )
        else:
            ax.bar(data.mean(), height=MAX_DENSITY, width=0.1, color=color, label=label)

    def _plot_partial_histogram(self, ax, mask, color, label):
        """Like ax.hist, but normalization is done with respect to the total length.

        Using ax.hist directly on a masked array with density=True would normalize the
        result, independently of the fraction of examples that are masked. This function
        uses a workaround by filling the masked values with a dummy value, which is
        hidden away by xlim."""
        dummy_value = -5.7
        y_masked = self.y.copy()
        y_masked[~mask] = dummy_value
        ax.hist(
            y_masked,
            bins=self.bins,
            density=True,
            alpha=0.7,
            color=color,
            label=label,
        )
        ax.set_xlim((-5.5, 5.5))

    def _save_fig(self, fig, name):
        if not mlflow.active_run():
            return

        with tempfile.NamedTemporaryFile(
            mode="w+b", prefix=f"{name}_", suffix=".png", delete=True
        ) as tmpfile:
            file_path = tmpfile.name
            fig.savefig(file_path)
            mlflow.log_artifact(file_path)
