"""Regression metrics and evaluation artifact helpers."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def regression_metrics(y_true: pd.Series, predictions: pd.Series) -> dict[str, float]:
    """Calculate metrics that remain interpretable for Energy regression."""
    errors = y_true.to_numpy() - predictions.to_numpy()
    nonzero = y_true.to_numpy() != 0
    mape = abs(errors[nonzero] / y_true.to_numpy()[nonzero]).mean() * 100
    return {
        "MAE": float(mean_absolute_error(y_true, predictions)),
        "RMSE": float(mean_squared_error(y_true, predictions) ** 0.5),
        "MAPE_percent": float(mape),
        "R2": float(r2_score(y_true, predictions)),
    }


def save_error_plots(y_true: pd.Series, predictions: pd.Series, output_dir: str | Path) -> None:
    """Save residual and actual-versus-predicted plots."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    residuals = y_true - predictions

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.scatterplot(x=y_true, y=predictions, s=12, alpha=0.35, ax=axes[0])
    limits = [min(y_true.min(), predictions.min()), max(y_true.max(), predictions.max())]
    axes[0].plot(limits, limits, linestyle="--", color="black")
    axes[0].set(title="Actual vs predicted", xlabel="Actual Energy", ylabel="Predicted Energy")
    sns.histplot(residuals, bins=40, kde=True, ax=axes[1])
    axes[1].axvline(0, color="black", linestyle="--")
    axes[1].set(title="Residual distribution", xlabel="Actual - predicted")
    figure.tight_layout()
    figure.savefig(output_path / "validation_errors.png", dpi=150)
    plt.close(figure)
