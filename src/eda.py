"""Create compact exploratory data analysis artifacts."""

from pathlib import Path
import sys

import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import TARGET, build_features, load_raw_data


def main(data_dir: str = ".", output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    energy, cell, base_station = load_raw_data(data_dir)
    data = build_features(energy, cell, base_station)
    data.describe(include="all").to_csv(output_path / "metrics" / "data_summary.csv")
    data.isna().sum().sort_values(ascending=False).to_csv(output_path / "metrics" / "missing_values.csv", header=["missing_count"])

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(data[TARGET], bins=40, kde=True, ax=axes[0])
    axes[0].set(title="Energy target distribution", xlabel="Energy")
    sns.histplot(data["load"], bins=40, kde=True, ax=axes[1])
    axes[1].set(title="Cell load distribution", xlabel="Load")
    figure.tight_layout()
    figure.savefig(output_path / "figures" / "target_and_load.png", dpi=150)
    plt.close(figure)
    print(f"Rows: {len(data):,}; columns: {data.shape[1]}")
    print(data[[TARGET, "load"]].describe().to_string())


if __name__ == "__main__":
    main()
