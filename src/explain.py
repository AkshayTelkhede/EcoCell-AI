"""Generate SHAP global and local explanations for the saved tree model."""

from pathlib import Path
import sys

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import TARGET, build_features, feature_columns, load_raw_data


def main(data_dir: str = ".", model_path: str = "results/best_model.joblib", output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    energy, cell, base_station = load_raw_data(data_dir)
    data = build_features(energy, cell, base_station)
    features, _, _ = feature_columns(data)
    model = joblib.load(model_path)
    sample = data[features].sample(min(500, len(data)), random_state=42)
    transformed = model.named_steps["preprocess"].transform(sample)
    estimator = model.named_steps["model"]
    names = list(model.named_steps["preprocess"].get_feature_names_out())
    explainer = shap.TreeExplainer(estimator)
    shap_values = explainer.shap_values(transformed)
    importance = pd.DataFrame({"feature": names, "mean_abs_shap": abs(shap_values).mean(axis=0)})
    importance = importance.sort_values("mean_abs_shap", ascending=False)
    importance.to_csv(output_path / "metrics" / "shap_importance.csv", index=False)

    shap.summary_plot(shap_values, transformed, feature_names=names, show=False, max_display=15)
    plt.tight_layout()
    plt.savefig(output_path / "figures" / "shap_global_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    local_index = 0
    base_value = float(np.asarray(explainer.expected_value).reshape(-1)[0])
    shap.plots.waterfall(shap.Explanation(values=shap_values[local_index], base_values=base_value, data=transformed[local_index], feature_names=names), max_display=12, show=False)
    plt.tight_layout()
    plt.savefig(output_path / "figures" / "shap_local_waterfall.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(importance.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
