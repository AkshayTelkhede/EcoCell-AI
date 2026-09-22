"""Train and compare reproducible Energy regression models."""

from pathlib import Path
import json
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.data_preprocessing import TARGET, build_features, feature_columns, load_raw_data, split_by_time
from src.evaluate import regression_metrics, save_error_plots

RANDOM_SEED = 42


def _preprocessor(categorical: list[str], numeric: list[str], one_hot: bool = False) -> ColumnTransformer:
    encoder = OneHotEncoder(handle_unknown="ignore") if one_hot else OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    return ColumnTransformer(
        [
            ("numeric", SimpleImputer(strategy="median"), numeric),
            ("categorical", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", encoder)]), categorical),
        ],
        remainder="drop",
    )


def _models(categorical: list[str], numeric: list[str]) -> dict[str, Pipeline]:
    tree_preprocessor = _preprocessor(categorical, numeric)
    return {
        "mean_baseline": Pipeline([("preprocess", _preprocessor([], numeric)), ("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([
            ("preprocess", _preprocessor(categorical, numeric, one_hot=True)),
            ("scale", StandardScaler(with_mean=False)),
            ("model", Ridge(alpha=10.0)),
        ]),
        "extra_trees": Pipeline([
            ("preprocess", tree_preprocessor),
            ("model", ExtraTreesRegressor(n_estimators=120, max_depth=28, min_samples_leaf=2, random_state=RANDOM_SEED, n_jobs=-1)),
        ]),
        "hist_gradient_boosting": Pipeline([
            ("preprocess", _preprocessor(categorical, numeric)),
            ("model", HistGradientBoostingRegressor(max_iter=180, learning_rate=0.08, max_leaf_nodes=31, l2_regularization=0.1, random_state=RANDOM_SEED)),
        ]),
    }


def _tune_hist_gradient_boosting(train_x: pd.DataFrame, train_y: pd.Series, categorical: list[str], numeric: list[str]) -> tuple[Pipeline, dict[str, object]]:
    """Select a small set of HGB configurations on a later training slice."""
    inner_cut = train_x["Time"].quantile(0.8)
    inner_train = train_x[train_x["Time"] <= inner_cut]
    inner_valid = train_x[train_x["Time"] > inner_cut]
    candidates = [
        {"max_leaf_nodes": 15, "learning_rate": 0.06},
        {"max_leaf_nodes": 31, "learning_rate": 0.08},
        {"max_leaf_nodes": 63, "learning_rate": 0.05},
    ]
    best_score = float("inf")
    best_params = candidates[0]
    for params in candidates:
        model = Pipeline([
            ("preprocess", _preprocessor(categorical, numeric)),
            ("model", HistGradientBoostingRegressor(max_iter=180, l2_regularization=0.1, random_state=RANDOM_SEED, **params)),
        ])
        model.fit(inner_train, train_y.loc[inner_train.index])
        score = regression_metrics(train_y.loc[inner_valid.index], pd.Series(model.predict(inner_valid), index=inner_valid.index))["MAE"]
        if score < best_score:
            best_score, best_params = score, params
    tuned = Pipeline([
        ("preprocess", _preprocessor(categorical, numeric)),
        ("model", HistGradientBoostingRegressor(max_iter=180, l2_regularization=0.1, random_state=RANDOM_SEED, **best_params)),
    ])
    return tuned, {"candidates": candidates, "selected": best_params, "inner_validation_MAE": best_score}


def main(data_dir: str = ".", output_dir: str = "results") -> None:
    output_path = Path(output_dir)
    (output_path / "metrics").mkdir(parents=True, exist_ok=True)
    (output_path / "figures").mkdir(parents=True, exist_ok=True)
    energy, cell, base_station = load_raw_data(data_dir)
    data = build_features(energy, cell, base_station)
    train, validation = split_by_time(data)
    features, categorical, numeric = feature_columns(data)
    x_train, y_train = train[features], train[TARGET]
    x_valid, y_valid = validation[features], validation[TARGET]

    results = []
    predictions = {}
    models = _models(categorical, numeric)
    tuned_model, tuning = _tune_hist_gradient_boosting(train, y_train, categorical, numeric)
    models["hist_gradient_boosting_tuned"] = tuned_model
    for name, model in models.items():
        model.fit(x_train, y_train)
        prediction = pd.Series(model.predict(x_valid), index=validation.index)
        predictions[name] = prediction
        results.append({"Model": name, **regression_metrics(y_valid, prediction), "Training": f"{len(train):,} chronological rows; validation: {len(validation):,} later rows"})

    metrics = pd.DataFrame(results).sort_values("MAE")
    metrics.to_csv(output_path / "metrics" / "model_comparison.csv", index=False)
    (output_path / "metrics" / "tuning.json").write_text(json.dumps(tuning, indent=2), encoding="utf-8")
    best_name = metrics.iloc[0]["Model"]
    joblib.dump(models[best_name], output_path / "best_model.joblib")
    pd.DataFrame({"actual": y_valid, **predictions}).to_csv(output_path / "metrics" / "validation_predictions.csv", index=False)
    save_error_plots(y_valid, predictions[best_name], output_path / "figures")
    print(metrics.to_string(index=False))
    print(f"Best model: {best_name}")


if __name__ == "__main__":
    main()
