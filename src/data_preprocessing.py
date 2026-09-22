"""Data loading, validation, and leakage-aware feature construction."""

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RANDOM_SEED = 42
TARGET = "Energy"
REQUIRED_COLUMNS = {
    "ECdata.csv": {"Time", "BS", TARGET},
    "CLdata.csv": {"Time", "BS", "CellName", "load"},
    "BSinfo.csv": {"BS", "CellName", "RUType", "Mode"},
}


def _require_columns(frame: pd.DataFrame, required: Iterable[str], name: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing required columns: {missing}")


def load_raw_data(data_dir: str | Path = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the energy, cell-level, and base-station tables with validation."""
    data_path = Path(data_dir)
    frames = {}
    for filename, required in REQUIRED_COLUMNS.items():
        path = data_path / filename
        if not path.exists():
            path = Path(filename)
        if not path.exists():
            raise FileNotFoundError(f"Could not find {filename} in {data_path} or the project root")
        frame = pd.read_csv(path)
        _require_columns(frame, required, filename)
        frames[filename] = frame

    energy = frames["ECdata.csv"].copy()
    cell = frames["CLdata.csv"].copy()
    base_station = frames["BSinfo.csv"].copy()
    energy["Time"] = pd.to_datetime(energy["Time"], errors="raise")
    cell["Time"] = pd.to_datetime(cell["Time"], errors="raise")
    return energy, cell, base_station


def build_features(
    energy: pd.DataFrame,
    cell: pd.DataFrame,
    base_station: pd.DataFrame,
    include_target: bool = True,
) -> pd.DataFrame:
    """Join source tables and create features available at prediction time.

    Target lags are deliberately excluded: future submission rows do not contain
    observed Energy values, so using them would make validation misleading.
    """
    cell = cell.copy()
    base_station = base_station.copy()
    energy = energy.copy()
    cell = cell[cell["CellName"].eq("Cell0")].copy()
    if cell.empty:
        raise ValueError("CLdata.csv contains no Cell0 rows")

    frame = energy.merge(cell, on=["Time", "BS"], how="left", validate="one_to_one")
    frame = frame.merge(
        base_station,
        on=["BS", "CellName"],
        how="left",
        validate="many_to_one",
        suffixes=("", "_bs"),
    )
    if frame[["load", "RUType", "Mode"]].isna().any().any():
        missing = frame[["load", "RUType", "Mode"]].isna().sum()
        raise ValueError(f"Feature join produced missing required values: {missing.to_dict()}")

    frame = frame.sort_values(["BS", "Time"]).reset_index(drop=True)
    frame["hour"] = frame["Time"].dt.hour
    frame["day_of_week"] = frame["Time"].dt.dayofweek
    frame["day_of_month"] = frame["Time"].dt.day
    frame["is_weekend"] = frame["day_of_week"].isin([5, 6]).astype(int)
    frame["hour_sin"] = np.sin(2 * np.pi * frame["hour"] / 24)
    frame["hour_cos"] = np.cos(2 * np.pi * frame["hour"] / 24)

    for lag in (1, 2, 3):
        for column in ["load", "ESMode1", "ESMode2", "ESMode3", "ESMode4", "ESMode5", "ESMode6"]:
            if column in frame:
                frame[f"{column}_lag_{lag}"] = frame.groupby("BS")[column].shift(lag)

    if not include_target and TARGET in frame:
        frame = frame.drop(columns=[TARGET])
    return frame


def split_by_time(frame: pd.DataFrame, validation_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return chronological train and validation partitions."""
    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    unique_times = np.sort(frame["Time"].dropna().unique())
    cutoff_index = max(1, int(len(unique_times) * (1 - validation_fraction))) - 1
    cutoff = unique_times[cutoff_index]
    train = frame[frame["Time"] <= cutoff].copy()
    validation = frame[frame["Time"] > cutoff].copy()
    if train.empty or validation.empty:
        raise ValueError("Chronological split produced an empty partition")
    return train, validation


def feature_columns(frame: pd.DataFrame) -> tuple[list[str], list[str], list[str]]:
    """Identify model features and categorical/numeric subsets."""
    excluded = {TARGET, "Time", "CellName"}
    features = [column for column in frame.columns if column not in excluded]
    categorical = [column for column in ["BS", "RUType", "Mode", "Frequency", "Bandwidth", "Antennas"] if column in features]
    numeric = [column for column in features if column not in categorical]
    return features, categorical, numeric
