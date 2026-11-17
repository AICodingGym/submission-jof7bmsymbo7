"""Compare causal, within-breath input history with the fixed PA3 baseline."""

import json
import platform
from datetime import datetime, timezone
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

from baseline import FEATURES as BASE_FEATURES
from baseline import ROOT, file_hash, record_experiment


HISTORY_FEATURES = [
    "u_in_lag_1", "u_in_lag_2", "u_in_lag_5", "u_in_cumsum",
    "u_in_diff", "time_delta", "u_in_area",
]
FEATURES = BASE_FEATURES + HISTORY_FEATURES


def add_history(inputs):
    """Use only current/past controls, returning rows in their original order."""
    frame = inputs[["id", "breath_id"] + BASE_FEATURES].copy()
    if not frame.index.is_unique or not frame["id"].is_unique:
        raise ValueError("Row indices and IDs must be unique for alignment.")
    frame = frame.sort_values(["breath_id", "time_step", "id"], kind="stable")
    groups = frame.groupby("breath_id", sort=False)
    for lag in [1, 2, 5]:
        frame[f"u_in_lag_{lag}"] = groups["u_in"].shift(lag).fillna(0)
    frame["u_in_cumsum"] = groups["u_in"].cumsum()
    frame["u_in_diff"] = groups["u_in"].diff().fillna(0)
    frame["time_delta"] = groups["time_step"].diff().fillna(0)
    if frame["time_delta"].lt(0).any():
        raise ValueError("Elapsed times must be nonnegative within each breath.")
    # Right-endpoint integration, as requested: sum(current u_in * elapsed time).
    increments = frame["u_in"].astype("float64") * frame["time_delta"]
    frame["u_in_area"] = increments.groupby(frame["breath_id"], sort=False).cumsum()
    frame[HISTORY_FEATURES] = frame[HISTORY_FEATURES].astype("float32")
    frame = frame.reindex(inputs.index)
    if not frame["id"].equals(inputs["id"]):
        raise ValueError("Feature engineering changed row-to-ID alignment.")
    if not np.isfinite(frame[FEATURES].to_numpy()).all():
        raise ValueError("Model features contain missing or nonfinite values.")
    return frame


def main():
    started = perf_counter()
    baseline_path = ROOT / "baseline.py"
    split_path = ROOT / "validation_split.csv"
    train_path = ROOT / "data/train.csv"
    baseline_hash = file_hash(baseline_path)
    split_hash = file_hash(split_path)
    records = pd.read_csv(ROOT / "experiments.csv")
    baseline = records.loc[records["model_name"].eq("LightGBM baseline")].iloc[0]
    if split_hash != baseline["split_sha256"]:
        raise ValueError("Split differs from the original baseline.")
    train_hash = file_hash(train_path)
    if train_hash != baseline["train_sha256"]:
        raise ValueError("Dataset differs from the original baseline.")
    parameters = json.loads(baseline["parameters"])
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train = pd.read_csv(train_path, dtype=dtype)
    split = pd.read_csv(split_path)
    if split["breath_id"].duplicated().any():
        raise ValueError("Saved split contains duplicate breath IDs.")
    if set(split["split"]) != {"train", "validation"}:
        raise ValueError("Saved split contains unexpected partition labels.")
    if set(split["breath_id"]) != set(train["breath_id"]):
        raise ValueError("Saved split does not cover exactly the dataset breaths.")
    partitions = train["breath_id"].map(split.set_index("breath_id")["split"])
    train_mask = partitions.eq("train")
    valid_mask = partitions.eq("validation")
    # Explicitly remove pressure before engineering any model inputs.
    inputs = add_history(train.drop(columns="pressure"))
    x_train = inputs.loc[train_mask, FEATURES]
    x_valid = inputs.loc[valid_mask, FEATURES]
    print(f"Features: {FEATURES}")
    print(f"Original baseline parameters: {json.dumps(parameters, sort_keys=True)}")
    print(f"Training rows: {train_mask.sum():,}; validation rows: {valid_mask.sum():,}",
          flush=True)
    model = lgb.LGBMRegressor(**parameters)
    fit_started = perf_counter()
    model.fit(x_train, train.loc[train_mask, "pressure"])
    training_seconds = perf_counter() - fit_started
    predictions = pd.Series(model.predict(x_valid), index=x_valid.index)
    validation = train.loc[valid_mask, [
        "id", "breath_id", "R", "C", "time_step", "u_out", "pressure"
    ]].copy()
    validation["prediction"] = predictions.reindex(validation.index)
    if validation["prediction"].isna().any():
        raise ValueError("Validation predictions are missing or misaligned.")
    scored = validation.loc[validation["u_out"].eq(0)].copy()
    if len(scored) != int(baseline["validation_inspiration_rows"]):
        raise ValueError("Scored rows differ from the baseline.")
    mae = mean_absolute_error(scored["pressure"], scored["prediction"])
    improvement = float(baseline["mae"]) - mae
    reduction = 100 * improvement / float(baseline["mae"])
    elapsed_seconds = perf_counter() - started
    importance = pd.DataFrame({
        "feature": FEATURES,
        "gain": model.booster_.feature_importance(importance_type="gain"),
        "split_count": model.booster_.feature_importance(importance_type="split"),
    }).sort_values("gain", ascending=False)
    importance["gain_fraction"] = importance["gain"] / importance["gain"].sum()
    importance.to_csv(ROOT / "experiment2_feature_importance.csv", index=False)
    # Persist this run's predictions for later analysis without another model fit.
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    validation.to_csv(output_dir / "experiment2_validation_predictions.csv", index=False)
    if file_hash(split_path) != split_hash or file_hash(baseline_path) != baseline_hash:
        raise ValueError("Original baseline or validation split changed during this run.")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "LightGBM Experiment 2 - time-series features",
        "features": json.dumps(FEATURES),
        "mae": mae,
        "training_runtime_seconds": training_seconds,
        "total_runtime_seconds": elapsed_seconds,
        "observations": (
            "Same baseline parameters and split. Trained on all phases, scored on u_out == 0. "
            "Within-breath causal features, original row order restored. "
            "Missing lags filled with zero, initial diff and time_delta zero. "
            "u_in_area=sum(current u_in * time_delta), first area zero. "
            f"MAE reduction vs baseline={improvement:.8f} ({reduction:.2f}%). "
            "Gain importance is descriptive, feature contributions require ablation."
        ),
        "train_breaths": train.loc[train_mask, "breath_id"].nunique(),
        "validation_breaths": validation["breath_id"].nunique(),
        "validation_inspiration_rows": len(scored),
        "random_state": parameters["random_state"],
        "split_file": split_path.name,
        "split_sha256": split_hash,
        "train_sha256": train_hash,
        "test_sha256": baseline["test_sha256"],
        "parameters": json.dumps(parameters, sort_keys=True),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
    }
    record_experiment(record)
    print(f"\nExperiment 2 inspiratory MAE: {mae:.8f}")
    print(f"Baseline inspiratory MAE: {baseline['mae']:.8f}")
    print(f"MAE reduction: {improvement:.8f} ({reduction:.2f}%)")
    print(f"Training runtime: {training_seconds:.3f} seconds")
    print(f"Total runtime through evaluation: {elapsed_seconds:.3f} seconds")
    print(f"\nFeature importance\n{importance.to_string(index=False)}", flush=True)


if __name__ == "__main__":
    main()
