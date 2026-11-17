"""Test inspiration-only training with Experiment 2 features and parameters."""

import json
import platform
from datetime import datetime, timezone
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error

from baseline import ROOT, file_hash, record_experiment
from experiment2 import FEATURES, add_history


def main():
    started = perf_counter()
    protected_paths = [ROOT / name for name in [
        "baseline.py", "experiment2.py", "validation_split.csv"
    ]]
    original_hashes = {path: file_hash(path) for path in protected_paths}
    records = pd.read_csv(ROOT / "experiments.csv")
    experiment2 = records.loc[records["model_name"].eq(
        "LightGBM Experiment 2 - time-series features"
    )].iloc[-1]
    split_path = ROOT / "validation_split.csv"
    train_path = ROOT / "data/train.csv"
    train_hash = file_hash(train_path)
    if original_hashes[split_path] != experiment2["split_sha256"]:
        raise ValueError("Validation split differs from Experiment 2.")
    if train_hash != experiment2["train_sha256"]:
        raise ValueError("Training dataset differs from Experiment 2.")
    if FEATURES != json.loads(experiment2["features"]):
        raise ValueError("Feature list differs from Experiment 2.")
    parameters = json.loads(experiment2["parameters"])
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train = pd.read_csv(train_path, dtype=dtype)
    split = pd.read_csv(split_path)
    if split["breath_id"].duplicated().any():
        raise ValueError("Saved split contains duplicate breaths.")
    if set(split["split"]) != {"train", "validation"}:
        raise ValueError("Saved split contains unexpected partition labels.")
    if set(split["breath_id"]) != set(train["breath_id"]):
        raise ValueError("Saved split and dataset breaths differ.")

    # Engineer history on complete breaths before applying either phase filter.
    # The feature function receives no pressure column and restores row order.
    inputs = add_history(train.drop(columns="pressure"))
    partitions = train["breath_id"].map(split.set_index("breath_id")["split"])
    inspiration = train["u_out"].eq(0)
    train_mask = partitions.eq("train") & inspiration
    valid_mask = partitions.eq("validation") & inspiration
    train_breaths = train.loc[train_mask, "breath_id"].nunique()
    valid_breaths = train.loc[valid_mask, "breath_id"].nunique()
    if train_breaths != int(experiment2["train_breaths"]):
        raise ValueError("Training breath count differs from Experiment 2.")
    if valid_breaths != int(experiment2["validation_breaths"]):
        raise ValueError("Validation breath count differs from Experiment 2.")
    if int(valid_mask.sum()) != int(experiment2["validation_inspiration_rows"]):
        raise ValueError("Scored validation rows differ from Experiment 2.")
    if (train_mask & valid_mask).any():
        raise ValueError("Training and validation rows overlap.")
    x_train = inputs.loc[train_mask, FEATURES]
    x_valid = inputs.loc[valid_mask, FEATURES]
    if not x_train["u_out"].eq(0).all() or not x_valid["u_out"].eq(0).all():
        raise ValueError("Training and evaluation must contain only inspiration.")
    print(f"Features: {FEATURES}")
    print(f"Experiment 2 parameters: {json.dumps(parameters, sort_keys=True)}")
    print(f"Training: {train_breaths:,} breaths, {len(x_train):,} inspiration rows")
    print(f"Validation: {valid_breaths:,} breaths, {len(x_valid):,} inspiration rows",
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
    if not validation["id"].equals(inputs.loc[valid_mask, "id"]):
        raise ValueError("Validation prediction IDs are misaligned.")
    if not np.isfinite(validation["prediction"]).all():
        raise ValueError("Validation predictions contain nonfinite values.")
    mae = mean_absolute_error(validation["pressure"], validation["prediction"])
    improvement = float(experiment2["mae"]) - mae
    reduction = 100 * improvement / float(experiment2["mae"])
    elapsed_seconds = perf_counter() - started
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    validation.to_csv(output_dir / "experiment3_validation_predictions.csv", index=False)
    for path, original_hash in original_hashes.items():
        if file_hash(path) != original_hash:
            raise ValueError(f"Protected file changed during training: {path.name}")
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "LightGBM Experiment 3 - inspiration-only training",
        "features": json.dumps(FEATURES),
        "mae": mae,
        "training_runtime_seconds": training_seconds,
        "total_runtime_seconds": elapsed_seconds,
        "observations": (
            "Experiment 2 features and parameters, exact saved breath split. "
            "History engineered from complete breaths before phase filtering. "
            f"Training only u_out == 0 ({len(x_train)} rows); validation only u_out == 0. "
            "Target excluded from feature engineering; original row-to-ID alignment retained. "
            f"MAE reduction vs Experiment 2={improvement:.8f} ({reduction:.2f}%). "
            "Training runtime measures fit only."
        ),
        "train_breaths": train_breaths,
        "validation_breaths": valid_breaths,
        "validation_inspiration_rows": len(validation),
        "random_state": parameters["random_state"],
        "split_file": split_path.name,
        "split_sha256": original_hashes[split_path],
        "train_sha256": train_hash,
        "test_sha256": experiment2["test_sha256"],
        "parameters": json.dumps(parameters, sort_keys=True),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
    }
    record_experiment(record)
    updated = pd.read_csv(ROOT / "experiments.csv")
    pd.testing.assert_frame_equal(updated.iloc[:len(records)].reset_index(drop=True), records)
    print(f"\nExperiment 3 inspiratory MAE: {mae:.8f}")
    print(f"Experiment 2 inspiratory MAE: {experiment2['mae']:.8f}")
    print(f"MAE reduction: {improvement:.8f} ({reduction:.2f}%)")
    print(f"Training runtime: {training_seconds:.3f} seconds")
    print(f"Experiment 2 latest training runtime: "
          f"{experiment2['training_runtime_seconds']:.3f} seconds")
    print(f"Total runtime through evaluation: {elapsed_seconds:.3f} seconds")
    print("Verified protected file hashes and preserved previous experiment records.",
          flush=True)


if __name__ == "__main__":
    main()
