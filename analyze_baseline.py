"""Reproduce the recorded baseline and summarize validation inspiration errors."""

import importlib
import json

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import FEATURES, ROOT, file_hash


def summarize(frame, columns, name):
    result = frame.groupby(columns, observed=True).agg(
        rows=("absolute_error", "size"),
        mae=("absolute_error", "mean"),
        median_reference_mae=("reference_error", "mean"),
        bias=("error", "mean"),
        p90_absolute_error=("absolute_error", lambda values: values.quantile(0.9)),
        mean_pressure=("pressure", "mean"),
    ).reset_index()
    result.insert(0, "group", result[columns].astype(str).agg(", ".join, axis=1))
    result.insert(0, "grouping", name)
    result = result.drop(columns=columns)
    print(f"\n{name}\n{result.to_string(index=False)}", flush=True)
    return result


def main():
    split_path = ROOT / "validation_split.csv"
    experiments_path = ROOT / "experiments.csv"
    original_split_hash = file_hash(split_path)
    original_experiments_hash = file_hash(experiments_path)
    records = pd.read_csv(experiments_path)
    record = records.loc[records["model_name"].eq("LightGBM baseline")].iloc[0]
    if original_split_hash != record["split_sha256"]:
        raise ValueError("Saved split differs from the recorded baseline.")
    if file_hash(ROOT / "data/train.csv") != record["train_sha256"]:
        raise ValueError("Training dataset differs from the recorded baseline.")
    if json.loads(record["features"]) != FEATURES:
        raise ValueError("Feature list differs from the recorded baseline.")
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train = pd.read_csv(ROOT / "data/train.csv", dtype=dtype)
    split = pd.read_csv(split_path)
    if split["breath_id"].duplicated().any():
        raise ValueError("Split contains duplicate breaths.")
    if set(split["split"]) != {"train", "validation"}:
        raise ValueError("Split contains unexpected partition labels.")
    if set(split["breath_id"]) != set(train["breath_id"]):
        raise ValueError("Split and training dataset have different breaths.")
    train_ids = split.loc[split["split"].eq("train"), "breath_id"]
    valid_ids = split.loc[split["split"].eq("validation"), "breath_id"]
    train_mask = train["breath_id"].isin(train_ids)
    valid_mask = train["breath_id"].isin(valid_ids)
    model = lgb.LGBMRegressor(**json.loads(record["parameters"]))
    print("Reproducing the original baseline for diagnostic predictions.", flush=True)
    model.fit(train.loc[train_mask, FEATURES], train.loc[train_mask, "pressure"])
    scored = train.loc[valid_mask & train["u_out"].eq(0)].copy()
    scored["prediction"] = model.predict(scored[FEATURES])
    scored["error"] = scored["prediction"] - scored["pressure"]
    scored["absolute_error"] = scored["error"].abs()
    median = train.loc[train_mask & train["u_out"].eq(0), "pressure"].median()
    scored["reference_error"] = (scored["pressure"] - median).abs()
    mae = scored["absolute_error"].mean()
    if not np.isclose(mae, record["mae"], rtol=0, atol=1e-10):
        raise ValueError(f"Reproduced MAE {mae} differs from recorded MAE.")
    print(f"Recorded training runtime: {record['training_runtime_seconds']:.6f}s")
    print(f"Reproduced inspiration MAE: {mae:.10f}")
    print(f"Median reference value: {median:.8f}")
    print(f"Median reference MAE: {scored['reference_error'].mean():.10f}")
    print(f"Inspiration time range: {scored['time_step'].min()} to "
          f"{scored['time_step'].max()} seconds")
    scored["time_bin"] = pd.cut(
        scored["time_step"], [0, 0.25, 0.5, 0.75, 1, 1.25, 1.5, np.inf],
        right=False,
    )
    # Relative position is diagnostic only. It is never supplied to the model.
    scored = scored.sort_values(["breath_id", "time_step", "id"])
    breaths = scored.groupby("breath_id", sort=False)
    progress = breaths.cumcount() / (breaths["id"].transform("size") - 1).clip(lower=1)
    scored["inspiration_position"] = pd.cut(
        progress, [0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    tables = [
        summarize(scored, ["R"], "R"),
        summarize(scored, ["C"], "C"),
        summarize(scored, ["R", "C"], "R,C"),
        summarize(scored, ["time_bin"], "time_step_seconds"),
        summarize(scored, ["inspiration_position"], "inspiration_position"),
    ]
    if file_hash(split_path) != original_split_hash:
        raise ValueError("Split changed during analysis.")
    if file_hash(experiments_path) != original_experiments_hash:
        raise ValueError("Experiment records changed during analysis.")
    pd.concat(tables, ignore_index=True).to_csv(
        ROOT / "baseline_error_analysis.csv", index=False
    )
    print("\nVerified original split and experiment record hashes.", flush=True)


if __name__ == "__main__":
    main()
