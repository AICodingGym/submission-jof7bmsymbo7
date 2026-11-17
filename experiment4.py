"""Reproduce Experiment 3 and compare lung interactions and longer input history.

Run all variants with python experiment4.py, or select one with --groups A.
All model settings and variant feature lists are in experiment4_configs.json.
"""

import argparse
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
from experiment2 import FEATURES as ORIGINAL_FEATURES
from experiment2 import add_history


def engineer_features(inputs, config):
    """Create causal features from complete breaths and restore original order."""
    frame = add_history(inputs)
    frame = frame.sort_values(["breath_id", "time_step", "id"], kind="stable")
    if frame["C"].le(0).any():
        raise ValueError("Compliance must be positive for interaction ratios.")
    resistance = frame["R"].astype("float32")
    compliance = frame["C"].astype("float32")
    frame["R_times_C"] = resistance * compliance
    frame["R_div_C"] = resistance / compliance
    frame["u_in_div_C"] = frame["u_in"] / compliance
    frame["u_in_times_R"] = frame["u_in"] * resistance
    controls = frame.groupby("breath_id", sort=False)["u_in"]
    conventions = config["history_conventions"]
    for lag in [10, 15, 20]:
        frame[f"u_in_lag_{lag}"] = controls.shift(lag).fillna(
            conventions["missing_lags"]
        )
    for window in [5, 10]:
        means = controls.rolling(
            window, min_periods=conventions["rolling_min_periods"]
        ).mean()
        frame[f"u_in_rolling_mean_{window}"] = means.reset_index(level=0, drop=True)
    deviations = controls.rolling(
        5, min_periods=conventions["rolling_min_periods"]
    ).std(ddof=conventions["rolling_std_ddof"])
    frame["u_in_rolling_std_5"] = deviations.reset_index(level=0, drop=True)
    added_features = config["groups"]["C"]
    frame[added_features] = frame[added_features].astype(conventions["feature_dtype"])
    frame = frame.reindex(inputs.index)
    if not frame["id"].equals(inputs["id"]):
        raise ValueError("Feature engineering changed ID alignment.")
    if not np.isfinite(frame[ORIGINAL_FEATURES + added_features].to_numpy()).all():
        raise ValueError("Model inputs contain nonfinite values.")
    return frame


def summarize_errors(validation, variant):
    scored = validation.sort_values(["breath_id", "time_step", "id"]).copy()
    scored["absolute_error"] = (scored["prediction"] - scored["pressure"]).abs()
    scored["bias"] = scored["prediction"] - scored["pressure"]
    scored["time_bin"] = pd.cut(
        scored["time_step"], [0, 0.25, 0.5, 0.75, 1, np.inf], right=False
    )
    breaths = scored.groupby("breath_id", sort=False)
    position = breaths.cumcount() / (breaths["id"].transform("size") - 1).clip(lower=1)
    scored["position"] = pd.cut(
        position, [0, 0.2, 0.4, 0.6, 0.8, 1], include_lowest=True,
        labels=["0-20%", "20-40%", "40-60%", "60-80%", "80-100%"],
    )
    summaries = []
    for name, columns in [
        ("R,C", ["R", "C"]), ("time_step_seconds", ["time_bin"]),
        ("inspiration_position", ["position"]),
        ("R,C,time_step_seconds", ["R", "C", "time_bin"]),
    ]:
        table = scored.groupby(columns, observed=True).agg(
            rows=("absolute_error", "size"), mae=("absolute_error", "mean"),
            bias=("bias", "mean"), maximum_error=("absolute_error", "max"),
            p90_error=("absolute_error", lambda values: values.quantile(0.9)),
        ).reset_index()
        table["group"] = table[columns].astype(str).agg(", ".join, axis=1)
        table["grouping"] = name
        table["variant"] = variant
        summaries.append(table[[
            "variant", "grouping", "group", "rows", "mae", "bias",
            "maximum_error", "p90_error",
        ]])
    summary = pd.concat(summaries, ignore_index=True)
    summary.to_csv(ROOT / f"experiment4_{variant}_error_analysis.csv", index=False)
    worst = scored.nlargest(20, "absolute_error")
    worst.to_csv(ROOT / "outputs" / f"experiment4_{variant}_largest_errors.csv", index=False)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--groups", nargs="+", choices=["reference", "A", "B", "C"],
        default=["reference", "A", "B", "C"],
    )
    args = parser.parse_args()
    started = perf_counter()
    config_path = ROOT / "experiment4_configs.json"
    config = json.loads(config_path.read_text(encoding="utf-8"))
    paths = [ROOT / name for name in [
        "baseline.py", "analyze_baseline.py", "experiment2.py", "experiment3.py",
        "validation_split.csv",
    ]]
    protected = {path: file_hash(path) for path in paths}
    records = pd.read_csv(ROOT / "experiments.csv")
    reference = records.loc[records["model_name"].eq(config["reference_model"])].iloc[-1]
    split_path = ROOT / "validation_split.csv"
    if protected[split_path] != reference["split_sha256"] or (
        protected[split_path] != config["split_sha256"]
    ):
        raise ValueError("Saved validation split differs from Experiment 3.")
    parameters = config["parameters"]
    if parameters != json.loads(reference["parameters"]):
        raise ValueError("Configured parameters differ from Experiment 3.")
    if ORIGINAL_FEATURES != json.loads(reference["features"]):
        raise ValueError("Original features differ from Experiment 3.")
    if config["groups"]["C"] != config["groups"]["A"] + config["groups"]["B"]:
        raise ValueError("Group C must contain exactly Groups A and B.")
    train_path = ROOT / "data/train.csv"
    train_hash = file_hash(train_path)
    if train_hash != reference["train_sha256"]:
        raise ValueError("Dataset differs from Experiment 3.")
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train = pd.read_csv(train_path, dtype=dtype)
    split = pd.read_csv(split_path)
    if split["breath_id"].duplicated().any() or set(split["split"]) != {"train", "validation"}:
        raise ValueError("Invalid saved split.")
    if set(split["breath_id"]) != set(train["breath_id"]):
        raise ValueError("Split does not cover the dataset breaths.")
    # No pressure reaches feature engineering. Complete breaths include both phases.
    inputs = engineer_features(train.drop(columns="pressure"), config)
    partitions = train["breath_id"].map(split.set_index("breath_id")["split"])
    train_mask = partitions.eq("train") & train["u_out"].eq(0)
    valid_mask = partitions.eq("validation") & train["u_out"].eq(0)
    train_breaths = train.loc[train_mask, "breath_id"].nunique()
    valid_breaths = train.loc[valid_mask, "breath_id"].nunique()
    if (train_breaths, valid_breaths, int(valid_mask.sum())) != (
        int(reference["train_breaths"]), int(reference["validation_breaths"]),
        int(reference["validation_inspiration_rows"]),
    ):
        raise ValueError("Breath counts or scored rows differ from Experiment 3.")
    output_dir = ROOT / "outputs"
    output_dir.mkdir(exist_ok=True)
    print(f"Prepared {train_mask.sum():,} training and {valid_mask.sum():,} "
          "validation inspiration rows.", flush=True)
    preparation_seconds = perf_counter() - started
    results = []
    for variant in args.groups:
        variant_started = perf_counter()
        features = ORIGINAL_FEATURES + config["groups"][variant]
        x_train = inputs.loc[train_mask, features]
        x_valid = inputs.loc[valid_mask, features]
        model = lgb.LGBMRegressor(**parameters)
        print(f"Training {variant} with {len(features)} features.", flush=True)
        fit_started = perf_counter()
        model.fit(x_train, train.loc[train_mask, "pressure"])
        training_seconds = perf_counter() - fit_started
        predictions = pd.Series(model.predict(x_valid), index=x_valid.index)
        validation = train.loc[valid_mask, [
            "id", "breath_id", "R", "C", "time_step", "u_out", "pressure"
        ]].copy()
        validation["prediction"] = predictions.reindex(validation.index)
        if not validation["id"].equals(inputs.loc[valid_mask, "id"]):
            raise ValueError("Prediction-to-ID alignment differs.")
        if not np.isfinite(validation["prediction"]).all():
            raise ValueError("Predictions contain nonfinite values.")
        mae = mean_absolute_error(validation["pressure"], validation["prediction"])
        if variant == "reference" and not np.isclose(
            mae, reference["mae"], rtol=0, atol=1e-10
        ):
            raise ValueError("Reproduced Experiment 3 score differs from its record.")
        elapsed_seconds = preparation_seconds + perf_counter() - variant_started
        validation.to_csv(
            output_dir / f"experiment4_{variant}_validation_predictions.csv", index=False
        )
        summarize_errors(validation, variant)
        importance = pd.DataFrame({
            "feature": features,
            "gain": model.booster_.feature_importance(importance_type="gain"),
        }).sort_values("gain", ascending=False)
        importance["gain_fraction"] = importance["gain"] / importance["gain"].sum()
        importance.to_csv(ROOT / f"experiment4_{variant}_feature_importance.csv", index=False)
        improvement = float(reference["mae"]) - mae
        record = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "model_name": f"LightGBM Experiment 4 - {variant}",
            "features": json.dumps(features), "mae": mae,
            "training_runtime_seconds": training_seconds,
            "total_runtime_seconds": elapsed_seconds,
            "observations": (
                f"Group {variant}. Experiment 3 parameters and inspiration-only training. "
                "Complete-breath causal features, pressure excluded, original ID alignment. "
                "Unavailable lags=0. Rolling windows include current row, min_periods=1, "
                "std ddof=0. Feature dtype=float32. "
                f"MAE reduction vs Experiment 3={improvement:.8f} "
                f"({100 * improvement / float(reference['mae']):.2f}%). "
                "Total runtime includes shared preparation plus this variant through evaluation. "
                f"Reproduce with experiment4.py --groups {variant}. "
                f"Config SHA256={file_hash(config_path)}."
            ),
            "train_breaths": train_breaths, "validation_breaths": valid_breaths,
            "validation_inspiration_rows": len(validation),
            "random_state": parameters["random_state"], "split_file": split_path.name,
            "split_sha256": protected[split_path], "train_sha256": train_hash,
            "test_sha256": reference["test_sha256"],
            "parameters": json.dumps(parameters, sort_keys=True),
            "python_version": platform.python_version(), "pandas_version": pd.__version__,
            "numpy_version": np.__version__, "sklearn_version": sklearn.__version__,
            "lightgbm_version": lgb.__version__,
        }
        for path, digest in protected.items():
            if file_hash(path) != digest:
                raise ValueError(f"Existing file changed during experiment: {path.name}")
        record_experiment(record)
        results.append({
            "variant": variant, "features": len(features), "mae": mae,
            "training_runtime_seconds": training_seconds,
            "mae_reduction_percent": 100 * improvement / float(reference["mae"]),
        })
        print(f"{variant}: MAE={mae:.8f}, fit={training_seconds:.3f}s", flush=True)
    updated = pd.read_csv(ROOT / "experiments.csv")
    pd.testing.assert_frame_equal(updated.iloc[:len(records)].reset_index(drop=True), records)
    comparison = pd.DataFrame(results)
    comparison.to_csv(ROOT / "experiment4_comparison.csv", index=False)
    print(comparison.to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
