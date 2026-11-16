"""Run the PA3 LightGBM baseline using a persisted split of whole breaths."""

import hashlib
import json
import os
import platform
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split


ROOT = Path(__file__).resolve().parent
FEATURES = ["R", "C", "time_step", "u_in", "u_out"]
SEED = 42
SPLIT_PATH = ROOT / "validation_split.csv"
RESULTS_PATH = ROOT / "experiments.csv"
PARAMETERS = {
    "objective": "regression_l1",
    "n_estimators": 300,
    "learning_rate": 0.05,
    "num_leaves": 31,
    "random_state": SEED,
    "n_jobs": min(8, os.cpu_count() or 1),
    "deterministic": True,
    "force_col_wise": True,
    "verbosity": -1,
}


def file_hash(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def inspect_data(train, test):
    for name, frame in [("train.csv", train), ("test.csv", test)]:
        print(f"\n{name}", flush=True)
        print(f"Shape: {frame.shape}")
        print(f"Columns: {frame.columns.tolist()}")
        print(f"Types: {frame.dtypes.astype(str).to_dict()}")
        print(f"Missing values: {frame.isna().sum().to_dict()}")
        print(f"Unique breaths: {frame['breath_id'].nunique():,}")
        lengths = frame.groupby("breath_id", sort=False).size()
        print(f"Rows per breath: {lengths.value_counts().to_dict()}")
        print(f"u_out counts: {frame['u_out'].value_counts().to_dict()}")
        print(f"R values: {sorted(frame['R'].unique().tolist())}")
        print(f"C values: {sorted(frame['C'].unique().tolist())}")
    quantiles = [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
    print("\nTraining pressure distribution (cmH2O)")
    print(train["pressure"].describe(percentiles=quantiles).to_string())
    print("\nTraining inspiration pressure distribution (cmH2O)")
    print(train.loc[train["u_out"].eq(0), "pressure"].describe(
        percentiles=quantiles
    ).to_string(), flush=True)


def load_split(train):
    # Sorting makes the seed independent of the ordering of rows in train.csv.
    breaths = np.sort(train["breath_id"].unique())
    train_breaths, valid_breaths = train_test_split(
        breaths, test_size=0.2, random_state=SEED
    )
    expected = pd.DataFrame({"breath_id": breaths, "split": "train"})
    expected.loc[expected["breath_id"].isin(valid_breaths), "split"] = "validation"
    if SPLIT_PATH.exists():
        saved = pd.read_csv(SPLIT_PATH)
        if not saved.equals(expected):
            raise ValueError(
                "Saved validation_split.csv differs from the fixed split. "
                "Use the original dataset and split for comparable experiments."
            )
    else:
        # Exclusive creation protects an existing split from being overwritten.
        with SPLIT_PATH.open("x", newline="", encoding="utf-8") as handle:
            expected.to_csv(handle, index=False)
    train_mask = train["breath_id"].isin(train_breaths)
    valid_mask = train["breath_id"].isin(valid_breaths)
    if set(train_breaths) & set(valid_breaths):
        raise ValueError("Training and validation breaths overlap.")
    if not (train_mask ^ valid_mask).all():
        raise ValueError("Each row must belong to exactly one partition.")
    print(f"\nSplit saved in {SPLIT_PATH.name}")
    print(f"Training: {len(train_breaths):,} breaths, {train_mask.sum():,} rows")
    print(f"Validation: {len(valid_breaths):,} breaths, {valid_mask.sum():,} rows")
    print("Overlapping breath IDs: 0", flush=True)
    return train_mask, valid_mask


def record_experiment(record):
    new_record = pd.DataFrame([record])
    if RESULTS_PATH.exists():
        previous = pd.read_csv(RESULTS_PATH)
        new_record = pd.concat([previous, new_record], ignore_index=True)
    # Replace only after a complete CSV is written, preserving previous records.
    with tempfile.NamedTemporaryFile(
        mode="w", suffix=".csv", dir=ROOT, delete=False,
        newline="", encoding="utf-8"
    ) as handle:
        temporary_path = Path(handle.name)
        new_record.to_csv(handle, index=False)
    os.replace(temporary_path, RESULTS_PATH)


def main():
    started = perf_counter()
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train_path = ROOT / "data" / "train.csv"
    test_path = ROOT / "data" / "test.csv"
    train = pd.read_csv(train_path, dtype=dtype)
    test = pd.read_csv(test_path, dtype=dtype)
    sample_ids = pd.read_csv(ROOT / "data" / "sample_submission.csv", usecols=["id"])
    if not test["id"].equals(sample_ids["id"]):
        raise ValueError("Test IDs and sample submission IDs differ.")
    if set(train["breath_id"]) & set(test["breath_id"]):
        raise ValueError("Competition train/test breath IDs overlap.")
    inspect_data(train, test)
    train_mask, valid_mask = load_split(train)
    x_train = train.loc[train_mask, FEATURES]
    y_train = train.loc[train_mask, "pressure"]
    x_valid = train.loc[valid_mask, FEATURES]
    y_valid = train.loc[valid_mask, "pressure"]
    inspiration = x_valid["u_out"].eq(0)
    if not inspiration.any():
        raise ValueError("Validation contains no inspiration rows.")
    model = lgb.LGBMRegressor(**PARAMETERS)
    print(f"\nLightGBM parameters: {json.dumps(PARAMETERS)}", flush=True)
    fit_started = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - fit_started
    predictions = model.predict(x_valid)
    mae = mean_absolute_error(y_valid.loc[inspiration], predictions[inspiration])
    # A training-only constant provides a reference for the model's score.
    median_pressure = y_train.loc[x_train["u_out"].eq(0)].median()
    constant_mae = mean_absolute_error(
        y_valid.loc[inspiration], np.full(inspiration.sum(), median_pressure)
    )
    elapsed_seconds = perf_counter() - started
    record = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "model_name": "LightGBM baseline",
        "features": json.dumps(FEATURES),
        "mae": mae,
        "training_runtime_seconds": training_seconds,
        "total_runtime_seconds": elapsed_seconds,
        "observations": (
            "Trained on all phases. MAE scored only on validation u_out == 0. "
            "R and C treated as numeric. No temporal history features. "
            f"Training inspiration median constant MAE={constant_mae:.8f}."
        ),
        "train_breaths": train.loc[train_mask, "breath_id"].nunique(),
        "validation_breaths": train.loc[valid_mask, "breath_id"].nunique(),
        "validation_inspiration_rows": int(inspiration.sum()),
        "random_state": SEED,
        "split_file": SPLIT_PATH.name,
        "split_sha256": file_hash(SPLIT_PATH),
        "train_sha256": file_hash(train_path),
        "test_sha256": file_hash(test_path),
        "parameters": json.dumps(PARAMETERS, sort_keys=True),
        "python_version": platform.python_version(),
        "pandas_version": pd.__version__,
        "numpy_version": np.__version__,
        "sklearn_version": sklearn.__version__,
        "lightgbm_version": lgb.__version__,
    }
    record_experiment(record)
    print(f"\nValidation inspiration rows: {inspiration.sum():,}")
    print(f"Validation inspiration MAE: {mae:.8f} cmH2O")
    print(f"Constant median reference MAE: {constant_mae:.8f} cmH2O")
    print(f"Training runtime (fit only): {training_seconds:.3f} seconds")
    print(f"Total runtime through evaluation: {elapsed_seconds:.3f} seconds")
    print(f"Results recorded in {RESULTS_PATH}", flush=True)


if __name__ == "__main__":
    main()
