"""Refit Experiment 4 Group C on all breaths and verify a local submission."""

import json
import os
from time import perf_counter

import lightgbm as lgb
import numpy as np
import pandas as pd

from baseline import ROOT, file_hash
from experiment4 import ORIGINAL_FEATURES, engineer_features


def verify_submission(path, test, sample):
    submission = pd.read_csv(path)
    if submission.columns.tolist() != ["id", "pressure"]:
        raise ValueError("Submission must contain exactly id and pressure, in order.")
    if len(submission) != len(test) or len(submission) != len(sample):
        raise ValueError("Submission, test and sample row counts differ.")
    if submission.isna().any().any():
        raise ValueError("Submission contains missing IDs or predictions.")
    if not submission["id"].is_unique:
        raise ValueError("Submission contains duplicate test IDs.")
    if set(submission["id"]) != set(test["id"]):
        raise ValueError("Submission does not cover exactly the test IDs.")
    if not submission["id"].equals(sample["id"]):
        raise ValueError("Submission IDs differ from sample submission order.")
    if not np.isfinite(submission["pressure"]).all():
        raise ValueError("Submission contains nonfinite predictions.")
    return {
        "columns": submission.columns.tolist(),
        "row_count": len(submission),
        "unique_ids": int(submission["id"].nunique()),
        "every_test_id_exactly_once": True,
        "sample_submission_id_order_matches": True,
        "missing_values": int(submission.isna().sum().sum()),
        "all_predictions_finite": True,
        "prediction_min": float(submission["pressure"].min()),
        "prediction_max": float(submission["pressure"].max()),
    }


def main():
    started = perf_counter()
    submission_path = ROOT / "submission.csv"
    verification_path = ROOT / "submission_verification.json"
    # Protect existing experiment scripts, configurations, records and reports.
    protected = {
        path: file_hash(path) for path in ROOT.iterdir()
        if path.is_file() and path not in {submission_path, verification_path}
    }
    config = json.loads((ROOT / "experiment4_configs.json").read_text(encoding="utf-8"))
    records = pd.read_csv(ROOT / "experiments.csv")
    group_c = records.loc[records["model_name"].eq("LightGBM Experiment 4 - C")].iloc[-1]
    features = ORIGINAL_FEATURES + config["groups"]["C"]
    parameters = config["parameters"]
    if features != json.loads(group_c["features"]):
        raise ValueError("Feature list differs from recorded Group C.")
    if parameters != json.loads(group_c["parameters"]):
        raise ValueError("Hyperparameters differ from recorded Group C.")
    if file_hash(ROOT / "validation_split.csv") != group_c["split_sha256"]:
        raise ValueError("Original validation split differs from Group C.")
    train_hash = file_hash(ROOT / "data/train.csv")
    test_hash = file_hash(ROOT / "data/test.csv")
    if train_hash != group_c["train_sha256"] or test_hash != group_c["test_sha256"]:
        raise ValueError("Dataset differs from recorded Group C.")
    dtype = {
        "id": "int64", "breath_id": "int64", "R": "int16", "C": "int16",
        "time_step": "float32", "u_in": "float32", "u_out": "int8",
    }
    train = pd.read_csv(ROOT / "data/train.csv", dtype=dtype)
    test = pd.read_csv(ROOT / "data/test.csv", dtype=dtype)
    sample = pd.read_csv(ROOT / "data/sample_submission.csv", usecols=["id"])
    if not test["id"].is_unique or not sample["id"].is_unique:
        raise ValueError("Test and sample IDs must be unique.")
    if len(test) != len(sample) or set(test["id"]) != set(sample["id"]):
        raise ValueError("Test and sample submission IDs differ.")
    if set(train["breath_id"]) & set(test["breath_id"]):
        raise ValueError("Train and test breaths overlap.")
    train_inputs = engineer_features(train.drop(columns="pressure"), config)
    inspiration = train["u_out"].eq(0)
    if train.loc[inspiration, "breath_id"].nunique() != train["breath_id"].nunique():
        raise ValueError("At least one training breath has no inspiration rows.")
    x_train = train_inputs.loc[inspiration, features]
    y_train = train.loc[inspiration, "pressure"]
    print(f"Training on all {train['breath_id'].nunique():,} breaths, "
          f"using {len(x_train):,} inspiration rows and {len(features)} features.", flush=True)
    model = lgb.LGBMRegressor(**parameters)
    fit_started = perf_counter()
    model.fit(x_train, y_train)
    training_seconds = perf_counter() - fit_started
    # Test histories are calculated separately from training histories.
    test_inputs = engineer_features(test, config)
    if not test_inputs["id"].equals(test["id"]):
        raise ValueError("Test feature IDs are misaligned.")
    test_predictions = pd.Series(
        model.predict(test_inputs[features]), index=test["id"], name="pressure"
    )
    submission = pd.DataFrame({
        "id": sample["id"],
        "pressure": test_predictions.reindex(sample["id"]).to_numpy(),
    })
    temporary_path = ROOT / "submission.pending.csv"
    submission.to_csv(temporary_path, index=False)
    verification = verify_submission(temporary_path, test, sample)
    for path, previous_hash in protected.items():
        if file_hash(path) != previous_hash:
            raise ValueError(f"Existing project file changed: {path.name}")
    os.replace(temporary_path, submission_path)
    verification.update({
        "submission_path": str(submission_path.resolve()),
        "submission_sha256": file_hash(submission_path),
        "source_model": "LightGBM Experiment 4 - C",
        "source_validation_mae": float(group_c["mae"]),
        "features": features,
        "parameters": parameters,
        "training_breaths": int(train["breath_id"].nunique()),
        "training_inspiration_rows": int(inspiration.sum()),
        "test_inspiration_rows": int(test["u_out"].eq(0).sum()),
        "test_expiration_rows": int(test["u_out"].eq(1).sum()),
        "training_runtime_seconds": training_seconds,
        "total_runtime_seconds": perf_counter() - started,
        "train_sha256": train_hash,
        "test_sha256": test_hash,
        "config_sha256": file_hash(ROOT / "experiment4_configs.json"),
        "existing_project_files_unchanged": True,
    })
    verification_path.write_text(json.dumps(verification, indent=2), encoding="utf-8")
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
