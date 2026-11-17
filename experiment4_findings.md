Experiment 4 compares input feature groups on the exact saved validation split with inspiration-only training and unchanged LightGBM parameters.

Experiment 3 reproduced its recorded MAE of 1.29386150 cmH2O. Its reproduced validation predictions match the previous saved predictions within absolute tolerance 1e-10. This run used 1,649,472 training inspiration rows from 54,324 breaths and scored 412,450 validation inspiration rows from 13,581 breaths.

| Variant | Input count | Inspiratory MAE | Fit seconds | MAE reduction vs Experiment 3 |
| --- | --- | --- | --- | --- |
| reference | 12 | 1.29386150 | 6.68 | 0.00% |
| A | 16 | 1.27421053 | 7.44 | 1.52% |
| B | 18 | 1.19476319 | 7.40 | 7.66% |
| C | 22 | 1.17897336 | 8.82 | 8.88% |

The reference uses Experiment 3's original 12 features. Group A adds R*C, R/C, u_in/C and u_in*R. Group B adds lags 10, 15 and 20, trailing means over 5 and 10 rows, and trailing population standard deviation over 5 rows. Group C adds both groups. Each variant retains all original features.

All variants use 300 trees, learning rate 0.05, 31 leaves, regression_l1 objective, random_state=42, eight threads, deterministic training and force_col_wise=True. Features are engineered on complete breaths before phase filtering. Rolling windows include the current row and available earlier rows, with min_periods=1. Population standard deviation uses ddof=0, producing zero at the first row. Missing lags are zero. Added features use float32. Features are sorted by breath_id, time_step and id for calculation, then restored to original row ordering. No pressure reaches feature engineering, and neither IDs nor breath IDs are model inputs.

Experiment 3's validation errors by lung settings are shown below.

| R, C | Rows | Experiment 3 MAE |
| --- | --- | --- |
| 5, 10 | 42,382 | 1.158 |
| 5, 20 | 47,044 | 0.940 |
| 5, 50 | 42,894 | 0.536 |
| 20, 10 | 33,957 | 1.231 |
| 20, 20 | 32,693 | 1.108 |
| 20, 50 | 42,969 | 1.075 |
| 50, 10 | 76,512 | 1.856 |
| 50, 20 | 46,670 | 1.566 |
| 50, 50 | 47,329 | 1.648 |

The largest average error by lung setting is R=50, C=10, at 1.856 cmH2O. All R=50 combinations remain relatively difficult. R=5, C=50 has the lowest average error, at 0.536 cmH2O.

| Time in seconds | Rows | Experiment 3 MAE |
| --- | --- | --- |
| [0.0, 0.25) | 108,645 | 0.874 |
| [0.25, 0.5) | 101,267 | 1.346 |
| [0.5, 0.75) | 105,972 | 1.432 |
| [0.75, 1.0) | 96,566 | 1.560 |

Average errors increase during inspiration. The hardest joint lung-setting/time bin is R=50, C=10 at 0.25-0.50 seconds, with MAE 2.020 across 18,650 scored rows. Later inspiration is difficult overall, while the precise time pattern varies with R and C.

The largest individual reference absolute error is 33.538 cmH2O at id 1961223, breath_id 46023, R=5, C=10, time_step=0.743804 seconds. Actual pressure is 6.400 and predicted pressure is 39.938. The 20 largest individual errors are saved in outputs/experiment4_reference_largest_errors.csv. This identifies observations for inspection without establishing a data-quality cause.

| Time in seconds | Experiment 3 | A | B | C |
| --- | --- | --- | --- | --- |
| [0.0, 0.25) | 0.874 | 0.863 | 0.842 | 0.824 |
| [0.25, 0.5) | 1.346 | 1.315 | 1.290 | 1.249 |
| [0.5, 0.75) | 1.432 | 1.410 | 1.319 | 1.318 |
| [0.75, 1.0) | 1.560 | 1.546 | 1.356 | 1.352 |

Group C is best with MAE 1.17897336, an 8.88% reduction from Experiment 3. Group B's 7.66% improvement exceeds Group A's 1.52% improvement. Combining them adds approximately 1.32% relative improvement over Group B. Longer history provides information about earlier controls and smoothed recent input trajectories. Lung interactions give trees direct access to nonlinear combinations of controls and lung settings. These results support complementary benefits on the fixed validation set.

In the combined model, u_in_area accounts for 50.25% of training split gain and R/C accounts for 11.28%. The added rolling and lag features also receive split gain. Correlated feature importance cannot isolate each feature's independent contribution. The group-level experiments provide the measured evidence for the bundle comparisons. Group C improves all nine R/C combinations relative to Experiment 3. The 0.20 target remains unmet.

The next controlled investigation should separate long-lag features from rolling statistics. Starting with winning Group C, remove only the rolling mean and standard-deviation features, keeping lung interactions, long lags, original features, parameters and validation split fixed. This ablation will test the incremental contribution of rolling statistics and clarify which history representation deserves further development. That model has not been trained.

Reproduce the reference with `python experiment4.py --groups reference`, or a variant with `python experiment4.py --groups A`, `--groups B` or `--groups C`. Running the script without options reproduces all four. Feature groups and parameters are saved in experiment4_configs.json. Results append to experiments.csv, including the reference reproduction. Fit runtime excludes feature preparation and diagnostic writing. Total runtime includes shared preparation plus the selected model through evaluation, so totals across variants overlap in their shared preparation cost.

Every variant's R/C, time, relative-inspiration and joint R/C/time summaries are saved in experiment4_<variant>_error_analysis.csv. Feature importance is saved in experiment4_<variant>_feature_importance.csv. Per-row predictions and the largest-error rows are saved under outputs/. Checks verified all prediction IDs and targets against the previous validation output, matching parameters and features, scored-row counts, recorded MAEs and weighted error summaries. Hash checks protected previous scripts and validation_split.csv. Existing experiment rows were verified after appending the four runs.
