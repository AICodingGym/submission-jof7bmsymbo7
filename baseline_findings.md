# Baseline validation findings

The recorded LightGBM baseline achieves 4.11851845 cmH2O validation MAE. Training takes 16.47 seconds, and total runtime through evaluation is 18.56 seconds. These runtimes come from the original experiment record.

The model uses R, C, time_step, u_in and u_out with 300 trees, learning rate 0.05, 31 leaves and an L1 objective. Training includes both phases. Evaluation includes only u_out == 0.

The constant reference predicts the training-inspiration median of 15.82039636 cmH2O. Its validation MAE is 7.46844227 cmH2O. LightGBM reduces MAE by 3.34992382 cmH2O, or 44.85%.

## Validation data and evidence

The existing split contains 54,324 training breaths and 13,581 validation breaths, with zero overlap. Validation contains 1,086,480 rows, including 412,450 inspiration rows. The split was originally created with 80% training breaths, 20% validation breaths and random_state=42.

This review loaded train.csv and selected validation breaths using validation_split.csv. It reused the previously saved baseline_error_analysis.csv prediction summaries. No model was fitted during this review. The earlier diagnostic run reproduced the original aggregate MAE before saving these summaries.

The training-file and split SHA-256 hashes match experiments.csv. All five saved breakdowns were checked against the current validation data for group counts, mean pressure and constant-reference MAE. Every breakdown covers all 412,450 scored rows, and its row-weighted MAE equals the recorded baseline MAE. Individual model prediction errors cannot be independently recomputed from aggregate summaries because per-row predictions and the fitted model were not saved.

Split SHA-256 is `e6c05db6b52c0c2bbb0d68b4cc1e47027d2bba163dbc77f9fa9e4769f2fe51d5`.

## Errors across resistance and compliance

Every cell below is validation inspiration MAE in cmH2O.

| R | C = 10 | C = 20 | C = 50 |
| --- | --- | --- | --- |
| 5 | 5.356 | 2.651 | 1.621 |
| 20 | 5.360 | 3.642 | 2.532 |
| 50 | 5.342 | 5.238 | 4.529 |

At C=10, all three resistance settings have MAE near 5.35. At C=50, MAE rises from 1.621 at R=5 to 4.529 at R=50. The difficulty therefore depends on the combination of R and C. Each combination contains between 32,693 and 76,512 scored rows. These are descriptive associations from the fixed validation set.

## Errors within inspiration

Relative position uses the ordered inspiration rows in each breath. The first fifth represents the beginning, the third fifth represents the middle and the final fifth represents the end. This position is used only for analysis.

| Position within inspiration | Rows | MAE | Mean prediction minus actual | Mean actual pressure |
| --- | --- | --- | --- | --- |
| 0-20% | 87,673 | 2.266 | -0.581 | 10.585 |
| 20-40% | 81,468 | 3.580 | -0.988 | 16.623 |
| 40-60% | 77,357 | 4.529 | -0.978 | 19.069 |
| 60-80% | 81,468 | 4.980 | -0.844 | 20.417 |
| 80-100% | 84,484 | 5.354 | -0.847 | 21.313 |

Errors increase from the beginning through the middle to the end. Final-fifth MAE is 2.36 times first-fifth MAE. Negative signed errors indicate average underprediction. Mean actual pressure also rises through inspiration, so increasing absolute error alone does not establish its cause.

The grouping by elapsed time confirms the same pattern.

| Time in seconds | Rows | MAE |
| --- | --- | --- |
| [0.0, 0.25) | 108,645 | 2.414 |
| [0.25, 0.5) | 101,267 | 3.986 |
| [0.5, 0.75) | 105,972 | 4.857 |
| [0.75, 1.0) | 96,566 | 5.365 |

## Recommended experiment

Add one feature that accumulates u_in over elapsed time separately within each breath. Sort by breath_id and time_step, initialize the feature to zero, and add 0.5 * (previous_u_in + current_u_in) * (current_time - previous_time) at each subsequent row. Reset the accumulation at each new breath. Use current and earlier control values only.

Pressure plausibly depends on previous air delivery as well as current valve opening and lung settings. The baseline's five features provide no explicit control history. Two rows with similar current inputs can have different accumulated input histories, and that difference may become more consequential later in inspiration. The higher late-inspiration errors motivate this hypothesis. u_in measures valve opening, so its integral is a proxy for accumulated air delivery.

For a controlled experiment, add only this feature and keep the model parameters, saved split, training phases and inspiration-only metric fixed. Compare overall MAE, the R/C breakdown and the time breakdown with this report. Improvement remains untested.
