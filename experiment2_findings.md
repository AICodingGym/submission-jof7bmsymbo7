Experiment 2 tests whether within-breath input history improves pressure prediction while holding the original model parameters and validation split fixed.

| Experiment | Inspiratory MAE (cmH2O) | Fit runtime (seconds) |
| --- | --- | --- |
| Baseline | 4.11851845 | 16.47 |
| Time-series features | 1.45853818 | 20.10 |

MAE decreased by 2.65998028 cmH2O, or 64.59%. This result supports the hypothesis for the seven-feature bundle on the fixed validation set. The target of 0.20 remains unmet.

The 12 inputs are R, C, time_step, u_in, u_out, u_in_lag_1, u_in_lag_2, u_in_lag_5, u_in_cumsum, u_in_diff, time_delta, u_in_area. The LightGBM parameters exactly match the baseline record, including 300 trees, learning rate 0.05, 31 leaves, L1 objective, random_state=42 and eight threads. Both runs train on all phases and score only inspiration.

Features are calculated after sorting by breath_id and time_step. Lags use earlier u_in values within the same breath. Unavailable lags are zero. The first difference and time_delta are zero at each breath start. u_in_cumsum includes the current input. u_in_area is the cumulative sum of current u_in multiplied by time_delta, with zero area at the first time step. This uses right-endpoint integration. Values are converted to float32, and original row ordering is restored before fitting and evaluation. Target pressure is removed before feature engineering.

Checks on a shuffled two-breath fixture verified all seven expected feature values, ID alignment, breath isolation, independence from target changes and independence from future control changes. The saved predictions were checked against original validation IDs, breath IDs, phases and target pressures. Recomputing inspiration MAE from these predictions matches experiments.csv. The saved split hash matches the original experiment, with 54,324 training breaths, 13,581 validation breaths and 412,450 scored validation rows. baseline.py and validation_split.csv hashes were checked before and after training.

| Position within inspiration | Baseline MAE | Experiment 2 MAE |
| --- | --- | --- |
| 0-20% | 2.266 | 0.944 |
| 20-40% | 3.580 | 1.406 |
| 40-60% | 4.529 | 1.545 |
| 60-80% | 4.980 | 1.642 |
| 80-100% | 5.354 | 1.786 |

| New feature | Share of total model gain |
| --- | --- |
| u_in_area | 14.74% |
| u_in_cumsum | 4.85% |
| u_in_lag_2 | 4.84% |
| u_in_lag_1 | 3.21% |
| time_delta | 2.74% |
| u_in_diff | 1.99% |
| u_in_lag_5 | 1.08% |

u_in_area has the largest gain among the new features. Its accumulated control history plausibly helps distinguish breaths with similar current settings but different earlier inputs. u_in_cumsum also represents accumulated input, while lags and u_in_diff capture recent dynamics. time_delta represents sample spacing. The full-bundle result supports adding input history, but correlated feature gain cannot identify each feature's independent contribution. Gain is calculated over training rows from both phases.

The next experiment should be an ablation using the original five baseline features plus u_in_area alone, with the same parameters and split. Comparing it with both recorded runs will show how much of the improvement can be recovered with this single history feature and whether the larger bundle provides additional value. This next experiment has not been run.

The complete R/C and time comparisons are saved in experiment2_error_comparison.csv. Training feature importance is saved in experiment2_feature_importance.csv. Per-row validation predictions are saved in outputs/experiment2_validation_predictions.csv for future analysis. Original runtime measurements cover fit only and total execution through evaluation respectively, excluding diagnostic file writing.
