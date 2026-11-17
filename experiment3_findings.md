Experiment 3 evaluates inspiration-only training with the same 12 model inputs, feature engineering, LightGBM parameters and breath-level validation split as Experiment 2.

| Training phases | Validation inspiratory MAE | Fit runtime, seconds |
| --- | --- | --- |
| All phases, Experiment 2 latest run | 1.45853818 | 19.64 |
| Inspiration, Experiment 3 | 1.29386150 | 6.55 |

Validation MAE improved by 0.16467668 cmH2O (11.29%). Training time fell by 66.64% compared with the latest Experiment 2 run. Experiment 2 has two recorded fit runtimes of 20.10 and 19.64 seconds, both with identical MAE. Runtime comparisons reflect single runs and machine load.

The result supports the hypothesis on this fixed validation set. Experiment 3 trains on 1,649,472 inspiration observations from 54,324 breaths. Experiment 2 trains on 4,345,920 observations including expiration. Both score exactly 412,450 inspiration rows from the same 13,581 validation breaths. The target MAE of 0.20 remains unmet.

The script imports FEATURES and add_history directly from experiment2.py and loads parameters from the latest Experiment 2 record. It computes temporal features on all 5,432,400 observations before filtering training and validation by u_out == 0. Target pressure is removed before feature engineering. Original row order and IDs are retained. Validation pressure is used only for evaluation.

A shuffled-breath check confirmed that history is computed before phase filtering, including a synthetic earlier expiratory row followed by an inspiratory row, and that breath boundaries and IDs remain correct. Original baseline.py, experiment2.py and validation_split.csv hashes were verified before and after the run. Previous experiment rows were checked for preservation after appending the new result. Saved Experiment 3 prediction IDs, breath IDs, lung settings, times, phases and target pressures exactly match Experiment 2's scored observations. Recomputed MAEs match experiments.csv.

| Position in inspiration | Experiment 2 MAE | Experiment 3 MAE |
| --- | --- | --- |
| 0-20% | 0.944 | 0.831 |
| 20-40% | 1.406 | 1.255 |
| 40-60% | 1.545 | 1.395 |
| 60-80% | 1.642 | 1.459 |
| 80-100% | 1.786 | 1.559 |

All nine R/C combinations and all five inspiration-position groups improve. Later inspiration remains harder, with MAE 1.559 in the final fifth versus 0.831 in the first fifth. R=50 groups retain the largest errors, ranging from 1.566 to 1.856. These remaining patterns motivate testing whether additional boosting can capture the remaining nonlinear relationships.

The paired R/C and inspiration-position results are saved in experiment3_error_comparison.csv. Predictions for future inspection are saved in outputs/experiment3_validation_predictions.csv. Feature engineering and all LightGBM parameters remain fixed. u_out stays in the feature list and is constant zero for this experiment's model inputs.

One plausible explanation is that the finite 300-tree model can allocate its splits to the scored pressure relationships when trained on inspiration alone. This also removes the many expiratory rows from the training objective, aligning it with evaluation. A smaller training set plausibly explains much of the runtime reduction. LightGBM also constructs feature bins from the filtered training data, so changed binning may contribute. These mechanisms were not isolated in this experiment.

The next experiment should keep the inspiration-only training setup, features and split fixed, and increase only n_estimators from 300 to 600. Record the validation MAE at each boosting iteration to test whether additional boosting reduces remaining errors. This is a hypothesis about remaining model capacity. The present results do not establish underfitting. No additional model has been trained in this review.
