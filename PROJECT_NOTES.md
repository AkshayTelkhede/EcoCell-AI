# Project Notes

## 1. Problem being solved
The project predicts continuous 5G base-station `Energy` consumption from timestamped cell measurements, energy-saving mode indicators, and base-station metadata.

## 2. Dataset explanation
`ECdata.csv` supplies 92,629 labeled energy rows. `CLdata.csv` supplies load and energy-saving mode features. `BSinfo.csv` supplies static radio metadata. The existing submission CSVs contain predictions for future rows without ground truth and are not used for evaluation.

## 3. Why the chosen models were used
The mean predictor establishes a minimum reference. Ridge provides a transparent linear alternative after one-hot encoding. ExtraTrees captures nonlinear interactions and is robust for tabular data. HistGradientBoosting provides an efficient boosted-tree model, and a small three-candidate tuning step tests whether its tree complexity and learning rate help.

## 4. Baseline approach
The baseline predicts the mean training energy for every validation row. It is intentionally simple and makes the value of feature-based models measurable.

## 5. Evaluation metrics and why they were chosen
MAE is the primary metric because it is in the target's units and is less dominated by large errors than squared loss. RMSE emphasizes larger misses. MAPE provides relative error but should be interpreted carefully around small target values. R2 summarizes variance explained relative to the mean baseline.

## 6. Main experimental findings
On the chronological holdout of 9,680 later rows, tuned HistGradientBoosting was best by MAE at 1.900004 and achieved RMSE 2.646730, MAPE 6.803583%, and R2 0.964758. ExtraTrees was close at MAE 1.935885. All feature-based models substantially beat the mean baseline MAE of 10.978056.

## 7. Important SHAP findings
The largest mean absolute SHAP values for the selected tuned tree model were `RUType` (7.319121), `load` (5.063510), `Antennas` (1.398547), `TXpower` (0.925021), and `BS` (0.905863). These are predictive contributions, not causal conclusions. A local waterfall is generated for one sampled row.

## 8. Limitations
The future submission rows have no ground truth in this repository. The evaluation is therefore a single chronological holdout, not a competition test score. The pipeline follows the original `Cell0` selection. Base-station identity can help prediction but may reduce generalization to unseen stations. SHAP explanations describe model behavior, not physical causality.

## 9. What was changed/improved
The Colab/Drive dependencies were removed from the runnable path. Data loading and validation were separated from feature construction. The join is validated, seeds are fixed, target-derived lags are excluded, a chronological split is explicit, a baseline and alternatives are compared, lightweight tuning is added, residual plots and metrics are saved, and SHAP global/local artifacts are generated. Thin notebooks now call the tested source modules.

## 10. Likely interview questions and concise answers

1. **Why use a chronological split?**  Energy prediction is time-dependent, so later timestamps are a more realistic validation scenario than a random split.
2. **Why exclude Energy lags?**  The future submission rows do not contain observed Energy, so target lags would not be available at inference and could create leakage.
3. **Why report MAE and RMSE together?**  MAE is easy to interpret in energy units, while RMSE makes unusually large errors more visible.
4. **What does SHAP add?**  It shows global feature influence and explains how features move an individual prediction relative to the model's expected value.
5. **Can these metrics be called test performance?**  No. They are chronological holdout metrics; the supplied future submission rows have no labels for an independent test score.
