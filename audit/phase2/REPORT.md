# Phase 2 — minimal ML revalidation

Completed 2026-11-04 using existing Step 2/3 implementations and hyperparameters. No tuning, robustness experiments, README rewrite, Streamlit changes, commit or push. Finalization made limited Step 3 plotting corrections. The main runner is `run_revalidation.py` and the isolated failed-step retry is `retry_benchmark.py`; detailed execution is in `execution.log`.

## Dataset and validation

- Confirmed 158 conditions: Y=110, NL=8, NG=40. Rate/lag regression: 118 growth-positive observations; Ymax: 108. Eight held-out strains; 72 target/model/fold results. NG records remain excluded rather than treated as zero rates. Phase 1 master CSV unchanged; SHA256 `a0c8b7404c7bcdf4f2b0d834227b484227fe5b2c67c71654af7507b960184fb2`.
- Source doses remain 0.026/0.13/0.26%; added NaCl unchanged; TSB total is added +0.5 percentage points; SJ total remains missing. PCS is purified condensed smoke; rates retain log10 CFU/h. Phase 1 source-verification evidence remains authoritative.
- `fold_checks.csv` verifies disjoint strain sets for every target and the benchmark. Each observation has exactly one held-out prediction per model. No strain/species/source-metadata feature was used.
- Existing label encoders fit the full dataset, but training-only refits were explicitly checked to produce identical encodings in every main LOSO fold; every test category occurs in training. The benchmark also has no unseen categories (its constant PCS category can have a different ordinal offset without providing information). This is vocabulary equivalence, not a claim that the notebook literally fits encoders inside folds. Existing ordinal category distances remain a modeling limitation.
- GPR StandardScaler fits inside each training-fold Pipeline; its mean was checked against training-only data. GPR target normalization and kernel optimization occur during training fits. RF/XGBoost use no scaling or fitted imputation. No missing lag values were present before the existing fillna operation.

## Old versus new LOSO metrics

These are **unweighted means across eight folds**, not pooled OOF R². Old values are saved historical artifacts, not an old-data rerun. Full precision and standard deviations are in `old_vs_new_metrics.csv`; pooled R²/MAE/RMSE are separately in `pooled_oof_metrics.csv`; sample predictions in `oof_predictions.csv`.

| target | model | R2_mean_old | R2_mean_new | MAE_mean_old | MAE_mean_new | RMSE_mean_old | RMSE_mean_new |
| --- | --- | --- | --- | --- | --- | --- | --- |
| lag_h | GPR | 0.59544 | 0.58109 | 10.536 | 10.075 | 17.928 | 17.749 |
| lag_h | RandomForest | 0.19958 | 0.19958 | 13.766 | 13.766 | 27.27 | 27.27 |
| lag_h | XGBoost | 0.57907 | 0.57229 | 9.9948 | 10.052 | 17.659 | 17.785 |
| mu_max | GPR | 0.8919 | 0.89538 | 0.0059742 | 0.0059287 | 0.0071605 | 0.0070571 |
| mu_max | RandomForest | 0.87156 | 0.87156 | 0.006289 | 0.006289 | 0.0076802 | 0.0076802 |
| mu_max | XGBoost | 0.83909 | 0.83574 | 0.0069224 | 0.0070206 | 0.008699 | 0.0087834 |
| ymax | GPR | -0.97564 | -0.93644 | 0.12899 | 0.13002 | 0.16147 | 0.16221 |
| ymax | RandomForest | -1.0411 | -1.0411 | 0.12936 | 0.12936 | 0.16151 | 0.16151 |
| ymax | XGBoost | -1.1876 | -1.2198 | 0.13016 | 0.12991 | 0.16718 | 0.16919 |

Units: rate MAE/RMSE log10 CFU/h; lag hours; Ymax source-reported log CFU/g, with the Phase 1 denominator ambiguity unresolved. Choosing the best model on these same folds is not an independent post-selection performance estimate.

## Ratkowsky comparison

| Model | R²_old | RMSE_old | R²_new | MAE | RMSE_new |
| --- | --- | --- | --- | --- | --- |
| GPR | 0.90285 | 0.0087017 | 0.90285 | 0.0065061 | 0.0087017 |
| RandomForest | 0.89701 | 0.0089595 | 0.89701 | 0.0068333 | 0.0089595 |
| XGBoost | 0.79882 | 0.012522 | 0.79852 | 0.01014 | 0.012531 |
| Ratkowsky (mechanistic, temp only) | 0.76786 | 0.013451 | 0.76786 | 0.010047 | 0.013451 |

Recomputed on the **same 41 growth-positive Temperature-experiment observations** in TSB/SJ, with eight strain-held-out folds and no skipped Ratkowsky fits. This is a different subset from the main 118-observation rate analysis. Table R²/MAE/RMSE are **pooled OOF metrics**; per-fold metrics and row-level predictions are saved separately.

Ratkowsky fits sqrt(rate) from temperature alone and squares its predictions. ML uses the existing five feature columns; within this subset only temperature and medium vary, while added salt, PCS concentration and PCS type are constant. Thus held-out observations/folds match, but predictors and fitting objectives differ. This is not an identical-information comparison or evidence of universal ML superiority. The source paper's original per-strain fit is not reproduced by this pooled training-strain benchmark.

## GPR intervals and RF SHAP

Historical saved GPR coverage was 94.9%. Fresh empirical coverage: **112/118 = 94.92%**, nominal 95%. The existing mean ±1.96 predictive-standard-deviation calculation was collected from the same LOSO fits, avoiding a duplicate fit loop. Intervals include the existing WhiteKernel behavior; coverage on these correlated, small-sample observations does not establish general calibration or food-safety reliability. Per-fold coverage and all intervals are saved.

**Finalized:** RF SHAP completed on 118 growth-positive records. Maximum additivity error: 6.94e-17. The earlier runner had not saved its full-data RF; the user explicitly approved one fit using the unchanged Step 3 settings. That RF is now saved as `rf_mu_max_for_shap.pkl`. No LOSO or benchmark training was repeated.

| Feature | Mean absolute SHAP (log10 CFU/h) |
| --- | --- |
| Temperature (°C) | 0.01070805 |
| PCS Concentration (%) | 0.00338629 |
| Medium (TSB/SJ) | 0.00198307 |
| Added NaCl (%) | 0.00181550 |
| PCS Type | 0.00067286 |

SHAP describes in-sample model attributions, not causal treatment effects. Repaired PCS bins cover 0%, 0.026%, 0.13%, 0.26%, with both growth-positive 0.26% records included. Dependence plots include the high dose. Heatmaps now show all five features and include positive as well as negative SHAP values; two-feature sums are not interaction effects.

Environment: `../.phase2-shap-env` outside the repository, Python 3.13.5, isolated NumPy 2.1.3 overlay with read-only reuse of base packages: Numba 0.61.0, SHAP 0.51.0, scikit-learn 1.8.0. Base NumPy 2.5.4 was not changed. `shap_environment.json` records versions and setup details. A nonfatal base numexpr-version warning remains; computations and checks passed.

## Preservation, execution and unresolved issues

- All pre-run output/model files were copied byte-for-byte to `before/`, including existing uncommitted Phase 1 outputs. Backup hashes were verified against `before_hashes.json`. Restore only explicitly selected files from this directory if rollback is needed; no reset/stash was used.
- Completed stages: LOSO, GPR intervals from the same LOSO fits, Final model exports, Ratkowsky and ML subset benchmark (41 rows; retry after pre-fit count assertion).
- Earlier SHAP dependency block resolved in the isolated environment; no finalization analysis remains blocked.
- The first benchmark attempt stopped before fitting due to an incorrect runner assertion of 40 rows; the verified subset has 41. Only that failed step was retried. Main LOSO was executed once.
- No new features or hyperparameter tuning. Final export follows the notebook's highest fold-mean R² selection per target; a saved best model does not imply an acceptable target. Weak lag/Ymax generalization must be disclosed.
- SD versus SE and Ymax denominator ambiguities from Phase 1 remain unresolved. Results concern growth-positive conditions and do not validate growth/no-growth classification, universal inhibition thresholds, regulatory limits or food-safety decisions.
- Notebook saved outputs, narrative metrics, README and Streamlit hard-coded values were deliberately preserved and can disagree with this report. Only figures listed below are refreshed; all other historical plots remain stale. Step 3 PCS bins and affected plotting code were corrected, but historical notebook narrative and unexecuted inline outputs remain unsynchronized.
- Package versions are recorded in `run_status.json`. Old/new differences may include runtime-version effects: old results were not rerun in the current environment, so changes cannot be attributed exclusively to the concentration repair.

## Changed files and readiness

Existing files overwritten after backup:

- `outputs/step2_loso_performance_summary.csv`
- `outputs/step2_loso_per_fold_results.csv`
- `outputs/step2_ml_vs_ratkowsky_benchmark.csv`
- `models/best_lag_h.pkl`
- `models/best_mu_max.pkl`
- `models/best_ymax.pkl`
- `models/encoders.pkl`

New files are confined to `audit/phase2/`: runner, report builder, report, logs, backups, checks, predictions, intervals, metrics comparisons and manifests. `changed_files.txt` lists every new audit file and changed existing artifact.

**Portfolio readiness: READY WITH CAVEATS for the freshly revalidated numerical results in this report; repository-wide presentation is not yet synchronized.** Use explicit metric aggregation and subset labels. Old plots/narratives and unvalidated safety claims must not accompany the new results as if they were verified. No model performance claim extends to NG conditions or production use.

## Finalization figures and preservation

- `outputs/09a_model_comparison_r2.png`
- `outputs/08_loso_mu_max_observed_predicted.png`
- `outputs/09b_gpr_prediction_intervals.png`
- `outputs/14_shap_beeswarm_mu_max.png`
- `outputs/15_shap_bar_hurdle_ranking.png`
- `outputs/16_shap_dependence_plots.png`
- `outputs/17_shap_per_species.png`
- `outputs/18a_shap_total_heatmaps.png`
- `outputs/18b_shap_2feature_heatmaps.png`
- `outputs/19_shap_waterfall_examples.png`

All ten figures were visually reviewed. The comparison plot reports mean fold R² ± fold SD; observed/predicted panels report pooled OOF R². The GPR figure reuses saved intervals (112/118, 94.92%). No validation estimates were recalculated by training.

Pre-finalization outputs, notebook, report/status and SHAP comparison are recoverable under `before_finalization/`; previous Phase 2 backups remain under `before/`. Protected model/data/validation hashes match `finalization_protected_hashes.json`. Updated files additionally include Step 3 source, its two output CSVs, SHAP comparison and audit records. See `finalization_files.txt`. Base package installation and validated model files were not modified. The newly saved RF is an explanation model fitted once with explicit approval, not a newly validated replacement for the saved best GPR.

Stop condition reached: SHAP additivity, PCS membership, protected hashes and essential figure review passed. Remaining limitations are the source SD/SE and Ymax-unit ambiguities, growth-positive-only scope, ordinal encodings, small correlated sample, weak Ymax performance, and unsynchronized README/app/notebook narrative. No commit or push.
