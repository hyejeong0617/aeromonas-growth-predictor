# Phase 1 source-data verification and repair

Completed 2026-11-04. Scope authorized by the user's attached Phase 1 request, including its explicit author confirmation of 0.26% PCS. No Phase 2 work, commit or push was performed.

## Repository and evidence

Repository: `hyejeong0617/aeromonas-growth-predictor`; local checkout `aeromonas-growth-predictor/`. Branch `main`, starting commit `0c2e4b8fcfd759675304337f62089d4029b41cab`. A new checkout was created because this workspace contained no checkout of the repository. Its initial working tree was clean; unrelated parent-workspace files were not overwritten.

Primary evidence and unit definitions: `../../DATA_PROVENANCE.md`. Original DOCX was inspected structurally through all four tables, including raw rows and captions; it was not rendered for visual source-layout review. Original bytes were verified unchanged by SHA256. Original source text, statistical marks and captions are retained in `supplementary_raw_extraction.json`. Source row numbers below are **one-based table rows including the header**. Notebook cells below are **zero-based JSON cell indices**, not execution counts.

## Complete comparison

| Target | VERIFIED before | MISMATCH before | MISSING before |
|---|---:|---:|---:|
| Original Step 1 data construction | 135 | 6 | 17 |
| Original master CSV | 135 | 6 | 17 |

These are condition-level comparisons of original scalar fields, not a claim that all metadata were complete. The 17 missing source conditions are additional to the original 141 records. Parameter uncertainties were separately absent from the original master; `source_metadata_findings.csv` records this and two AMBIGUOUS source-definition findings. Every condition now matches the normalized source transcription, including the explicit author correction. Post-repair: **158 VERIFIED**, zero condition mismatches or missing reported conditions. This does not resolve the two scientific ambiguities.

| Table | Conditions | Repaired experiment |
|---|---:|---|
| A | 48 | Temperature × medium |
| B | 42 | Added NaCl × temperature |
| C | 34 | VTABB PCS × temperature |
| D | 34 | JJT01 PCS × temperature |

Final flags: Y=110, NL=8, NG=40; growth-positive=118; Ymax available=108; NA Ymax=10. Eight strains, seven species. No duplicate condition keys. No unreported experimental combinations were invented.

## Confirmed corrections and exact source locations

| ID | Source | Original → corrected | Implementation and downstream dependencies |
|---|---|---|---|
| E01 | Main article §2.3.3; Tables C/D captions | Bacteriocin → purified condensed smoke (PCS); removed unsupported bacteriocin-specific and legal-limit explanations | README, all three notebook source texts, Streamlit source labels. Existing saved SHAP/model interpretations remain stale. |
| E02 | Table D rows 21, 28; author confirmation | A. piscicola and A. salmonicida SU2, 8°C JJT01: 2.6 → 0.26% | Step 1 cell 7, reconciliation cell 8, master, figure/filter cell 21. Their measured μmax, lag, Ymax, R² and SE are unchanged. Changes predictor coordinates and therefore requires later fitting/evaluation/SHAP refresh. |
| E02-NG | Tables C/D all high-dose rows; author confirmation | Published 2.6% → actual 0.26%, including 14 NG observations | Source snapshot and cell 8 preserve raw label and correction reason; 16 high-dose records overall, 14 NG and 2 growth-positive. |
| E03 | Table headers; article §2.6 | Misleading h−1 labels → log10 CFU/h; time formula drops extra ln(10) | Step 1 cells 8/28, Step 2 target/metric labels, Step 3 labels, `streamlit_app.py::time_to_threshold`. Numerical rate targets unchanged. Ratkowsky fitting was not run. |
| E05 | Table B; article §3.1.2 | Unqualified salt → added NaCl; TSB total = added + 0.5 percentage points | Step 1 cell 8 metadata, cells 19/25 axes and categorical status, README, other notebook labels, Streamlit text. Existing feature values unchanged; SJ total remains unknown. |
| D01 | Table C rows 2, 23, 33 | Growth flag Y → NL for A. media 4°C, A. salmonicida SU2 4°C, A. salmonicida Nr.21 8°C, all VTABB 0.026% | Step 1 cell 8 reconciliation and master. Lag stays 0; raw NL retained. Affects flag counts, not measured rates. |
| D02 | Table D row 10 | A. bestiarum, 4°C, JJT01 0.13%, final SE: 0.060 → 0.066 | Step 1 cell 7 and master. Clear transcription error; not a change to parameter uncertainty or target values. |
| D03 | Table C rows 3,7,10,14,21,28,35,39,42,43,47; Table D rows 7,14,35,39,43,47 | Missing → 17 reported NG conditions | Cell 8 loads source snapshot and appends only these reported conditions. Preserves absent kinetic values. NG excluded from ordinary rate regression; do not treat them as zero-rate targets. |
| D04 | Parameter entries in Tables A–D | Omitted uncertainties/flags → explicit source fields | Cell 8 and master preserve raw values, numerical ± quantities, caption-defined SD/SE, n=3, source rows and corrected species metadata. Not added to ML feature lists. |

The two high-dose growth-positive records remain μmax=0.012 for both, lag=351.8 h and 245.4 h respectively; Ymax is NA and 9.30 respectively; R²=0.91/0.98 and SE=0.149/0.086. Source-table means in the other previously present records were preserved except the confirmed SE transcription correction.

`correction_log.csv` contains each missing/mismatched condition with its key, source row and original/corrected values or insertion reason. `source_comparison_before.csv` compares both original targets independently; `source_comparison_after.csv` checks all repaired fields. `tracked_hashes_before.json` preserves the initial tracked-file fingerprints.

## Unresolved questions — no speculative correction

1. **AMBIGUOUS: SD versus SE.** Table A says SD; B–D say SE; the main article describes SD. Preserve each caption and value. Do not convert, infer weighting or equate parameter uncertainties with the separate final SE column pending author clarification.
2. **AMBIGUOUS: Ymax denominator.** Supplementary headers say log CFU/g despite liquid experiments; main text uses CFU/ml for liquids. Preserve source headers and values. No unit conversion applied.

## Executed QA and limits

- Executed only Step 1 setup constants, canonical construction cells 4–9 and export cell 11. Full notebooks were not run. Fresh stdout is recorded in `data_generation_log.txt`; stale Step 1 inline outputs were cleared.
- Compared all 158 normalized records/fields with the source snapshot, and source raw cells with the DOCX. Verified row counts, unique keys, permitted doses, the two high-dose growth-positive records, NG/NL/NA handling and NaCl mapping.
- Compiled notebook code cells and Streamlit source without importing or launching the app. Tested the isolated time function: rates 0.05/0.025/0.1, log10 rise 2→5, lag 5 h produce 65/35/125 h. Also checked a source-rate example and zero-rate handling.
- Rendered **only** EDA cells 19, 21, 25: figures 03, 04a, corrected 04b and 06. Visually inspected the four panels. NG markers are categorical rather than measured zero rates; unreported combinations are not labelled NG; unsupported EU annotation removed.
- Hash checks confirmed original model files, evaluation CSVs, SHAP artifacts and all nonselected existing PNGs were unchanged. `qa_results.json` records PASS for executed checks. This is source fidelity QA, not model validation.
- No Ratkowsky, RF, GPR, XGBoost fitting; no LOSO, SHAP computation, full notebook run, saved-model loading, deployment, app launch, commit or push. Plotting emitted nonfatal headless-show and installed numexpr-version warnings; files rendered successfully without package installation.

## Stale dependencies and Phase 2 checklist

Treat all existing model performance claims as historical pending revalidation. The app's current cold-start path trains RF/GPR; do not launch it as a supposedly read-only review.

1. Review repaired dataset and source ambiguities; approve Phase 2 explicitly.
2. Freeze the approved dataset and provenance hashes. Ensure source metadata cannot enter features. Review NG exclusion, NL zeros, NA outcomes, experimental-condition grouping and any preprocessing leakage.
3. Revalidate split design and grouped held-out-strain evaluation, tuning boundaries, target transformations, uncertainty calibration and extrapolation limits before interpreting metrics. Do not call interval coverage or a time scenario a validated safety guarantee.
4. Refit the approved Ratkowsky and ML models; regenerate `models/best_mu_max.pkl`, `models/best_lag_h.pkl`, `models/best_ymax.pkl`, `models/encoders.pkl` only in approved Phase 2.
5. Recompute `outputs/step2_loso_per_fold_results.csv`, `step2_loso_performance_summary.csv`, `step2_ml_vs_ratkowsky_benchmark.csv` and figures `07_ratkowsky_v3.png` through `13_ml_vs_ratkowsky_benchmark.png`. Reconcile README, notebook saved outputs and app metrics with the fresh results; old metric discrepancies are not certified fixed by this data repair.
6. Recompute `outputs/step3_shap_values.csv`, `step3_hurdle_ranking.csv` and figures 14–19. Review the Step 3 cell 21 PCS bins (old upper bin ends at 0.20%) so the corrected 0.26% cases are represented intentionally; review dependence-axis limits and causal/efficiency interpretations. SHAP is model attribution, not treatment-effect proof.
7. Refresh or explicitly retire other historical EDA assets, especially `05a_correlation_heatmap.png`, `05b_scatter_matrix.png`. Figures 01/02 were not rerendered and can retain historical units. The old `04b_jjt01_2p6_extreme.png` remains untouched for traceability and must not be used; its replacement is `04b_jjt01_0p26_extreme.png`.
8. Test the app against approved refreshed artifacts and the source units; review domain limits and unsupported regulatory/safety claims before publication. No app redesign was undertaken in Phase 1.

Phase 1 is complete for source-data repair and ready for review. Model/portfolio results are **not freshly validated**. Stop here pending Phase 2 approval.

## Exact changed-file inventory

See `changed_files.txt` for repository-relative modified/new files and `workspace_helper_files.txt` for task scripts created outside the checkout. No original source DOCX or unrelated user files were changed.
