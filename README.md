# Aeromonas Growth | Predictive Microbiology & Machine Learning

**Microbial growth kinetics · Leave-one-strain-out validation · Gaussian process regression · Explainable ML**

[![Publication](https://img.shields.io/badge/Peer--reviewed-International%20Journal%20of%20Food%20Microbiology-blue)](https://doi.org/10.1016/j.ijfoodmicro.2022.109985)
[![Streamlit app](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://aeromonas-growth-predictor.streamlit.app/)

**[Open the interactive case study](https://aeromonas-growth-predictor.streamlit.app/)** · **[Read the publication](https://doi.org/10.1016/j.ijfoodmicro.2022.109985)** · **[Review data provenance](DATA_PROVENANCE.md)**

## Project overview

**Can machine learning estimate the growth rate of an *Aeromonas* strain that was excluded from training?**

This project connects **published food microbiology research on *Aeromonas* isolated from ready-to-eat seafood** with a subsequent Python machine learning extension. Experimental kinetic summaries were reconstructed from the publication's supplementary tables, checked against their source and evaluated with strain-held-out cross-validation. A lightweight Streamlit application presents the published experimental conditions, later ML validation and interpretation.

The original paper and the later ML project are **distinct contributions**. The ML performance numbers below are from the revalidated 2026 repository analysis, **not** from the original peer-reviewed publication.

| Study scope | Verified repository evidence |
|---|---|
| **158 experimental conditions** | 110 growth observations (Y), 8 growth observations with no lag (NL), 40 no-growth observations (NG) |
| **8 strain groups, 7 species** | Temperature, medium, added NaCl and purified condensed smoke (PCS) treatments |
| **118 growth-positive observations** | Main growth-rate and lag regression dataset; NG is not assigned a numeric zero rate |
| **8 held-out strain folds** | GPR, Random Forest and XGBoost compared on maximum growth rate (μmax) |

*The reconstructed data consist of published kinetic summaries, not independent raw growth-curve time series or an externally validated food-product dataset.*

## Interactive case study

**[Launch the Streamlit application](https://aeromonas-growth-predictor.streamlit.app/)**

The app is a three-section research explorer rather than a shelf-life or safety optimizer:

1. **Original research:** experimental conditions and measured growth-rate exploration, with NG/NL status preserved.
2. **Machine learning approach:** models, explanatory workflow and eight-fold leave-one-strain-out (LOSO) evaluation.
3. **Results & insights:** held-out predictions, Random Forest SHAP attribution, GPR prediction intervals and limitations.

The application reads committed CSV artifacts and **does not retrain models, run SHAP or load pickle files on startup**. The public URL is linked for convenience; a fresh hosted deployment smoke test should be performed before external promotion.

## Key scientific and ML findings

- **LOSO performance:** GPR attained mean strain-fold R² **0.8954**, compared with **0.8716** for Random Forest and **0.8357** for XGBoost. These are unweighted means across eight strain folds, **not** pooled out-of-fold scores; selecting the best model on these folds is not independent post-selection validation.
- **Uncertainty:** 112/118 growth-positive observations (**94.92%**) fell within GPR's nominal 95% intervals during internal LOSO evaluation. This does not establish independent calibration or reliable uncertainty outside the study conditions.
- **Model interpretation:** RF mean-absolute SHAP importance ranked **temperature > PCS concentration > medium > added NaCl > PCS type**. This is an attribution to a fitted model, **not** evidence of causal treatment effects, antimicrobial efficiency or interaction.
- **Ratkowsky reference:** on a **separate 41-observation temperature-experiment subset**, pooled OOF R² was **0.90285** for GPR and **0.76786** for a temperature-only Ratkowsky fit. The ML model also uses medium, and the fitting objectives differ; this is **not** an identical-information comparison or a reproduction of the paper's per-strain Ratkowsky analysis.
- **Scope decision:** lag-time performance was variable and Ymax generalization was poor (negative mean fold R²); neither is presented as a validated application target.

The validated source and result definitions are documented in **[DATA_PROVENANCE.md](DATA_PROVENANCE.md)**, the [Phase 1 source audit](audit/phase1/REPORT.md) and the [Phase 2 revalidation report](audit/phase2/REPORT.md).

## Research and project contributions

**Original published research (PhD):** Lee et al. (2023) investigated *Aeromonas* growth kinetics under temperature, sodium chloride, purified condensed smoke (PCS) and culture-medium conditions. This is collaborative peer-reviewed research. First authorship does not imply that all experiments or analysis stages were performed individually; consult the publication for the author list and study methods.

**Later ML extension:** This repository structures the published kinetic summaries for Python analysis and implements GPR, Random Forest, XGBoost, LOSO evaluation, RF SHAP analysis and an interactive research explorer. These analyses are separate from the original publication. Attribution of individual implementation work and AI assistance should not be inferred solely from repository ownership.

**Source reconciliation and revalidation (2026):** The supplementary tables were compared condition by condition (158/158 reconciled). The author confirmed that the high PCS dose was **0.26%**, correcting conflicting **2.6%** table labels; omitted NG conditions and record flags were restored. Existing ML settings were then rerun and results checked. See the audit records for exact changes and remaining ambiguities.

## Methods and technologies

| Area | Methods / tools |
|---|---|
| **Published microbiology study** | Growth kinetics, experimental temperature/NaCl/PCS conditions, Baranyi and Roberts primary growth-model parameters, Ratkowsky temperature-based modeling |
| **Data preparation** | Python, pandas, supplementary-table reconstruction, source-level flag and unit checks |
| **Machine learning and validation** | scikit-learn GPR and Random Forest, XGBoost, eight-fold LOSO, R²/MAE/RMSE, GPR intervals |
| **Interpretability** | SHAP TreeExplainer for full-data RF; descriptive feature attribution |
| **Exploration and visualisation** | Jupyter, Matplotlib, Plotly, Streamlit |

**Five environmental predictors:** temperature, culture medium, **added** NaCl (%), PCS concentration (%) and PCS type. Strain identity was withheld as a predictor; the held-out strain's observations were excluded from fitting. Ordinal encoding of categorical features remains a limitation.

## Repository guide

| Path | Contents |
|---|---|
| [`step1_data_extraction_eda.ipynb`](step1_data_extraction_eda.ipynb) | Data reconstruction, exploratory analysis and Ratkowsky-related analysis |
| [`step2_ml_modeling_v5.ipynb`](step2_ml_modeling_v5.ipynb) | Original ML workflow, LOSO and benchmark implementations; saved notebook outputs/narratives can be historical |
| [`step3_shap_analysis.ipynb`](step3_shap_analysis.ipynb) | RF feature-attribution analysis and plots; some historical inline outputs remain |
| [`streamlit_app.py`](streamlit_app.py) | Three-section, CSV-driven scientific case study |
| [`outputs/aeromonas_kinetics_master.csv`](outputs/aeromonas_kinetics_master.csv) | Corrected 158-condition experimental summary dataset |
| [`outputs/step2_loso_performance_summary.csv`](outputs/step2_loso_performance_summary.csv) | Revalidated per-model mean strain-fold metrics |
| [`outputs/step3_hurdle_ranking.csv`](outputs/step3_hurdle_ranking.csv) | Recalculated RF SHAP importance |
| [`audit/phase1/`](audit/phase1/REPORT.md) | Source reconciliation, discrepancy log and QA |
| [`audit/phase2/`](audit/phase2/REPORT.md) | Revalidated fold checks, out-of-fold predictions, intervals and caveats |
| [`DATA_PROVENANCE.md`](DATA_PROVENANCE.md) | Source conventions, PCS correction and unresolved unit/uncertainty notes |

**Figure provenance:** Use the dated Phase 2 audit and refreshed saved figures for current numerical claims. Older notebook inline results and some historical plots may not have been regenerated.

## Run the app locally

```bash
git clone https://github.com/hyejeong0617/aeromonas-growth-predictor.git
cd aeromonas-growth-predictor
python -m pip install -r requirements.txt
streamlit run streamlit_app.py
```

Launching the Streamlit case study only reads saved result tables; **do not rerun the notebooks to launch it**. The existing requirements also include optional research/ML dependencies and are not a locked environment. A clean hosted-environment installation should be checked before promoting the deployed URL.

## Scientific interpretation and limitations

- **Study scale:** a small, correlated set of laboratory conditions with eight strain groups; LOSO is internal validation, not an independent laboratory or food-product test.
- **Growth/no growth:** μmax regression uses only 118 observed-growth conditions; 40 NG conditions remain recorded but are not used to fit a growth/no-growth classifier.
- **Experimental conventions:** PCS is **purified condensed smoke**, with source doses 0.026%, 0.13% and author-confirmed **0.26%**. NaCl is **added salt**; in TSB total salt = added salt + 0.5 percentage points. SJ total salt is not inferred.
- **Original-source ambiguities:** the supplementary SD/SE terminology and Ymax concentration denominator are unresolved and have not been silently converted.
- **Model limitations:** ordinal category distances, small sample size and fold-based model selection constrain claims; SHAP attribution does not prove causality.
- **Not a food-safety decision tool:** no validated shelf-life estimates, inhibition thresholds, commercial optimization, regulatory advice or general population-level risk predictions are supported.

For correction provenance and reproducibility boundaries, see [Phase 1](audit/phase1/REPORT.md) and [Phase 2](audit/phase2/REPORT.md).

## Publication and data

**Lee H-J, et al. (2023).** *The effect of food processing factors on the growth kinetics of Aeromonas strains isolated from ready-to-eat seafood.* **International Journal of Food Microbiology**, 384, 109985. [DOI: 10.1016/j.ijfoodmicro.2022.109985](https://doi.org/10.1016/j.ijfoodmicro.2022.109985).

The dataset was reconstructed from the publication's supplementary Tables A–D. It is **not** a newly generated independent experimental dataset; see [source data and corrections](DATA_PROVENANCE.md).

## Related portfolio projects

- [AMR Genomics — *Aeromonas*](https://github.com/hyejeong0617/amr_genomics_aeromonas) — microbial genomics, AMR and scientific analysis
- [RASFF Risk Predictor](https://github.com/hyejeong0617/rasff_risk_predictor) — food safety notification data and machine learning
- [Foodborne Outbreaks EDA](https://github.com/hyejeong0617/foodborne_outbreaks_eda) — SQL and outbreak surveillance
- [EU Food Regulatory Agent](https://github.com/hyejeong0617/Regulatory_agent) — LLM/RAG and regulatory document investigation

---

**Scientific data science for food safety, biotechnology and applied AI**

[GitHub profile](https://github.com/hyejeong0617) · [LinkedIn](https://www.linkedin.com/in/hyejeong-lee-75887465/)
