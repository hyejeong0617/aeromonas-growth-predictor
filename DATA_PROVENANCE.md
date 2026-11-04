# Phase 1 data provenance — 2026-11-04

Canonical generation: `step1_data_extraction_eda.ipynb`, zero-based cells 4–9 and 11. Cell 8 reconciles the original table literals with `data/supplementary_records.json`. Both files are required; the JSON is a reviewed source transcription, not a model-generated dataset. Run from the repository directory. Full notebook execution also performs fitting and is outside Phase 1.

Source: Lee et al. (2023), DOI [10.1016/j.ijfoodmicro.2022.109985](https://doi.org/10.1016/j.ijfoodmicro.2022.109985), Supplementary Tables A–D, supplied as `1-s2.0-S0168160522004573-mmc1 (1).docx`. Original file remains unchanged in the user's Downloads directory. SHA256: `45a1385dbb59531755a6372969c18b983fb168f2450b554d72eba574e88db7a9`.

[Main article manuscript](https://ntnuopen.ntnu.no/ntnu-xmlui/bitstream/handle/11250/3047433/1-s2.0-S0168160522004573-main.pdf?isAllowed=y&sequence=1): sections 2.3.3 (PCS), 2.6 (kinetics), 3.1.2 (salt), and 3.1.3 (PCS response).

## Definitions and explicit correction

- PCS means **purified condensed smoke**. VTABB and JJT01 are not bacteriocins.
- The author explicitly confirmed in the Phase 1 request that the high-dose treatment was **0.26%**, overriding the published **2.6%** row labels in Tables C/D. All 16 corresponding records use 0.26%; the raw labels and correction reason remain in provenance fields. This is an author-confirmed correction, not an independently published erratum. It does not establish an EU legal limit.
- `mu_max` retains the source numeric scale, interpreted as decimal-log concentration increase per hour: **log10 CFU/h**, not natural-log specific rate h−1. No targets were multiplied or divided by ln(10). Ratkowsky uses the square root of this rate; ML targets use this rate; SHAP contributions have the target's rate units. For a log10 concentration difference, exponential-phase duration is `delta_log10 / mu_max`; with an assumed lag, `t = lag_h + delta_log10 / mu_max`. This approximation is not a validated food-safety prediction.
- `NaCl_pct` means **added NaCl**. In TSB only, `NaCl_total_TSB_pct = NaCl_pct + 0.5`: 0/1.5/3/4.5/6% added maps to 0.5/2/3.5/5/6.5% total. No basal-salt addition is applied to SJ; its total-salt field remains missing. The numerical model feature is unchanged.
- NG means no growth: kinetic estimates remain missing, not zero. NL means no lag: lag is encoded as zero with the original NL flag preserved. NA means no asymptote: Ymax remains missing, not zero. Unreported combinations are not synthesized as NG.
- Original `species` identifiers remain for compatibility. `species_name` separately represents seven species across eight strains, including two A. salmonicida strains. Source metadata are not additional predictor features.

## Unresolved source ambiguities

Table A labels parameter ± values SD; Tables B–D label them SE, whereas the main article describes SD. Values and table labels are preserved without conversion. The separate final `se` column is captioned standard error of R² and must not be confused with parameter ± uncertainties.

All supplementary Ymax headers say log CFU/g although these tables describe liquid media; the main article uses CFU/ml for liquids. Source header text and numeric Ymax values are preserved. No density conversion or silent unit reinterpretation was performed. These two matters need author clarification before uncertainty weighting or unit-dependent Ymax applications.

## Verification boundary

The repaired master contains 158 reported conditions: A=48, B=42, C=34, D=34; Y=110, NL=8, NG=40. There are 118 growth-positive records and 108 available Ymax estimates. Verification covers transcription and explicit author correction, not independent reproduction of the original biological experiments or fresh model validation.

See `audit/phase1/REPORT.md`, the row-level before/after comparison CSVs, raw extraction, and `qa_results.json`. Existing fitted models, validation metrics and SHAP artifacts remain historical and require Phase 2 approval and regeneration.
