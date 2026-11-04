# Streamlit portfolio redesign — 2026-11-04

Implemented a three-section English scientific case study: original research, ML approach, and results/insights. Interactive observed-rate filtering, an illustrative LOSO diagram, saved model comparison, held-out predictions, refreshed SHAP ranking and saved GPR intervals replace the former simulator and optimization interface.

## Files

- `streamlit_app.py`: CSV-only presentation; lightweight cached reads; no fitting, SHAP execution, model deserialization or prediction engine.
- `.streamlit/config.toml`: restrained light theme; no new dependencies.
- `audit/app_redesign/`: previous app backup, protected-file hashes, UI check script/log, QA results and screenshot.

Original uncommitted app preserved as `streamlit_app.before.py`. All existing Phase 1/2 data, outputs, models, notebooks, README and provenance files match pre-redesign hashes.

## Checks performed

- Successful local startup at http://127.0.0.1:8502, with browser-rendered research, ML and results sections. The initial sandboxed server could not start; the local server succeeded outside the sandbox.
- Streamlit AppTest: all three sections; empty temperature filter; PCS treatment filter; held-out-group slider; GPR/RF/XGBoost selections; simulated missing optional SHAP CSV. Passed after fixing the test harness's absolute app path.
- Mean-fold μmax R² matches Phase 2: GPR 0.89538, RF 0.87156, XGBoost 0.83574. Pooled OOF metrics remain separately labelled. GPR coverage displays 112/118 (94.92%).
- NG values remain missing and excluded from the measured-rate plot; NL retains measured rates. Displayed PCS concentrations and added-NaCl definitions match the corrected data.
- Source AST/import inspection confirms no fit or model-loading calls. Unsupported former calculator/optimizer and incorrect treatment terminology are absent from the active app.
- Browser review at desktop 1280×900 and mobile 390×844. Navigation wraps, controls stack, charts resize; no horizontal document overflow at mobile width. This is viewport simulation, not testing on physical phones.
- DOI/GitHub hrefs checked. External fetch attempts were inconclusive; remote availability is not certified.
- `git diff --check` passed. No training, LOSO, SHAP recalculation, deployment, commit or push.

## Readiness and limits

Ready for local design review. Preview: http://127.0.0.1:8502. Deploy only after approval and a hosting-specific smoke test. The existing research requirements file is unchanged and still includes training packages, although this app imports only Streamlit, pandas and Plotly. No clean-host installation test was performed. Installed Streamlit emits nonfatal deprecation notices for its backward-compatible container-width argument. Recheck external links before publication.

Scientific limitations remain explicit in the interface: small correlated data, growth-positive regression, internal LOSO only, no independent product validation, noncausal SHAP and no validated shelf-life/regulatory decision support.
