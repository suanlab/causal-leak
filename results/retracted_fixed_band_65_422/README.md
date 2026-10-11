# Retracted results — fixed 65-422 Hz band

Every JSON here was computed with the `['month'] + bands_in_range(65, 422) +
MAX_COLS` feature set. That band was selected from permutation importance and
SHAP measured **on region 43**, then used for models evaluated **on region 43**
(AUDIT.md Blocker 1). The numbers are therefore optimistically biased by
model-selection leakage and must not be cited.

Superseded by runs against `band_selection.causal_feature_cols()`, whose bands
come from `run_nested_causal_bands.py` with the evaluation region excluded from
selection and a per-fold manifest proving it.

See `../FINDINGS_NESTED_BANDS.md`.
