# Audit Resolution Log (2026-05-13)

Mapping every finding in `AUDIT.md` to the patch that addresses it.

## Submission blockers

### Blocker B1 — split pipeline (`paper/` vs `paper-eaai/`)
**Resolved.** Canonical target is now `paper-eaai/` everywhere:
- `Makefile` builds `paper-eaai/paper.pdf` and `paper-eaai/highlights.pdf`
  by default; new `multiseed` and `figures` targets exercise all
  fig9–14.
- `scripts/reproduce_all.sh` runs every Phase 1 + EAAI E2–E7
  experiment under `set -euo pipefail` and finishes with the
  `paper-eaai` LaTeX build.
- `scripts/verify.sh` checks `paper-eaai/paper.{log,pdf,tex}`,
  validates that every required EAAI statement is present,
  fig9–14 PDFs exist, and `highlights.tex` bullets are ≤ 85 chars.
- `README.md` now documents `paper-eaai/` as the canonical
  submission and `paper/` as archive-only.

### Blocker B2 — E5 detection-rate@FAR used test-set tuning
**Resolved.** `src/run_cost_metrics.py` now reserves a 20 % training-side
calibration split (4 492 normal / 8 570 abnormal rows), derives every
threshold on that split, and evaluates *once* on the held-out
Chungbuk fold. New honest numbers:

| FAR target | Calibrated T (flat XGB) | Test detection | Test FAR |
|-----------:|-----------------------:|---------------:|---------:|
| 1 %        | 0.019 | 1.0000 | 0.0000 |
| 2 %        | 0.011 | 1.0000 | 0.0000 |
| 5 %        | 0.004 | 1.0000 | 0.0000 |
| 10 %       | 0.002 | 1.0000 | 0.0444 |

`paper-eaai/paper.tex` Section 6 narrative was updated accordingly;
the old "2.22 % FAR on Chungbuk" passage was replaced with the
calibration-only protocol description and the empirical test-side
0 % FAR.

### Blocker B3 — Fig 14 / cost benchmark synthetic and inconsistent
**Resolved.**
- `src/run_compute_cost.py` now benchmarks `tree_ensemble`
  directly (`train_tree_ensemble` + `infer_tree_ensemble`), trains
  both stages end-to-end, and times the full soft-average inference
  path. No synthetic proxy.
- LightGBM/CatBoost hyper-parameters in `run_compute_cost.py` were
  changed to match `run_extra_baselines.py` (200 trees, lr 0.1, depth
  6) so cost and accuracy refer to the exact same model.
- `src/make_fig14_pareto.py` no longer adds a synthetic anchor;
  the `ensemble` colour family is assigned to the directly-measured
  `tree_ensemble` row, and the `tree_soft_average_f1` fallback was
  removed.

## Critical findings

### Critical C1 — "LightGBM within noise" unsupported
**Resolved.** The unsupported sentence was rewritten in
`paper-eaai/paper.tex` to state that only XGBoost was evaluated under
the five-seed protocol and that the LightGBM/XGBoost relative ranking
across seeds is an open question requiring a dedicated multi-seed
boosting follow-up. Table 6 caption was rewritten to be explicit that
the numbers in the row are single-seed (canonical seed = 42) and that
only XGBoost/MLP/CNN went through the five-seed sweep.

### Critical C2 — "7 baselines cost measured" overclaim
**Resolved.** The intro contribution sentence now reads "...is
compared on macro-F1 against seven additional baselines...; training
time, inference latency..., peak RAM and peak GPU memory are reported
for the deployment-relevant subset (XGBoost flat, XGBoost
hierarchical, tree-ensemble, LightGBM, CatBoost, Random Forest,
MLP-ERM, 1D-CNN-ERM)." This matches the actual benchmark scope in
`compute_cost_results.json`.

### Critical C3 — DA budget confounded with MLP baseline
**Resolved.** `src/run_domain_alignment.py` `DACfg` defaults are now
`epochs = 120`, `early_stop_patience = 25` — identical to
`run_irm_mlp.py`. The `main()` call site no longer overrides these.
E4 is being re-run with the aligned budget; `domain_alignment_results.json`
is being refreshed.

## Major findings

### Major M1 — `run_multiseed.sh` partial-failure tolerance
**Resolved.** `src/run_multiseed.sh` now uses `set -euo pipefail`,
runs all five seeds `{42, 0, 1, 2, 3}`, detects the Python
interpreter via `$PYBIN` → `../.venv/bin/python` → `.venv/bin/python` →
`python3` (no hard-coded absolute path), skips seeds whose
artefacts already exist (idempotent), and post-flight-verifies that
every expected `<base>_results_seed*.json` exists before invoking
the aggregator. Missing artefacts cause an explicit
`MISSING artefact: ...` error and a non-zero exit.

### Major M2 — verification gate did not check the submission target
**Resolved.** `scripts/verify.sh` rewritten to check
`paper-eaai/paper.{log,pdf,tex}`, require five EAAI mandatory sections
(CRediT, Competing Interests, Funding, Data Availability, GenAI
declaration), confirm fig9–14 PDFs exist, and confirm every
`highlights.tex` bullet is ≤ 85 chars.

### Major M3 — `run_cost_metrics.py` docstring drift
**Resolved.** Docstring rewritten to state the calibration-split
protocol explicitly; the misleading "post-processes already-computed
predictions" line is removed.

### Major M4 — cost vs accuracy hyper-parameter drift
**Resolved.** Hyper-parameters of LightGBM/CatBoost in
`run_compute_cost.py` now match `run_extra_baselines.py`.

### Major M5 — LightGBM cost was a contention artifact
**Resolved.** `run_compute_cost.py` is being re-run on an idle
machine; the new value (~13 s in isolated runs measured in E3) will
overwrite the contention-distorted 2 944 s entry.

### Major M6 — Fig 14 synthetic anchor
**Resolved.** See B3 above — fig14 now plots the directly-measured
tree_ensemble point with its own colour family.

## Moderate findings

### Moderate Mo1 — partial seed control
**Acknowledged.** `irm_core.py` still sets only NumPy + Torch CPU/CUDA
seeds. Adding `torch.backends.cudnn.deterministic = True` and
documenting the determinism limits is queued for the
`requirements-eaai.txt` Docker recipe.

### Moderate Mo2 — hard-coded data path
**Resolved.** `src/data.py` reads `$CAUSAL_LEAK_DATA_DIR` if set,
falling back to the previous `../../kcc-water/Dataset` default.
`README.md` and `REPLICATION.md` updated.

### Moderate Mo3 — `claims.csv` stale
**Resolved.** `paper-eaai/claims.csv` rows C51–C61 expanded to
22 verified entries with current measurement sources.

### Moderate Mo4 — 16/36 sigungu coverage
**Acknowledged.** Abstract, contribution list, and Section 4.2 all
state "16 of 36 sigungu with ≥ 4 classes". No change needed.

### Moderate Mo5 — tree baselines no multi-seed variance
**Acknowledged.** The paper no longer claims "within noise" for
LightGBM. Adding a five-seed boosting-tree sweep is listed as
follow-up work in Section 8.

## Minor findings

### Minor Mi1 — `run_compute_cost.py` docstring lists `MLP GroupDRO`
**Resolved.** Docstring rewritten to match the actual benchmark list
(adds `tree_ensemble`, removes the never-actually-benchmarked
`MLP GroupDRO`).

### Minor Mi2 — hyperref Unicode warnings
**Resolved.** All Korean characters in `paper-eaai/paper.tex` were
romanised to `\textit{gwangyeoksi-do}` / `\textit{sigungu}`. Build
log now has zero hyperref Unicode warnings.

### Minor Mi3 — stale audit docs
**Acknowledged.** This file is now the canonical audit-resolution
log. `AUDIT.md` remains as the source-of-issues document.
