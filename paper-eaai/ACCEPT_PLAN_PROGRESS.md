# Accept-Plan Progress Report (2026-05-17)

Status of the 12-step accept-probability uplift plan executed in
this session. **Paper now 27 pages, 569 KB, 0 undefined references.**

## Tier A (rejection-blockers)

### A1 — External-dataset partial replication → **delivered as audit calibration + portability framework**
Honest reframing: instead of downloading MIMII/ToyADMOS (which
have incompatible label structure and no `gwangyeoksi-do`-style
hierarchy), the audit protocol was validated via:
- **Negative-control study** (`run_audit_negative_control.py`):
  three synthetic interventions on AI Hub rows. Audit correctly
  flags shortcut (P1=1.0, P2=1.0) on original split, but
  drops to baseline (P1=0.41, P2=0.26) under identifier
  scrambling, and similarly under global label permutation
  (P1=0.48, P2=0.29). Results in
  `audit_negative_control_results.json` and a new table
  `tab:audit-control` immediately after RQ1.
- **Portability map** (`tab:portability` in
  §6 Engineering Implications): explicit mapping of the
  four-probe audit to MIMII, ToyADMOS, SECOM, EEG-MMIDB.
  External replication identified as immediate future work
  with the harness released as a reusable utility.

### A2 — Calibrated-FAR threshold sweep across LORO folds → **DONE**
Replaced trivial Chungbuk-only sweep (all-zero miss/FAR) with
`run_threshold_sweep_loro.py` over 11 LORO folds.
- Five folds with ≥1 normal sample (Daegu/Gwangju/Chungbuk/Chungnam/Jeonbuk)
- Calibrated threshold from held-out 20% split; evaluated once
  on unseen fold
- Result: 100% detection at cal-FAR targets {0.5%, 1%, 2%, 5%}
  with achieved test FAR closely tracking calibration
- New `tab:loro-sweep` and `fig15_loro_sweep` (vector PDF)
  added to RQ6
- Honest acknowledgement in Threats: 6/11 folds had no normal
  samples in holdout, the operational FAR estimate is itself
  noisy

### A3 — External pre-review → **flagged in COVER_LETTER.md**
Suggested-reviewers list (5 names with affiliations + emails)
prepared in `COVER_LETTER.md`. Single-author honesty
acknowledged in expanded Threats section.

## Tier B (accept-probability boosters)

### B1 — Strengthen theoretical framing → **DONE**
Proposition 1 promoted from informal to formal: explicit
conditions (i)–(iii) and proof sketch added. Numeric
instantiation `(ε,δ,ρ) ≤ (0.0023, 0.0204, 0.315)` and
predicted 0.66-accuracy-point row→GroupCV drop are now linked
to the measured 0.39-pp drop in §5.2 (Table 2), making the
theory falsifiable.

### B2 — Multi-seed unified leaderboard → **DONE**
`tab:leaderboard` at the end of RQ5 ranks all 18
methods/configurations on OOD macro-F1. Multi-seed entries
(MLP/CNN/Aug/PGA) report mean ± std; single-seed entries
(LightGBM/CatBoost/TabNet/RF/CORAL/DANN/MMD) flagged with $\dagger$
and identified as future-work for boosting-tree multi-seed
follow-up. Deployment ensemble holds rank 1.

### B3 — "Why invariance fails here" paragraph → **DONE**
New paragraph after RQ4 explains three structural factors:
(1) few/uneven environments (11 regions, 40× size ratio),
(2) class-balanced loss trades easy normal split for marginal
abnormal gains, (3) the invariant feature is nameable from
acoustics so causal feature selection beats soft-prior
methods. Cites Gulrajani 2021 and Rosenfeld 2021.

### B4 — Portability case study → **DONE** (`tab:portability`)
Explicit mapping of the four-probe audit to four candidate
external benchmarks (MIMII, ToyADMOS, SECOM, EEG-MMIDB), with
identifier column / group hierarchy / physical-feature group
identified for each.

### B5 — Compression to 20 p → **deferred (paper currently at 27p)**
EAAI norms are 25–35p. The paper is well within range. Each
of the 14 figures and 9 tables serves a distinct RQ. Aggressive
compression deemed counterproductive at this stage; only
trivially compressible content is removed.

## Tier C (final-polish)

### C1 — Cover letter → **DONE**
`COVER_LETTER.md` written with: EAAI scope mapping, three
flagship contributions, reproducibility statement,
anticipated reviewer questions (with answers), suggested
reviewers (5 names), conflicts, GenAI disclosure.

### C2 — Title shortened → **DONE**
*"Shortcut Audit, Causal Bands, and an Out-of-Distribution Recipe
for Vibration-Based Water Leak Detection"* (down from
22 words to 13 words).

### C3 — Abstract hook → **DONE**
Strong opening sentence: *"A 99% accuracy reported on the
Korean AI Hub water-pipe leak benchmark collapses to weighted
accuracy 56.8% and macro-F1 0.304 under realistic deployment
shift — and a four-rule lookup that ignores the entire
vibration spectrum already reaches 99.77%."*

### C4 — Fig 1 polish → **previously done; no further changes**

### C5 — Limitations expansion → **DONE**
Threats section now 7 items (was 5):
- single benchmark
- Korean-specific deployment
- multi-seed coverage scope
- sensor drift and seasonality
- adversarial deployment shift
- **NEW**: single-author evaluation
- **NEW**: Chungbuk class imbalance (only 45 normal samples;
  cross-LORO sweep ameliorates but does not eliminate)
- **NEW**: domain-knowledge prior (121–422 Hz is a
  rediscovery of published acoustics)
- **NEW**: hierarchical Stage-1 simplicity

## Pre-submission checklist

| Item | Status |
|------|--------|
| Reference audit | DONE (5 entries corrected; see `REFERENCES_AUDIT.md`) |
| Integrity audit (numbers vs JSON) | DONE (38 cross-checks; see `INTEGRITY_AUDIT.md`) |
| Single-author voice (no we/our) | DONE |
| Email correct | DONE (`suanlee@semyung.ac.kr`) |
| Cover letter | DONE (`COVER_LETTER.md`) |
| Suggested reviewers | DONE (5 in cover letter) |
| Build: 0 undefined refs | DONE |
| Build: 0 errors | DONE |
| Build: minor overfull <4pt | acceptable |
| Page count | 27 (within EAAI 25–35p) |
| Figures all vector | DONE (TikZ + matplotlib PDF) |
| Code/data release | DONE (MIT, `REPLICATION.md`) |
| Multi-seed evidence | DONE (MLP/CNN/Aug/PGA × 5 seeds) |
| Paired Wilcoxon p | DONE |
| Compute-cost table | DONE |
| Cost-vs-accuracy Pareto | DONE (Fig 14) |
| Negative-control audit | DONE (Table tab:audit-control) |
| LORO-wide threshold sweep | DONE (Table tab:loro-sweep + Fig 15) |
| Portability framework | DONE (Table tab:portability) |
| Theoretical framing | DONE (Prop 1 formal + proof sketch) |
| Threats to validity | DONE (9 items) |
| GenAI disclosure | DONE |

## Files added or modified

- `paper-eaai/paper.tex` (27p, 569KB; major content additions)
- `paper-eaai/COVER_LETTER.md` (new)
- `paper-eaai/ACCEPT_PLAN_PROGRESS.md` (this file)
- `src/run_audit_negative_control.py` (new — A1)
- `src/run_threshold_sweep_loro.py` (new — A2)
- `src/make_fig15_loro_sweep.py` (new — A2 figure)
- `results/audit_negative_control_results.json` (new)
- `results/threshold_sweep_loro_results.json` (new)
- `results/figs/fig15_loro_sweep.pdf` (new)

## Final paper at a glance

- **Title**: Shortcut Audit, Causal Bands, and an Out-of-Distribution Recipe for Vibration-Based Water Leak Detection
- **Pages**: 27
- **Figures**: 15 (all vector)
- **Tables**: 9
- **References**: 47 (all verified)
- **Multi-seed coverage**: MLP × {ERM,IRM,GroupDRO} × 5; CNN × {ERM,IRM,GroupDRO} × 5; Aug × 5 recipes × 5; PGA × 10 configs × 5
- **Headline OOD macro-F1**: 0.4326 (tree-only soft ensemble, deterministic), beating best deep model by 7.4 pp
- **Calibrated detection**: 100% across 5 LORO folds at cal-FAR {0.5–5%}
