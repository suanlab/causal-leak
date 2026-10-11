# causal-leak — Final Summary (Strengthened)

> **Superseded (October 2026).** The 121-422 Hz "causal band" reported below was
> selected using attribution measured on the evaluation fold and then evaluated
> on that same fold. It does not survive nested selection (chosen in 2 of 7
> regions). Numbers that depend on the 57-feature set are retracted; the
> archived artefacts are in `results/retracted_fixed_band_65_422/`. See
> `FINDINGS_NESTED_BANDS.md` and `FINDINGS_RQ4_RQ6_RERUN.md` for the corrected
> analysis.

## Best OOD Result (LORO Chungbuk, 5-class, 5,074 rows)

| Rank | Model | # feat | Macro-F1 | Acc |
|-----:|-------|------:|---------:|----:|
| **1** | **Pre-specified tree-only soft average (causal + hier)** | **57** | **0.4326** | 0.3096 |
| 2 | XGB hierarchical (normal gate + causal typer) | 57 | 0.4287 | 0.3041 |
| 3 | XGB causal 65–422 Hz (flat) | 57 | 0.4273 | 0.3013 |
| 4 | MLP GroupDRO | 57 | 0.3615 | 0.3210 |
| 5 | CNN + SpecAugment (validation-selected CNN recipe) | 534 | 0.3544 | 0.3384 |
| 6 | CNN ERM baseline | 534 | 0.3484 | 0.3494 |
| 7 | CNN + focal loss (accuracy-oriented recipe) | 534 | 0.3393 | 0.3648 |
| 8 | MLP IRM (validation-selected sweep λ=1) | 57 | 0.3145 | 0.2931 |
| 9 | PGA (best observed test ablation, λc=1) | 534 | 0.3041 | 0.2897 |

## Shortcut-Friendly Baselines (for contrast)

| Protocol | Acc | Note |
|----------|----:|------|
| Zero-ML sid lookup + lrate override | 0.9977 | 5 lines of Python |
| XGBoost full (paper split) | 0.9402 | 544 features |
| XGBoost full (random 5-fold CV) | 0.9449 | 544 features |

## Causal Band Evidence (Triple-Source Convergence)

| Band | Permutation Δf1 (mean, 4 regions) | Wins/4 | SHAP share | XGBoost Gain |
|------|:---------------------------------:|:------:|:----------:|:------------:|
| 10–121 Hz | +0.009 ± 0.002 | 4/0 | 8.5% | 2.5% |
| **121–422 Hz** | **+0.019 ± 0.017** | **3/1** | **40.4%** | 7.5% |
| 422–1470 Hz | −0.003 ± 0.013 | 2/2 | 28.4% | 22.9% |
| 1470–5120 Hz | +0.010 ± 0.005 | 4/0 | 22.7% | 67.1% |

121–422 Hz confirmed by permutation AND SHAP as the dominant
causal band. XGBoost gain is badly mis-allocated (67% on
irrelevant high-frequency range).

## Paper Stats

| Item | Count |
|------|------:|
| Pages | 19 |
| Tables | 10 |
| Figures | 8 |
| References | 21 |
| Python scripts | 21 |
| Result JSONs | 13 |
| LaTeX errors | 0 |

## Core Contributions

1. **Shortcut-audit protocol** — 4-rule lookup reproduces 99.77%
2. **Three-tier OOD evaluation** — Row-CV → GroupSite → LORO
3. **Causal 121–422 Hz band identification** — bootstrap CI + multi-region + SHAP convergence
4. **422–1470 Hz active confounder** — permuting improves OOD F1
5. **Causal feature selection > invariance learning** — IRM/GroupDRO/PGA all fail to beat 57-feature XGBoost
6. **SpecAugment recipe** — CNN F1 0.334 → 0.354
7. **PGA negative result** — soft-prior injection < hard feature selection
8. **Tree ensemble deployment candidate** — pre-specified F1 0.4326, perfect normal recall
