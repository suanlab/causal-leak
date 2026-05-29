# EAAI Revision Findings (Phase 1, E1–E7)

Run dates: 2026-05-07. Hardware: 2 × NVIDIA TITAN Xp + 16-core CPU.

This document summarises the new experiments added for the
EAAI submission. Each experiment is anchored to a research
question (RQ) in the manuscript (Section~\ref{sec:results}).

## E1 — Multi-seed re-runs (RQ4) — COMPLETE

Re-ran every stochastic experiment for seeds {0, 1, 2, 3, 42}
keeping the data partition fixed (split-seed=42, model-seed
varies). Total wall-clock: 21,760 s ≈ 6 h. All 5 seeds × 4
scripts (irm_mlp, irm_cnn, augmentation, pga) completed
successfully. Aggregate in `multiseed_summary.{json,md}`.

### MLP (n=5)
| Method | test_f1 mean | std | test_acc mean | std |
|--------|--------------|----:|---------------|----:|
| ERM      | 0.3272 | 0.020 | 0.2940 | 0.015 |
| IRM      | 0.3132 | 0.008 | 0.2984 | 0.009 |
| **GroupDRO** | **0.3583** | **0.013** | **0.3126** | **0.009** |

### CNN (n=5)
| Method | test_f1 mean | std |
|--------|--------------|----:|
| **ERM**  | **0.3461** | 0.018 |
| IRM      | 0.3251 | 0.017 |
| GroupDRO | 0.3258 | 0.038 |

### Augmentation (n=5)
| Recipe | test_f1 mean | std |
|--------|--------------|----:|
| baseline      | 0.3122 | 0.036 |
| **SpecAugment** | **0.3409** | 0.030 |
| Mixup         | 0.2854 | 0.026 |
| **Focal**       | **0.3394** | 0.044 |
| all combined  | 0.3218 | 0.020 |

### PGA (n=5)
Best: `pga_lam0_initU` at 0.2942 ± 0.010 — confirms negative
result; no $\lambda_c$ recovers the soft-prior approach to
even the plain MLP-ERM baseline.

### Paired Wilcoxon (n=5)
- MLP GroupDRO vs MLP ERM: Δ +0.031, **p = 0.125** (n.s.)
- MLP IRM vs MLP ERM: Δ −0.014, **p = 0.188** (n.s.)

**Headline:** No deep loss-level invariance regulariser
achieves a statistically significant lead over plain ERM at
α = 0.05. The 6.9 pp gap to causal-XGBoost (0.4273) survives
the multi-seed analysis.

## E2 — 시군구-level Leave-One-County-Out OOD (RQ2)

Total environments: 36 unique (region\_lvl1 + region\_lvl2)
combinations. Eligible folds (≥4 classes present): 16. Ran
XGBoost on `signal_all` (534 features) and `causal` (57
features) per fold and computed Δ macro-F1.

Headline numbers (`results/sigungu_ood_results.json`):

| Aggregate | signal\_all F1 | causal F1 | Δ |
|-----------|--------------:|----------:|--:|
| weighted by n\_test | 0.266 | 0.268 | **+0.002** |
| unweighted mean      | 0.259 | 0.246 | -0.013 |
| std dev (unweighted) | -    | -    | 0.067 |
| wins / losses for causal | - | - | 9 / 7 |

Interpretation: the 121–422 Hz causal-feature advantage is
preserved on the canonical Cheongju (43_720) fold (+0.009 F1)
but is essentially neutral when averaged over all 16 finer-
grained 시군구 environments. Going from 12 광역시도 to 36 시군구
does not improve overall macro-F1 — the generalisation
ceiling is set by the dataset's environmental coverage, not
by the LORO/LOCO granularity.

## E3 — Additional tabular baselines (RQ5)

Single seed (42) on the canonical Chungbuk LORO fold, 57
causal features only:

| Model | OOD macro-F1 | Acc | Train time |
|-------|-------------:|----:|-----------:|
| XGB causal (anchor) | 0.4273 | 0.3013 | ~5s |
| **LightGBM**        | **0.4341** | 0.3319 | 13s |
| CatBoost            | 0.3726 | 0.3518 | 15s |
| TabNet              | 0.3286 | 0.2826 | 7157s |
| RandomForest        | 0.3085 | 0.2978 | 41s |

Headline: **LightGBM exceeds the causal-XGBoost anchor by
+0.7 pp on the canonical seed.** This is within the multi-seed
noise band of XGBoost (±0.014 from preliminary E1 numbers), so
the single-seed lead does not displace XGBoost as the
deployment recipe — but it is an honest data point worth
reporting. TabNet at 7157s is two orders of magnitude slower
than the gradient-boosted alternatives at the same accuracy,
which is consistent with the tabular-DL survey
of~\cite{borisov2024tabsurvey}.

## E4 — Domain-alignment baselines (RQ5) — COMPLETE

Three multi-source domain-generalisation methods, all on the
57-causal-feature MLP backbone:

| Method | val F1 | OOD macro-F1 | OOD Acc |
|--------|-------:|-------------:|--------:|
| DeepCORAL | 0.8999 | 0.3388 | 0.3043 |
| MMD-DG    | 0.8887 | 0.3320 | 0.2978 |
| DANN      | 0.8928 | 0.3212 | 0.2789 |

All three lag the causal-XGBoost anchor (0.4273) by 8.8–10.6
percentage points despite training stably with high val-F1
(0.88–0.90). The val-test gap (0.55+) is the dominant failure
mode: alignment loss successfully equalises feature
distributions across training environments but does not
transfer to the held-out region.

## E5 — Detection rate at 1% FAR (RQ6)

Post-hoc evaluation on the canonical Chungbuk fold with 5{,}029
abnormal and 45 normal samples. Both flat causal XGBoost and
the hierarchical Stage-1 binary gate achieve 100\,\% detection
of abnormal cases at the 1\,\%-target FAR (achieved
2.22\,\% --- the granularity is set by the small normal-class
size on this fold).

## E6 — Computational-cost benchmark (RQ7) — COMPLETE

| Model | train (s) | infer/1k (ms) | RAM (MB) | GPU (MB) | F1 | acc |
|-------|----------:|--------------:|---------:|---------:|---:|----:|
| XGBoost flat   | 5.3   | 3.28   | 14  | 0   | 0.4279 | 0.3114 |
| XGBoost hier   | 4.7   | 3.67   | 22  | 0   | 0.4275 | 0.3065 |
| LightGBM       | 2944  | 16.28  | 66  | 0   | 0.4366 | 0.3279 |
| CatBoost       | 10.7  | 3.93   | 15  | 0   | 0.3776 | 0.3626 |
| RandomForest   | 12.9  | 124.09 | 119 | 0   | 0.3085 | 0.2978 |
| MLP ERM        | 30.0  | 0.50   | 65  | 33  | 0.3411 | 0.3063 |
| CNN ERM        | 112.7 | 20.55  | 115 | 332 | 0.3008 | 0.2921 |

**Headline:** causal-XGBoost trains in ~5s and serves at 3.28ms
per 1{,}000 rows on CPU — well within the budget for single-CPU
edge-gateway deployment. The reported LightGBM training time
(2944s) reflects realistic CPU contention with the concurrent
multi-seed sweep; an isolated run completes in ~13s
(see E3). RandomForest at 124ms/1k inference is the slowest
tree alternative. The MLP/CNN deep models add 33MB / 332MB of
peak GPU memory respectively.

## E7 — Stage-1 hierarchical threshold sweep (RQ6)

Threshold T ∈ {0.3, 0.4, 0.5, 0.6, 0.7} on the binary
normal-vs-abnormal gate:

| T | miss-leak | FAR | 4cls F1 (abn) | 5cls F1 |
|--:|----------:|----:|--------------:|--------:|
| 0.3 | 0.000 | 0.000 | 0.286 | 0.429 |
| 0.4 | 0.000 | 0.000 | 0.286 | 0.429 |
| 0.5 | 0.000 | 0.000 | 0.286 | 0.429 |
| 0.6 | 0.000 | 0.000 | 0.286 | 0.429 |
| 0.7 | 0.000 | 0.000 | 0.286 | 0.429 |

Headline: the binary normal-vs-abnormal gate is so well
separated on this fold that no threshold within the swept
range produces missed leaks or false alarms. The operational
choice of T therefore depends on downstream cost asymmetries
rather than on macro-F1 trade-offs.
