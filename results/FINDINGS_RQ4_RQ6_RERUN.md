# RQ4–RQ7 Re-run Under Leakage-Free Band Selection

Every experiment whose feature set came from the retracted 65–422 Hz subset has
been re-run against `band_selection.causal_feature_cols()`, which resolves the
bands chosen for the evaluation region without that region entering selection.
The canonical fold is unchanged (region 43 held out, `SPLIT_SEED = 42`); only
the feature set differs, from **57 columns to 503**.

Retracted artefacts are preserved in `results/retracted_fixed_band_65_422/`.
Protocol and band evidence: `FINDINGS_NESTED_BANDS.md`.

## RQ4 — Invariance regularisers (region 43, macro-F1)

| Model | Retracted (57 f) | Leakage-free (503 f) | Δ |
|---|---:|---:|---:|
| MLP ERM (5 seeds) | 0.3272 ± 0.020 | 0.2887 ± 0.018 | −0.0385 |
| MLP IRM (5 seeds) | 0.3132 ± 0.008 | 0.3030 ± 0.009 | −0.0102 |
| **MLP GroupDRO (5 seeds)** | **0.3583 ± 0.013** | **0.3115 ± 0.017** | **−0.0468** |
| IRM λ-sweep, best | 0.3372 (λ=1000) | 0.3128 (λ=100) | −0.0244 |
| CNN ERM (5 seeds) | 0.3461 ± 0.018 | 0.3455 ± 0.027 | −0.0006 |
| CNN IRM (5 seeds) | 0.3251 ± 0.017 | 0.3268 ± 0.025 | +0.0017 |
| CNN GroupDRO (5 seeds) | 0.3258 ± 0.038 | 0.3429 ± 0.007 | +0.0171 |

The MLP family, which consumed the causal feature set directly, loses across the
board; the CNN family, which reads the full 534-band spectrum and is therefore
untouched by band selection, moves only within seed noise. That split is a
useful internal control: the fix moved what it should and left alone what it
should.

The published headline for deep models — MLP GroupDRO at 0.358 — was inflated by
the leaked band. Its honest value is **0.3115**, and the best deep model on this
fold is now CNN ERM (0.3455), not an invariance-regularised MLP.

## RQ5 — Additional baselines (region 43, macro-F1)

| Model | Retracted | Leakage-free | Δ |
|---|---:|---:|---:|
| **LightGBM (5 seeds)** | 0.4341 ± 0.000 | **0.4577 ± 0.000** | **+0.0236** |
| CatBoost (5 seeds) | 0.3699 ± 0.004 | 0.3562 ± 0.005 | −0.0137 |
| Random Forest (5 seeds) | 0.3052 ± 0.003 | 0.2652 ± 0.002 | −0.0400 |
| TabNet, causal features (5 seeds) | 0.3133 ± 0.011 | 0.3203 ± 0.021 | +0.0070 |
| TabNet, full spectrum (5 seeds) | 0.2845 ± 0.039 | 0.2845 ± 0.039 | ±0.0000 |

## RQ5 — Domain alignment (region 43, best λ per method)

| Method | Retracted | Leakage-free | Δ |
|---|---:|---:|---:|
| DeepCORAL | 0.3723 (λ=1.0) | 0.3100 (λ=1.0) | −0.0623 |
| DANN | 0.3152 (λ=0.5) | 0.2761 (λ=0.5) | −0.0391 |
| MMD-DG | 0.3216 (λ=1.0) | 0.3068 (λ=1.0) | −0.0148 |

Every alignment method loses. Combined with the MLP results, the gap between
gradient-based invariance and plain tree ensembles **widens** under honest
features: the paper's claim that feature selection beats loss-level invariance
is not merely intact, it is stronger than reported.

## RQ6 — Hierarchical classifier and ensemble (region 43, macro-F1)

| Configuration | Retracted | Leakage-free | Δ |
|---|---:|---:|---:|
| XGBoost flat | 0.4273 | 0.4363 | +0.0090 |
| XGBoost hierarchical | 0.4287 | 0.4356 | +0.0069 |
| — stage-1 binary F1(pos) | 1.0000 | 1.0000 | ±0.0000 |
| **Tree-only soft average** | **0.4326** | **0.4388** | **+0.0062** |
| Soft average incl. CNN | 0.3633 | 0.3903 | +0.0270 |
| CNN (validation-selected) | 0.3544 | 0.3892 | +0.0349 |

The deployment candidate survives and improves. The CNN rows moved because CNN
training is not bit-reproducible on GPU across re-runs, not because of the band
change — those paths never used the causal feature set.

### The ranking has changed, and it matters

**LightGBM on the same leakage-free features reaches 0.4577, beating the
pre-specified tree-only ensemble at 0.4388 by +0.019.** Under the retracted
protocol the ensemble (0.4326) narrowly led LightGBM (0.4341) — in fact it did
not, and the ordering was already marginal. It is now clearly reversed.

The manuscript cannot continue to present the soft-average ensemble as *the*
deployment recipe on performance grounds. Either LightGBM becomes the
recommendation, or the ensemble is retained for a stated non-performance reason
(the ensemble has perfect out-of-distribution `normal` recall through its
stage-1 gate, which matters operationally for false-alarm control) and the
performance cost is acknowledged.

## RQ7 — Computational cost

| Model | Train (s) | Inference (ms / 1k rows) | Peak RAM (MB) |
|---|---:|---:|---:|
| XGBoost flat | 5.0 → 31.6 | 3.12 → 4.71 | 13.7 → 113.7 |
| XGBoost hierarchical | 4.1 → 29.7 | 3.54 → 9.99 | 22.1 → 188.4 |
| **Tree ensemble** | **8.3 → 59.9** | **6.84 → 9.80** | **22.1 → 188.4** |
| CatBoost | 5.8 → 19.9 | 3.14 → 8.65 | 14.8 → 114.8 |
| Random Forest | 12.3 → 17.8 | 117.84 → 123.84 | 118.7 → 216.8 |
| MLP ERM | 21.0 → 20.1 | 0.28 → 3.11 | 65.2 → 165.2 |
| CNN ERM | 42.7 → 42.0 | 10.84 → 12.30 | 115.3 → 115.3 |

The published deployment figures — 8 s training and 6.84 ms per 1,000 rows on
57 features — become **60 s and 9.80 ms on 503 features**, with peak RAM 8.5×
higher. In absolute terms the recipe is still cheap enough for a utility to run
on commodity hardware, but the "57 features" selling point is gone and the
efficiency argument must be restated in absolute rather than comparative terms.

One caveat: the retracted LightGBM training time (634 s, versus 55 s now) is
implausible as a like-for-like measurement and most likely reflects CPU
contention during the original run. Treat that single row as unreliable in both
directions.

## What the manuscript must change

1. **The 57-feature framing is retired everywhere.** Every "57 causal features"
   mention becomes the leakage-free set (503 features for the canonical fold),
   and the leanness claim goes with it.
2. **MLP GroupDRO 0.358 → 0.3115.** The best deep model is now CNN ERM.
3. **LightGBM 0.4577 outranks the ensemble 0.4388.** The deployment
   recommendation must be re-argued or re-chosen.
4. **RQ7 figures change by roughly an order of magnitude in training time** and
   8.5× in memory.
5. **The invariance-versus-selection conclusion strengthens** and can be stated
   more confidently than in the submitted version.
