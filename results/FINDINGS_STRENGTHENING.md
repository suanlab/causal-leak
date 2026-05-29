# Strengthening Findings (Phases A–D)

Four reinforcement experiments were run to address the weaknesses
identified in the original manuscript. Results are summarised below.

## Strengthening 1 — Robust Causal Band Identification
(`run_robust_bands.py` · CPU · 1333 s total)

### 1a. Bootstrap 95% CIs on multiple regions

Four LORO folds ($\ge$4 classes, $\ge$5k rows each) with $B=100$
bootstrap permutations per fold.

| Region | 10--121 | 121--422 | 422--1470 | 1470--5120 |
|--------|--------:|---------:|----------:|-----------:|
| 43 (Chungbuk, 5 cls) | +0.007 (CI 0.002, 0.013) | **+0.041 (0.033, 0.049)** | **−0.021 (−0.029, −0.013)** | +0.008 (0.004, 0.014) |
| 45 (Jeonbuk, 5 cls) | +0.011 (0.005, 0.018) | −0.005 (−0.016, +0.004) | +0.003 (−0.004, +0.011) | +0.018 (0.013, 0.024) |
| 46 (Jeonnam, 4 cls) | +0.008 (0.005, 0.011) | +0.026 (0.020, 0.031) | +0.015 (0.012, 0.018) | +0.003 (0.000, 0.007) |
| 47 (Gyeongbuk, 4 cls) | +0.012 (0.009, 0.015) | +0.013 (0.010, 0.019) | −0.008 (−0.014, −0.003) | +0.011 (0.007, 0.014) |

**Consensus**: 121--422 Hz mean Δf1 = **+0.0186 ± 0.0169**, wins 3/4.
422--1470 Hz mean Δf1 = −0.003, mixed (wins 2/2). The strongest
single-region signal remains the canonical Chungbuk fold where
121--422 Hz improves by 4 pp and 422--1470 Hz hurts by 2 pp, both
with CIs excluding 0.

### 1b. SHAP second-source attribution

TreeSHAP on 500 Chungbuk test rows. Mean $|\text{SHAP}|$ aggregated
per band group and normalised over HZ features:

| Band | SHAP share | Rank |
|------|----------:|:----:|
| 10--121 Hz     | 8.5\%  | 4 |
| **121--422 Hz**  | **40.4\%** | **1** |
| 422--1470 Hz   | 28.4\% | 2 |
| 1470--5120 Hz  | 22.7\% | 3 |

SHAP independently agrees with permutation: 121--422 Hz is the
top contributor (40.4\% of SHAP magnitude) even though it holds
only 7.5\% of the XGBoost gain. The gain-vs-attribution mismatch
is therefore not a permutation artefact --- it is confirmed by an
independent tree-path attribution method.

## Strengthening 2 — Augmentation + Robust Losses
(`run_augmentation.py` · GPU · 627 s total)

Six (recipe $\times$ architecture) configurations trained from
scratch.

| Recipe | Architecture | test F1 | test Acc |
|--------|-------------|--------:|---------:|
| baseline (ERM)       | CNN | 0.3341 | 0.3343 |
| + **SpecAugment**    | **CNN** | **0.3544** | 0.3384 |
| + Mixup              | CNN | 0.3169 | 0.3431 |
| + Focal loss ($\gamma$=2) | CNN | 0.3393 | **0.3648** |
| all three combined   | CNN | 0.3226 | 0.3007 |
| mixup+focal          | MLP (causal) | 0.2933 | 0.3147 |

**Findings**: SpecAugment provides the biggest per-recipe
macro-F1 gain (+2.0 pp over baseline CNN). Focal loss gives the
highest accuracy (0.3648) but a smaller macro-F1 gain; Mixup hurts
macro-F1 on this dataset.

## Strengthening 3 — Permutation-Guided Attention (Novel Method)
(`run_pga.py` · GPU~0 · 1116 s total)

A novel architecture where MLP band attention weights are
regularised via $\mathrm{KL}(\text{softmax}(a) \,\|\, p)$ where
$p$ is the normalised, ReLU-rectified per-band permutation
importance computed on a training-internal CV split. Sweep over
$\lambda_c \in \{0, 0.1, 1, 10, 100\}$ and prior initialisation.

| Config | test F1 | test Acc |
|--------|--------:|---------:|
| $\lambda_c=0$, uniform init | 0.2932 | 0.2663 |
| $\lambda_c=0$, prior init   | 0.2781 | 0.2783 |
| $\lambda_c=0.1$, U | 0.2656 | 0.2550 |
| $\lambda_c=0.1$, P | 0.2856 | 0.2720 |
| **$\lambda_c=1$, U (best)** | **0.3041** | 0.2897 |
| $\lambda_c=1$, P | 0.2851 | 0.2850 |
| $\lambda_c=10$, U | 0.2629 | 0.2434 |
| $\lambda_c=10$, P | 0.2926 | 0.2797 |
| $\lambda_c=100$, U | 0.2840 | 0.2828 |
| $\lambda_c=100$, P | 0.2942 | 0.2730 |

### Negative result (reported honestly)

No PGA configuration beats the plain MLP ERM baseline (0.3155 from
the original Phase 2B). This is likely because the permutation prior
must itself be computed on training data where sensor overlap is
still extreme, so it inherits the shortcut bias it was meant to
regularise away. The result reinforces the paper's main conclusion
that in the presence of a nameable causal feature (here, the
121--422 Hz range), hard feature selection is a more reliable
intervention than any soft-prior method we tested.

## Strengthening 4 — Ensemble
(`run_ensemble.py` · CPU · 366 s total)

Probabilistic ensemble of the strongest individual models. The CNN
member is selected by validation macro-F1 within the `cnn_*` recipes;
the deployment candidate is fixed in advance as the average of the two
causal tree models.

| Method | test F1 | test Acc | Members |
|--------|--------:|---------:|---------|
| XGB causal flat       | 0.4273 | 0.3013 | --- |
| XGB hierarchical      | 0.4287 | 0.3041 | --- |
| CNN val-selected (SpecAugment) | 0.3544 | 0.3384 | --- |
| soft-average (3)      | 0.3633 | 0.3159 | tree, hier, CNN |
| **pre-specified tree-only soft-avg**  | **0.4326** | **0.3096** | causal + hier |

The pre-specified tree-only average achieves OOD macro-F1 0.4326
vs 0.4287 for the hierarchical tree alone (+0.39 pp) by
soft-averaging two complementary tree models trained on the same
features. Adding the validation-selected CNN degrades macro-F1
because the CNN's one-hot fallback probabilities dilute the tree models'
confidence.

## Aggregate Impact

The headline OOD macro-F1 on the Chungbuk LORO fold moves from:

| Stage | Best macro-F1 |
|-------|--------------:|
| Original Phase 2C (causal XGBoost)         | 0.4273 |
| Phase 2D hierarchical                      | 0.4287 |
| **Strengthening 4 (pre-specified tree average)** | **0.4326** |

Alongside:
- **3-source independent evidence** for 121--422 Hz as the causal
  generalisable leak band (bootstrap CIs + multi-region consensus
  + SHAP independent attribution).
- **Honest negative result on PGA**, reinforcing that causal
  feature selection beats soft-prior injection on this benchmark.
- **Augmentation recipe** (SpecAugment) that pulls the CNN baseline
  from 0.3341 to 0.3544 (+2.0 pp).
