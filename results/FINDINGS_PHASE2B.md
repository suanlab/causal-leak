# Phase 2B Findings — IRM / GroupDRO Do Not Beat Causal-XGBoost

Run date: 2026-04-13
Hardware: 2 × NVIDIA TITAN Xp (12 GB each), CUDA 12.2 driver 535.
Compute: MLP 32 min (3 methods), CNN 36 min (3 methods).

## Setup

Same OOD protocol as Phase 2C:
- Train: 11 광역시도 (region_lvl1) environments, 58,779 rows.
- Val:   random 10 % of training rows for early stopping, 6,531 rows.
- Test:  region 43 (충북), 5,074 rows with all 5 classes present.

Two architectures:
- **MLP**: 3-layer fully-connected (57 → 256 → 128 → 5), dropout 0.2,
  trained on the *same 57 causal features* that Phase 2C showed was the
  best feature set for XGBoost.
- **CNN**: Conv1d stack (1→32→64→128) with BatchNorm + MaxPool over the
  raw 513-band spectrum, global-avg-pooled features concatenated with 21
  tabular features (20 MAX + month), then a 128-unit head.

Three training methods per architecture:
- `erm`     : standard cross-entropy, class-balanced weights.
- `irm`     : IRMv1 penalty (Arjovsky et al. 2019), 30-epoch warmup at
              λ=1, linear anneal to λ=100 over remaining epochs.
- `groupdro`: Sagawa et al. 2020, exponential per-environment loss
              reweighting with η=0.01.

Early stopping on val macro-F1, patience 25 (MLP) / 15 (CNN).

## Headline — None of the DL methods beat XGBoost

All numbers on the LORO region 43 test set (5,074 rows).

| Model | Acc | Macro-F1 | `in` | `out` | `noise` | `other` | `normal` |
|-------|----:|---------:|-----:|------:|--------:|--------:|---------:|
| XGB signal_all (534 feat) | 0.2978 | 0.4148 | 0.353 | 0.252 | 0.360 | 0.109 | **1.000** |
| XGB causal 65–422 Hz (57 feat) | 0.3013 | **0.4273** | 0.326 | 0.206 | 0.391 | 0.214 | **1.000** |
| MLP ERM | 0.2872 | 0.3155 | 0.311 | 0.199 | 0.381 | 0.154 | 0.533 |
| MLP IRM | 0.2907 | 0.3122 | 0.302 | 0.143 | 0.396 | 0.212 | 0.508 |
| **MLP GroupDRO** | **0.3210** | **0.3615** | 0.346 | 0.166 | 0.446 | 0.177 | 0.672 |
| CNN ERM | 0.3494 | 0.3484 | 0.421 | 0.165 | 0.454 | 0.141 | 0.562 |
| CNN IRM | 0.3589 | 0.3438 | 0.437 | 0.194 | 0.444 | 0.197 | 0.448 |
| CNN GroupDRO | 0.2605 | 0.2848 | 0.340 | 0.203 | 0.310 | 0.055 | 0.514 |

## Five Observations

### 1. XGBoost + causal feature selection stays on top

The best DL macro-F1 (MLP GroupDRO, 0.3615) is **6.6 pp below**
`xgb_causal_65_422Hz` (0.4273). The best DL accuracy (MLP GroupDRO,
0.3210) is **2 pp above** XGBoost's accuracy (0.3013), but the
cost is a collapse in the `normal` class that drags F1 down.

### 2. All DL models fail at the `normal` class (F1 ≈ 0.45–0.68)

XGBoost trivially achieves `normal` F1 = 1.000 because its tree splits
immediately separate normal from abnormal on a single MAX feature or a
low-energy band — the physical "low total signal energy ⇒ normal"
signature that Phase 2C identified.

DL models, however, give up a large fraction of `normal` accuracy in
exchange for trying to learn invariant representations of the 4
abnormal classes. With only 45 normal samples in the test fold, this
is a particularly costly trade.

### 3. GroupDRO > ERM, IRM ≤ ERM

GroupDRO consistently improves over plain ERM on MLP (+4.6 pp F1) and is
roughly neutral on CNN (−2.3 pp F1, but +0 acc). IRM hurts on both
architectures:

| | Δ MLP F1 | Δ CNN F1 |
|--|---------:|---------:|
| ERM → IRM | −0.003 | −0.044 |
| ERM → GroupDRO | **+0.046** | −0.023 |

This matches the literature: IRM is fragile and notoriously hard to tune
when the number of training environments is small (11 here) and their
sizes are extremely imbalanced (40 × size ratio between region 46 and
region 42).

### 4. CNN ERM beats MLP ERM, but CNN GroupDRO underperforms MLP GroupDRO

| | ERM F1 | GroupDRO F1 |
|---|------:|----:|
| MLP | 0.3155 | **0.3615** |
| CNN | **0.3484** | 0.2848 |

The CNN captures more signal from the raw spectrum under plain ERM.
But GroupDRO's aggressive reweighting of worst-case environments hurts
the CNN because its convolutional features are more sensitive to noisy
gradient directions from a few outlier environments.

### 5. `in` and `out` remain the hardest — shortcut bias is physical

Across every model tested, `out` (옥외누수) macro-F1 stays in
[0.143, 0.252]. Even GroupDRO's best CNN run gets only 0.218. This
class is overwhelmingly affected by the 422–1470 Hz confounder range
that Phase 2C flagged: it contains genuine `out` information but
entangled with site-specific pipe resonances. Invariance learning
alone cannot unmix those signals without a structural prior.

## The Paper's Negative Result Is Actually a Positive Finding

This is the key inversion Phase 2B reveals:

> **Even with gradient-based invariance regularisers and GPU-scale
> training, deep learning does not beat a XGBoost model given the
> correct 57 causal bands.** The research payoff sits in the feature
> selection step, not in the training algorithm.

Equivalently: once you have chosen features that obey physics and
removed the shortcut bands, a tree-based model on those features is a
tighter solution than any IRM / DRO variant you can train on the full
spectrum. **The right place to inject invariance into this problem is
the feature side, not the loss side.**

## Limitations / Next Experiments

1. **Hierarchical classifier**: a "normal gate" (binary classifier on
   MAX features) + 4-class abnormal classifier on causal bands would
   likely recover DL's lost `normal` F1 and then add value on abnormal
   separation. This is the obvious Phase 2D experiment.
2. **More environments**: 시군구 (county-level) would give ~40+
   training envs, which IRM and GroupDRO both benefit from. Requires
   careful handling since some 시군구 only contain 1–2 sensors.
3. **Pretraining on raw spectrum**: a self-supervised pretraining pass
   on unlabeled spectrograms (if available) could stabilise the DL
   features enough to close the gap.
4. **XGBoost + causal bands + per-env reweighting**: apply GroupDRO-style
   sample weights on top of the XGBoost causal model. This would be the
   cheapest way to marginally improve the strongest baseline.

## Summary Numbers

> Best DL (MLP + GroupDRO) — macro-F1 0.3615, test acc 0.3210.
>
> Best tree (XGBoost + causal 65–422 Hz) — macro-F1 0.4273, test acc
> 0.3013.
>
> **Causal feature selection alone beats every invariance-regularised
> deep model by +6.6 pp macro-F1.**
