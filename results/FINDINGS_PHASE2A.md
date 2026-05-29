# Phase 2A Findings — Leave-One-Region-Out Collapses Macro-F1 to 0.30

Run date: 2026-04-13
Protocol: 12-fold LORO over `region_lvl1` (광역시도).
Model: same XGBoost config as Phase 1 (200 trees, depth 6).
Total compute: 24 fits, 5,069 s (~84 min).

## Headline

| Feature | Macro Acc (unweighted) | Weighted Acc (by n_test) | Macro-F1 |
|---------|----------------------:|-------------------------:|---------:|
| full    | 0.5881 ± 0.287 | 0.5621 | **0.2997** |
| signal  | 0.6020 ± 0.261 | 0.5680 | **0.3038** |

Compare to Phase 1:
- Group-site 5-fold (Phase 1): signal F1 = **0.438**
- LORO 12-fold (this phase): signal F1 = **0.304**

The jump from group-site to region-OOD **loses another 13 pp of macro-F1**.

## Per-region acc (signal)

| Region | n_test | n_classes | acc | f1 | Notes |
|-------:|-------:|----------:|-----|-----|-------|
| 27 (대구) | 4908 | 4 | 0.937 | 0.275 | 93% normal — high acc is class imbalance |
| 29 (광주) | 4484 | 3 | 0.941 | 0.320 | 92% normal |
| 36 (세종) | 476 | **1** | 0.857 | 0.231 | degenerate: only `out` |
| 41 (경기) | 470 | 2 | 0.696 | 0.587 | mostly `out` |
| 42 (강원) | 520 | 2 | 0.340 | 0.170 | worst-bad (small, few classes) |
| **43 (충북)** | **5074** | **5** | **0.298** | **0.415** | **only region with all 5 classes** |
| 44 (충남) | 1682 | 5 | 0.762 | 0.333 | mostly normal |
| 45 (전북) | 15052 | 5 | 0.867 | 0.406 | 82% normal |
| 46 (전남) | 19322 | 4 | 0.400 | 0.261 | largest abnormal region |
| 47 (경북) | 13989 | 4 | 0.384 | 0.229 | large abnormal region |
| 48 (경남) | 3138 | 4 | 0.195 | 0.186 | worst: below random 5-way |
| 50 (제주) | 1269 | 2 | 0.548 | 0.234 | `out` + `in` only |

## Four Observations

### 1. The "full" feature set has become mildly harmful under LORO too

Macro-F1: signal 0.304 vs full 0.300. Accuracy: signal 0.602 vs full 0.588.
The identifier features continue to fail to transfer — and when they do
contribute, they push the model toward memorized patterns that don't fit new
regions. Phase 1's "full ≈ signal" finding **holds under the stricter
12-fold region protocol**.

### 2. The only region with all 5 classes gets 29.78 % accuracy (5-class)

Region 43 (충북) is the only LORO fold that is genuinely representative:
it contains every leak type in substantial quantities (1868 in, 1685 noise,
757 other, 719 out, 45 normal). On this fold, `signal` gets **acc = 0.298,
macro-F1 = 0.415**. **Random baseline is 0.20.** That is the honest answer
to "can this dataset train a 5-class leak classifier that generalises": it
gets about 10 pp above chance.

### 3. Weighted accuracy < unweighted accuracy — bigger regions are harder

| Feature | Unweighted | Weighted (by n_test) | Δ |
|---------|-----------:|---------------------:|---:|
| signal  | 0.6020 | 0.5680 | −0.034 |
| full    | 0.5881 | 0.5621 | −0.026 |

The largest folds (46 전남, 47 경북, 43 충북) — which are also the ones
with the most balanced label distributions — are the hardest. The high
unweighted numbers are carried by small, degenerate regions that only
contain one or two classes.

**The honest single-number summary is weighted-acc ≈ 0.568**.

### 4. Class-imbalanced regions create a reporting trap

Regions 27, 29, 36 each score ~85–94% accuracy, but those regions contain
only 1–4 classes and are dominated by whichever leak type happens to match
the model's default prediction. Reporting only unweighted mean accuracy
would *inflate* the headline to 0.60. Authors of the original paper who
report 99.79 % on a held-out val set with 98 % sensor overlap are doing a
more extreme version of the same trap.

## Honest Summary Number

> **On the only region that resembles a realistic 5-class evaluation
> (충북, 5074 rows, all classes present), signal-only XGBoost reaches
> 29.8 % accuracy and 0.415 macro-F1.**
>
> For 5-way classification, this is ~10 pp above random and ~0 pp above a
> "predict-most-common-abnormal" policy.

This is the baseline Phase 2B (IRM / GroupDRO) must beat, and Phase 2C will
dig into whether there is any per-class spectral signature that survives
the OOD collapse.
