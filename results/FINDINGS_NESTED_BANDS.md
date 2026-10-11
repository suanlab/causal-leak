# Nested Band Selection — Blocker 1 Resolution

Re-run of the causal-band experiment with frequency-band selection nested
strictly inside the training regions. Supersedes the band-selection evidence
in `FINDINGS_PHASE2C.md` and `robust_bands_results.json`, both of which
measured attribution on the evaluation fold.

- Protocol: `src/band_selection.py`, `src/run_nested_causal_bands.py`
- Raw results: `results/nested_causal_bands_results.json`
- Log: `results/run_nested_causal_bands.log` (6,144 s, 7 evaluation regions)
- Config identical to the original: XGBoost 200 trees, depth 6, lr 0.1, seed 42

## What changed

Previously, band importance was computed on region 43 (`run_freq_analysis.py:90-129`),
the 65-422 Hz subset was derived from it, and `run_causal_bands.py` then reported
that subset's score **on region 43**. Selection and evaluation shared a fold.

Now: for each evaluation region R, bands are ranked by permutation importance
averaged over leave-one-region-out folds *within the training pool*, R is never
an input to selection, and each fold emits a manifest that `verify_manifest()`
checks. Every fold passed.

## Result 1 — the 121-422 Hz band does not survive honest selection

Selection frequency across the 7 evaluation regions, under the pre-specified rule
(mean inner-fold macro-F1 drop > 0) and under a stricter sensitivity rule
(mean > 0 **and** positive in a majority of inner folds):

| Band group | Primary rule | Strict rule | Mean inner-fold Δf1 |
|---|---:|---:|---:|
| 10-121 Hz | 7/7 | 6/7 | +0.0078 |
| **121-422 Hz** | **2/7** | **2/7** | **-0.0041** |
| 422-1470 Hz | 5/7 | 5/7 | +0.0027 |
| 1470-5120 Hz | 7/7 | 7/7 | +0.0202 |

Both directions of the published claim invert:

- **121-422 Hz was reported as the causal leak band.** It is selected in 2 of 7
  folds and its mean permutation importance is *negative* — permuting it tends to
  *improve* out-of-region macro-F1. On region 27 the improvement is large
  (Δf1 -0.098 to -0.166 depending on outer fold), i.e. for that region the band
  carries actively harmful, region-specific signal.
- **422-1470 Hz was reported as an active confounder whose removal improves OOD
  macro-F1.** It is selected in 5 of 7 folds with positive mean importance.

The band with the most consistent, largest generalisable contribution is
**1470-5120 Hz** (7/7, +0.0202) — the range the original analysis treated as
anti-causal.

## Result 2 — selection still helps; the published band does not

Macro-F1 on the held-out region, mean over the 7 evaluation regions:

| Feature set | macro-F1 | acc | weighted F1 |
|---|---:|---:|---:|
| `signal_all` (534 features, no selection) | 0.3006 ± 0.081 | 0.5490 | 0.3000 |
| `nested_causal` (leakage-free selection) | **0.3123 ± 0.087** | **0.5552** | **0.3046** |
| `fixed_65_422` (published band) | 0.2990 ± 0.098 | 0.5350 | 0.3027 |

Feature selection remains worthwhile (+0.0117 macro-F1 over no selection), so the
broader claim that restricting the spectrum helps OOD transfer survives. The
*specific published subset* does not: averaged across regions it is marginally
worse than using the whole spectrum, and it has the highest variance of the three
(± 0.098), ranging from 0.3593 on region 27 to 0.1335 on region 48.

## Result 3 — the canonical fold, re-measured

On region 43 — the fold the paper reports — with selection blind to it:

| Feature set | n feat | acc | macro-F1 |
|---|---:|---:|---:|
| `signal_all` | 534 | 0.2978 | 0.4148 |
| `fixed_65_422` | 57 | 0.3013 | 0.4273 |
| `nested_causal` | 503 | **0.3210** | **0.4363** |

The first two rows reproduce the published numbers exactly. The third is the
honest protocol, and it *beats* the published configuration — while having
selected `['10-121', '422-1470', '1470-5120']`, that is, everything **except**
the band the paper calls causal, and **including** the band the paper calls a
confounder.

## Per-region detail

| Region | n test | classes | Selected groups | signal_all | nested | fixed |
|---:|---:|---:|---|---:|---:|---:|
| 27 | 4,908 | 4 | 10-121, 121-422, 1470-5120 | 0.2753 | 0.3121 | 0.3593 |
| 43 | 5,074 | 5 | 10-121, 422-1470, 1470-5120 | 0.4148 | 0.4363 | 0.4273 |
| 44 | 1,682 | 5 | 10-121, 422-1470, 1470-5120 | 0.3327 | 0.3661 | 0.2814 |
| 45 | 15,052 | 5 | 10-121, 422-1470, 1470-5120 | 0.4056 | 0.4002 | 0.4073 |
| 46 | 19,322 | 4 | 10-121, 1470-5120 | 0.2613 | 0.2638 | 0.2619 |
| 47 | 13,989 | 4 | all four | 0.2286 | 0.2286 | 0.2221 |
| 48 | 3,138 | 4 | 10-121, 422-1470, 1470-5120 | 0.1857 | 0.1790 | 0.1335 |

## Limitations

- Effect sizes are small (most |Δf1| < 0.02) against substantial inner-fold
  variance. The 121-422 Hz rejection is driven largely by region 27, where the
  band is strongly harmful; excluding region 27 would leave the band near zero
  rather than clearly negative. The honest statement is that 121-422 Hz is *not
  robustly generalisable*, not that it is reliably harmful.
- Band groups are the four coarse ranges from `run_robust_bands.py`. A finer grid
  might isolate a narrower generalisable sub-band inside 121-422 Hz; this run
  cannot rule that out.
- Selection optimises mean inner-fold macro-F1 only. No hyperparameter is tuned
  per fold, so the comparison isolates the band choice.

## Downstream impact

`irm_core.py:78` hard-codes `bands_in_range(65, 422)` for `mode='causal'`, so every
MLP / CNN / IRM / GroupDRO / PGA / ensemble result computed on the "57 causal
features" inherits the retracted band. Those runs need repeating against a
leakage-free feature set before any of the RQ4-RQ6 numbers can be reported.

Unaffected: the identifier-shortcut audit (RQ1), the row-CV → GroupKFold → LORO
ladder (RQ2), and the ten-benchmark external replication — none of these involve
band selection.

---

# Part 2 — The Other Fold Structures

The analysis above holds the evaluation axis fixed at the administrative region.
Four further experiments use different fold structures, and each needed its own
leakage-free selection rather than borrowing the region-43 bands, which were
chosen with knowledge of the other regions. `band_selection.py` was generalised
to take the held-out axis (`env`) and the inner-fold axis (`inner_env`)
separately; selections are memoised in `results/nested_band_cache.json`.

Comparison values for the retired protocol come from `paper-eaai/claims.csv`,
not from the original JSONs: these four scripts were re-run before their outputs
were archived, so the per-fold raw files were overwritten. Claim-level values
survive; per-fold detail for the old protocol does not.

## Cross-region (11 LORO folds, per-fold nested bands)

| Region | $n$ test | Selected groups | signal\_all | nested | $\Delta$ |
|---:|---:|---|---:|---:|---:|
| 27 | 4,908 | 10-121, **121-422**, 1470-5120 | 0.2753 | 0.3121 | +0.0368 |
| 29 | 4,484 | 10-121, 422-1470, 1470-5120 | 0.3202 | 0.3458 | +0.0256 |
| 41 | 470 | 10-121, 1470-5120 | 0.5872 | 0.7977 | +0.2104 |
| 42 | 520 | 10-121, 1470-5120 | 0.1697 | 0.1774 | +0.0076 |
| 43 | 5,074 | 10-121, 422-1470, 1470-5120 | 0.4148 | 0.4363 | +0.0215 |
| 44 | 1,682 | 10-121, 422-1470, 1470-5120 | 0.3327 | 0.3661 | +0.0334 |
| 45 | 15,052 | 10-121, 422-1470, 1470-5120 | 0.4056 | 0.4002 | −0.0054 |
| 46 | 19,322 | 10-121, 1470-5120 | 0.2613 | 0.2638 | +0.0024 |
| 47 | 13,989 | all four | 0.2286 | 0.2286 | ±0.0000 |
| 48 | 3,138 | 10-121, 422-1470, 1470-5120 | 0.1857 | 0.1790 | −0.0067 |
| 50 | 1,269 | 10-121, 1470-5120 | 0.2336 | 0.2464 | +0.0128 |

Eight wins, one tie, two losses. Weighted $\Delta$ macro-F1 $= +0.0075$ against
$+0.004$ for the retired fixed band (`claims.csv` C32) -- the leakage-free
protocol is roughly twice as good on the aggregate the paper actually quotes.
The unweighted mean ($+0.0308 \pm 0.0586$) is much larger than the weighted one
because region 41 contributes $+0.2104$ on 470 rows with two classes; the
weighted figure is the honest one.

121-422 Hz is selected in 2 of 11 regions, consistent with the 2 of 7 above.

## County-level OOD (16 sigungu folds)

Bands for a county fold are selected with that county's entire region excluded --
stricter than necessary, but unambiguously safe and cache-reusable.

| Statistic | Retired fixed band | Nested |
|---|---:|---:|
| Unweighted mean $\Delta$ macro-F1 | $-0.013 \pm 0.067$ | $+0.0042 \pm 0.0193$ |
| Weighted mean $\Delta$ macro-F1 | $+0.002$ | $-0.0024$ |
| Wins (restricted / full spectrum) | 9 / 7 | 7 / 4 (5 ties) |

Both protocols land within $\pm 0.005$ of zero: at county granularity,
restricting the spectrum neither helps nor hurts on average. The difference that
does matter is dispersion --- the across-fold standard deviation falls from
$0.067$ to $0.019$, a 3.5-fold tightening. The retired band's apparent
per-fold swings were largely an artefact of a band chosen against one region.

Five folds return exactly $\Delta = 0$: for those counties the nested selection
retained all four band groups, so the two feature sets coincide.

## Temporal split

Selection restricted to pre-cutoff recordings, with regions as inner folds.

| Band group | Temporal axis | Spatial axis (regions) |
|---|---:|---:|
| 10-121 Hz | 0/6 ($-0.0071$) | 7/7 ($+0.0078$) |
| **121-422 Hz** | **4/6 ($+0.0113$)** | **2/7 ($-0.0041$)** |
| 422-1470 Hz | 4/6 ($+0.0138$) | 5/7 ($+0.0027$) |
| 1470-5120 Hz | 2/6 ($+0.0137$) | 7/7 ($+0.0202$) |

This is close to the mirror image of the spatial result: the temporal axis keeps
121-422 Hz and discards 10-121 Hz. A band can be stable in time and unstable in
space, so "this band generalises" is not a well-formed claim until the axis is
named. Restricting the spectrum makes no difference to temporal performance
either way ($\Delta$ macro-F1 $= +0.0003$ across six model families).

## Calibrated threshold sweep

Five of eleven LORO folds carry normal samples in both the calibration and test
partitions. Across those folds, detection remains $1.00$ at every calibrated FAR
target ($0.5\,\%$, $1\,\%$, $2\,\%$, $5\,\%$), with achieved test FAR tracking
somewhat above target ($0.0105$, $0.0182$, $0.0302$, ...). The band change does
not move this result: binary leak-vs-normal detection is easy enough on this
benchmark that the feature set is not the binding constraint.

## Reading across the four

Restricting the spectrum helps at region granularity ($+0.0075$ weighted),
does nothing at county granularity ($-0.0024$ weighted), does nothing across
time ($+0.0003$), and is irrelevant to binary detection. The honest summary is
that leakage-free band selection is a small and granularity-dependent effect,
not the deployment lever the retired analysis presented it as -- and that its
main benefit may be variance reduction rather than mean improvement.
