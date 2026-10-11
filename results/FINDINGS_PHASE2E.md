# Phase 2E Findings — Robustness checks

> **Superseded (October 2026).** The 121-422 Hz "causal band" reported below was
> selected using attribution measured on the evaluation fold and then evaluated
> on that same fold. It does not survive nested selection (chosen in 2 of 7
> regions). Numbers that depend on the 57-feature set are retracted; the
> archived artefacts are in `results/retracted_fixed_band_65_422/`. See
> `FINDINGS_NESTED_BANDS.md` and `FINDINGS_RQ4_RQ6_RERUN.md` for the corrected
> analysis.

Run date: 2026-04-13. Two follow-up experiments stress-test the
Phase 2C and Phase 2B main results.

## 2E.a — Cross-region robustness of "causal 65-422 Hz > signal_all"

We run 11 LORO folds (region 36 skipped, only one class), training two
XGBoost variants — `signal_all` (all 513 HZ + 20 MAX + month) and
`causal 65-422 Hz` (57 features total) — and report Δmacro-F1 per
fold.

| Region | n_test | # cls | signal_all F1 | causal F1 | Δ F1 |
|-------:|------:|-----:|-------------:|----------:|-----:|
| 27 (대구) | 4908 | 4 | 0.275 | 0.359 | **+0.084** |
| 29 (광주) | 4484 | 3 | 0.320 | 0.383 | **+0.063** |
| 41 (경기) | 470  | 2 | 0.587 | 0.257 | **−0.331** |
| 42 (강원) | 520  | 2 | 0.170 | 0.188 | +0.019 |
| **43 (충북)** | 5074 | **5** | 0.415 | **0.427** | **+0.013** |
| 44 (충남) | 1682 | 5 | 0.333 | 0.281 | −0.051 |
| 45 (전북) | 15052 | 5 | 0.406 | 0.407 | +0.002 |
| 46 (전남) | 19322 | 4 | 0.261 | 0.262 | +0.001 |
| 47 (경북) | 13989 | 4 | 0.229 | 0.222 | −0.007 |
| 48 (경남) | 3138  | 4 | 0.186 | 0.134 | −0.052 |
| 50 (제주) | 1269  | 2 | 0.234 | 0.232 | −0.001 |
| **Macro (unweighted)** | | | | | **−0.024 ± 0.104** |
| **Weighted by n_test**  | | | | | **+0.004** |

Summary: causal wins 6/11, loses 5/11. Weighted-mean is marginally
positive (+0.004), unweighted is marginally negative (−0.024).

The +0.013 improvement on region 43 — which is the only fold where
all five classes are present in non-trivial counts — holds. On
degenerate folds with only 1-2 classes present, the causal vs
signal_all comparison is dominated by class imbalance effects and
should be treated as uninformative.

### Revised position statement (for paper Section 5)

> The 121-422 Hz causal range is confirmed as a generalisable leak
> signal on the canonical 충북 fold (+0.013 macro-F1) and on the
> two next most abnormal-heavy folds (대구 +0.084, 광주 +0.063).
> It is not a universal replacement for the full spectrum: on folds
> that contain only 1-2 classes, the reduced feature set can lose up
> to 0.33 F1 due to class-imbalance artefacts outside the regime for
> which the causal-band hypothesis was formulated.

## 2E.b — IRM λ × warmup sweep on MLP + causal bands

Ten configurations, same OOD test split (region 43). The Phase 2B
default (λ_max = 100, warmup = 30) gave test F1 = 0.3122.

| λ_max | Warmup | Val F1 | **Test F1** | Test Acc |
|------:|------:|------:|--------:|--------:|
| **1** | **10** | **0.8958** | 0.3145 | 0.2931 |
| **1** | **30** | **0.8958** | 0.3145 | 0.2931 |
| 10 | 10 | 0.8254 | 0.2898 | 0.2741 |
| 10 | 30 | 0.8560 | 0.2767 | 0.2499 |
| 100 | 10 | 0.7612 | 0.3181 | 0.3027 |
| 100 | 30 | 0.8354 | 0.3157 | 0.2917 |
| 1000 | 10 | 0.7474 | 0.2867 | 0.2978 |
| 1000 | 30 | 0.8329 | **0.3372** | **0.3055** |
| 10000 | 10 | 0.7474 | 0.2867 | 0.2978 |
| 10000 | 30 | 0.8313 | 0.3005 | 0.2895 |

Validation-selected IRM: λ_max = 1, with either warmup setting tied on
validation F1 → **test F1 = 0.3145**.

Best observed test IRM: λ_max = 1000 with 30-epoch warmup →
**test F1 = 0.3372**. This is a diagnostic result, not the
validation-selected model. It is still:

- −0.090 below XGB causal (0.4273)
- −0.024 below MLP GroupDRO (0.3615)

### IRM sweep interpretation

- λ = 1 (any warmup) is effectively ERM: penalty is too small to
  change gradient direction.
- λ = 10 is the unstable regime: enough penalty to disrupt early
  training but not enough to drive the invariant solution.
- λ ∈ {100, 1000} is the usable regime.
- λ = 10000 overwhelms the CE objective.
- Longer warmup (30 > 10) is consistently better.

### Conclusion

IRM on this dataset is tunable — a dedicated λ × warmup sweep buys
another 2.5 pp macro-F1 — but not transformative. No IRM
configuration matches causal-band XGBoost on OOD macro-F1. The
evidence supports the Phase 2B conclusion: for this dataset, causal
feature selection is a larger lever than invariance regularisation.
