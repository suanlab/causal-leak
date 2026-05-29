# Figure captions

## Figure 1 — Shortcut collapse matrix (Phase 1)
`fig1_matrix.png`

Bar chart of test accuracy (blue) and macro-F1 (red) on the 5-class leak
typing task under six (feature set × split protocol) configurations:
{full, signal} × {paper_split, random 5-fold CV, group_site 5-fold CV}.
Error bars are fold-to-fold standard deviations. Dashed grey line marks
the random-choice baseline (0.20). The dramatic −39 pp drop from
group_site relative to the other two protocols demonstrates that the
benchmark's reported accuracy is an evaluation-protocol artefact,
independent of whether identifier features are included.

## Figure 2 — Leave-One-Region-Out per-region performance (Phase 2A)
`fig2_loro_per_region.png`

Per-region accuracy (top) and macro-F1 (bottom) for each of the 12 LORO
folds over Korean 광역시도 regions, comparing the `full` (blue) and
`signal` (red) feature sets. The overall weighted accuracy is 0.568 but
the point estimates span 0.19 (경남) to 0.94 (대구). Regions with
near-trivial label distributions (e.g. 세종: only `out`) inflate the
unweighted mean. Region 43 (충북, 5,074 rows, all five classes present)
is the only fold that constitutes a realistic 5-class evaluation, with
accuracy 0.298 and macro-F1 0.415.

## Figure 3 — Frequency-band importance alignment (Phase 2C)
`fig3_band_importance.png`

Left: grouped permutation importance (macro-F1 drop when each log-spaced
band range is permuted on the held-out region 43 test set). Positive
values indicate bands whose permutation hurts OOD performance — i.e.
bands that carry generalisable signal. Negative values indicate the
permutation *improves* performance — harmful confounder bands.
Right: XGBoost's native summed gain per band group. The two distributions
are misaligned: gain is concentrated in 1470-5120 Hz, but permutation
importance localises the real signal to 121-422 Hz.

## Figure 4 — Confusion matrices for three key cells (Phase 1)
`fig4_confusion.png`

Row-normalised confusion matrices (true class on the y-axis, predicted on
the x-axis) for three representative evaluation cells:
(left) `full / random` — paper-style shortcut-friendly evaluation where
every class reaches 0.9+ recall;
(middle) `full / group_site` — OOD collapse: the 4 abnormal classes
merge into a shared prediction blob dominated by `out`;
(right) `signal / group_site` — identifier-free OOD, essentially
identical behaviour to `full / group_site`, showing that the identifier
features add no genuine information when the shortcut is closed.

## Figure 5 — Causal-band subset retraining (Phase 2C)
`fig5_causal_bands.png`

Left: accuracy (blue) and macro-F1 (red) for five XGBoost variants on
LORO region 43 as the input HZ band range is varied while keeping the
MAX aggregates and `month` column constant. Dashed red line marks the
macro-F1 of the all-bands baseline. The 57-feature `causal 65-422 Hz`
model achieves the best macro-F1 (0.4273 vs 0.4148 for the full
513-band model).
Right: per-class F1 heatmap (rows = classes, columns = variants). The
`other` class is most sensitive to the chosen band range — F1 = 0.214
with causal bands, 0.050 with anti-causal bands.

## Figure 6 — Invariance-learning comparison (Phase 2B)
`fig6_irm_comparison.png`

Left: accuracy and macro-F1 for XGBoost baselines (signal_all, causal)
and six DL configurations (MLP {ERM, IRM, GroupDRO}, CNN {ERM, IRM,
GroupDRO}) on LORO region 43. The dashed red line marks the best
observed macro-F1 (XGBoost causal, 0.4273) — no DL configuration
crosses it.
Right: per-class F1 heatmap. Note the `normal` class F1 at 1.000 for
XGBoost vs 0.45-0.68 for DL: the DL collapse in `normal` is the
primary cause of the macro-F1 gap.
