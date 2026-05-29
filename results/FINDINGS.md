# Phase 1 Findings — Shortcut Learning Is Real, and It Costs 39 Points

Run date: 2026-04-13
Model: XGBoost (n_estimators=200, max_depth=6, lr=0.1, tree_method=hist)
Total compute: 22 fits, 4,683 s (~78 min).

## Headline 2 × 3 Matrix

| Feature Set | Split Protocol | # Feat | Accuracy | Macro-F1 |
|-------------|----------------|-------:|---------:|---------:|
| full | paper_split | 544 | **0.9402** | 0.9274 |
| full | random 5-fold CV | 544 | **0.9449 ± 0.002** | 0.9331 |
| full | **group_site (OOD)** | 544 | **0.5529 ± 0.127** | **0.4188** |
| signal | paper_split | 534 | 0.8881 | 0.8610 |
| signal | random 5-fold CV | 534 | 0.8915 ± 0.002 | 0.8669 |
| signal | **group_site (OOD)** | 534 | **0.5537 ± 0.113** | **0.4378** |

Random-choice baseline = 0.20.
Majority-class baseline (always `normal`) = 0.315.

## Five Observations

### 1. The paper's accuracy is a split artifact, not a model property

Under the paper's train/val protocol, XGBoost reaches 94.02 % (we did not
hyper-tune to chase the reported 99.79 % — see §Caveats). Under 5-fold *random*
CV, the number is actually higher (94.49 %). Under 5-fold **group-by-site**
CV, where test sites are unseen at train time, it collapses to **55.29 ± 12.7 %**.
**Δ = −39 percentage points** purely from changing the split.

### 2. Identifier features add *zero* value once the shortcut is closed

| | `full` | `signal` | Δ |
|---|---:|---:|---:|
| paper_split | 0.9402 | 0.8881 | +0.052 |
| random CV | 0.9449 | 0.8915 | +0.053 |
| **group_site** | **0.5529** | **0.5537** | **−0.001** |

Under the shortcut-friendly splits, `site`, `sid`, `lrate`, `llevel`, and
regional codes together contribute a ~5 pp accuracy bump. Under OOD
evaluation, that bump **completely disappears**. The identifier features are
not just redundant — on macro-F1 they become *mildly harmful*
(0.4188 vs 0.4378), because the model tries to use them for patterns that do
not transfer.

### 3. The `normal` class is trivially perfect everywhere

Per-class F1 on `normal` is 0.996–1.000 in every single cell, including OOD.
Reason: `lrate` perfectly separates normal from any-leak in *both* the full and
signal feature sets (wait — `lrate` is not in `signal`!). After rechecking:
`signal/group_site` also gets 0.996 F1 on normal. So the `normal` class is
**independently identifiable from the vibration spectrum alone**
(low overall energy → normal). That is a real physical signal and a
reproducible finding.

### 4. The *real* 5-class problem is a 4-class abnormal problem

Under group_site, the 4 abnormal classes collapse into a confused mass:

```
full/group_site confusion (row = true, column = predicted)
               in     out   noise   other   normal
true     in   5213    7740     600    1379        0
true    out   5884   11977     550    1320        0
true  noise   1730    1874     609    1445        0
true  other   2865    3837     537     658        0
```

The model essentially predicts "out" for anything abnormal. `other` class
recall drops to **8 %**, `noise` recall to **10 %**. A naive "always predict
out" baseline on abnormal samples would score comparably.

### 5. High variance across folds is itself a finding

Group_site fold accuracies: [0.74, 0.53, 0.36, 0.63, 0.50] (full). The
difference between fold 0 and fold 2 is 38 pp. This reflects how uneven the
49 sites are — some held-out folds happen to contain sites whose leak
signatures resemble the training set; others don't. **No single accuracy
number meaningfully summarises "how well leak detection works on new
sites"** — we need per-region reporting for honesty.

## Caveats

- **Paper reproduction gap**: we get 94.02 %, not 99.79 %. We did not
  hyper-tune XGBoost and did not include `sid` in the original notebook's
  exact form (we added it). The point of this experiment is not a faithful
  reproduction but a relative comparison — the 39 pp collapse under site-OOD
  holds regardless of the starting number.
- **Zero-ML lookup gives 99.77 %** (earlier shortcut probe). That is the
  real upper bound of the shortcut, not 99.79 %. An ML model is
  unnecessary when the split leaks labels.
- **5 folds × 49 sites** means some folds hold out only 7 sites. Variance
  is high. A proper Leave-One-Region-Out (12 folds by 광역시도) will be
  cleaner for the paper — planned for Phase 2.

## Immediate Implications for the Research Narrative

1. **Story Figure 1** writes itself: side-by-side bar chart of the 6 cells
   above with the Δ = −39 pp arrow between `random` and `group_site`.
2. **Story Figure 2**: the confusion matrix under `full/group_site` shows
   the model collapsing to "predict `out`". This visually kills the "99.79 %
   is real" narrative.
3. **The honest 5-class accuracy for signal-only OOD is 55 %**. This is the
   number that would actually transfer to a new city. Subsequent phases need
   to close the gap between 55 % (ERM) and whatever Invariant Risk
   Minimisation / GroupDRO / domain-invariant representation learning can
   achieve.

## Next Steps (Phase 2 preview)

- [ ] Leave-One-Region-Out evaluation using `region_lvl1` (12 광역시도) for
      a tighter, lower-variance OOD protocol.
- [ ] IRM / GroupDRO using 광역시도 as environment — target: beat 55 %.
- [ ] Frequency-band causal importance: which HZ ranges drive the signal
      residual after identifiers are removed?
- [ ] Worst-fold failure mode analysis — which specific sites are
      unlearnable and why?
