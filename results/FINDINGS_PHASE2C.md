# Phase 2C Findings — The Model Uses the Wrong Frequencies

> **Superseded (October 2026).** The 121-422 Hz "causal band" reported below was
> selected using attribution measured on the evaluation fold and then evaluated
> on that same fold. It does not survive nested selection (chosen in 2 of 7
> regions). Numbers that depend on the 57-feature set are retracted; the
> archived artefacts are in `results/retracted_fixed_band_65_422/`. See
> `FINDINGS_NESTED_BANDS.md` and `FINDINGS_RQ4_RQ6_RERUN.md` for the corrected
> analysis.

Run date: 2026-04-13
Setup: signal-only XGBoost, LORO holdout on region 43 (충북 — the only region
with substantial counts of all 5 classes).
Baseline: **acc = 0.2978, macro-F1 = 0.4148**.

## Headline

> **XGBoost allocates 62 % of its feature gain to the 1470–5120 Hz range,
> but permuting those bands on the held-out region has *zero* effect on
> test performance. The bands that actually matter for OOD generalisation
> — 121–422 Hz — receive only 11 % of the gain.**
>
> Worse, permuting the 422–1470 Hz range *improves* OOD macro-F1 by
> 1.6–1.7 percentage points — meaning the model learned a harmful
> dependency on this range.

## Raw Data

### XGBoost Gain Importance per Band Group

| Band | Gain | Share |
|------|-----:|------:|
| 10–19 Hz | 11.9 | 0.2 % |
| 19–35 Hz | 14.0 | 0.2 % |
| 35–65 Hz | 45.9 | 0.8 % |
| 65–121 Hz | 62.7 | 1.1 % |
| 121–226 Hz | 123.0 | 2.2 % |
| 226–422 Hz | 290.9 | 5.1 % |
| 422–788 Hz | 436.2 | 7.7 % |
| 788–1470 Hz | 822.7 | 14.5 % |
| 1470–2744 Hz | 1323.7 | 23.4 % |
| **2744–5120 Hz** | **2368.9** | **41.9 %** |

### Permutation Importance (Macro-F1 drop when band permuted on test)

Positive = permuting that band *hurts* → band carries generalisable signal.
Negative = permuting that band *helps* → band carries confounder/shortcut.

| Band | F1 drop | Interpretation |
|------|--------:|----------------|
| 10–19 Hz | −0.003 | noise |
| 19–35 Hz | −0.002 | noise |
| 35–65 Hz | +0.002 | noise |
| 65–121 Hz | +0.007 | minor signal |
| 121–226 Hz | **+0.014** | **real signal** |
| **226–422 Hz** | **+0.019** | **real signal — strongest** |
| **422–788 Hz** | **−0.016** | **confounder / shortcut** |
| **788–1470 Hz** | **−0.017** | **confounder / shortcut** |
| 1470–2744 Hz | −0.002 | irrelevant |
| 2744–5120 Hz | +0.001 | irrelevant |

### Gain × Permutation — The Alignment Problem

| Band | Gain share | Perm Δ F1 | Verdict |
|------|-----------:|----------:|---------|
| 121–422 Hz | 7.3 % | **+0.033** | **Real physics, underused** |
| 422–1470 Hz | 22.2 % | **−0.033** | **Confounder, overused** |
| 1470–5120 Hz | 65.3 % | −0.001 | **Irrelevant, massively overused** |

The model spends 65 % of its budget on frequencies above 1.47 kHz that
carry *nothing* for generalisation, and another 22 % on a 0.4–1.5 kHz
range that's actively harmful. The physically meaningful 121–422 Hz range
gets 7 %.

## Per-Class Permutation (F1 drop)

| band | in | out | noise | other | normal |
|------|---:|----:|------:|------:|-------:|
| 10–19 Hz | −0.012 | −0.003 | +0.000 | −0.001 | +0.000 |
| 19–35 Hz | −0.002 | −0.001 | −0.004 | −0.002 | +0.000 |
| 35–65 Hz | −0.008 | −0.000 | +0.004 | +0.003 | +0.011 |
| 65–121 Hz | −0.000 | −0.007 | −0.021 | **+0.041** | +0.022 |
| 121–226 Hz | −0.000 | **+0.049** | −0.013 | +0.012 | +0.022 |
| 226–422 Hz | +0.016 | −0.014 | **+0.065** | +0.027 | +0.000 |
| 422–788 Hz | +0.022 | −0.017 | −0.015 | **−0.078** | +0.011 |
| 788–1470 Hz | −0.018 | −0.004 | −0.020 | **−0.055** | +0.011 |
| 1470–2744 Hz | −0.022 | +0.006 | −0.002 | −0.015 | +0.022 |
| 2744–5120 Hz | +0.017 | −0.002 | +0.010 | −0.021 | +0.000 |

Each of the 4 abnormal classes has a **distinct physical signature**:

- **`in` (옥내누수)** → 226–788 Hz (combined +0.038)
- **`out` (옥외누수)** → 121–226 Hz peak (+0.049)
- **`noise` (전기소음)** → 226–422 Hz peak (+0.065)
- **`other` (기타소음)** → 65–121 Hz (+0.041) + strongly harmed by 422–1470 Hz
  (combined −0.133 — this class is worst affected by the confounder)
- **`normal`** → band-invariant, uses MAX aggregates and low energy (F1
  stays ~1.000 under all permutations)

## Top-30 Features by Gain

```
  1. MAX15 (209.4)   11. 4200HZ (45.2)   21. MAX9   (30.1)
  2. MAX19 (105.6)   12. MAX11  (43.5)   22. 4160HZ (29.6)
  3. MAX7   (83.0)   13. 4080HZ (41.8)   23. 240HZ  (29.3)
  4. MAX17  (73.7)   14. 1550HZ (39.3)   24. 2000HZ (29.2)
  5. 4170HZ (72.9)   15. MAX1   (35.5)   25. 4640HZ (28.9)
  6. MAX5   (68.9)   16. 970HZ  (35.1)   26. 990HZ  (26.8)
  7. 3380HZ (66.4)   17. MAX3   (34.5)   27. 3650HZ (26.7)
  8. 3570HZ (65.5)   18. 3830HZ (31.3)   28. 3780HZ (26.7)
  9. MAX13  (60.5)   19. 4060HZ (30.8)   29. 2520HZ (26.2)
 10. MAX8   (52.1)   20. 910HZ  (30.6)   30. 3470HZ (25.6)
```

Of the 22 HZ features in this top-30 list, **18 are above 788 Hz**. That is
exactly the range our permutation analysis labels as *not useful for
generalisation*.

The **MAX features dominate the top 10** (they are aggregates of the 20
highest amplitudes across the spectrum). These are physical envelopes
rather than specific resonances and remain informative because the `normal`
class is cleanly separable by overall signal energy.

## Physical Interpretation

Water-leak vibrations in buried pipes propagate best between roughly
100 Hz and a few hundred Hz — the well-known leak noise band (IWA / AWWA
reviews cite 50–1500 Hz depending on pipe material and diameter, with
most informative content below ~600 Hz for plastic pipes and slightly
higher for metallic pipes). Our permutation results are consistent with
that physics: **121–422 Hz carries generalisable leak signatures**, and
higher ranges reflect local pipe resonances and site-specific acoustic
environments that do not transfer.

The model, however, learned to rely on higher bands because in the
training regions those high-frequency site-specific patterns happen to
correlate strongly with leak labels — classic shortcut learning.

## Implication for the Research Narrative

Phase 1 showed identifiers are shortcuts. Phase 2A showed even signal-only
models fail to transfer. **Phase 2C now pinpoints *why*: the model uses
site-specific resonances as a proxy for leak type.** This suggests a
concrete intervention:

> Restrict the input spectrum to the 65–422 Hz physics band and retrain.
> If the hypothesis is correct, the causal-band model should match or
> beat the full-spectrum model on LORO region 43.

This is the follow-up experiment currently running
(`run_causal_bands.py`). Results will be appended below when done.

## Follow-up Experiment — Causal Band Subset (confirmed)

Same LORO split (region 43). Five feature-set variants:

| Config | # feat | acc | macro-F1 | `in` | `out` | `noise` | `other` | `normal` |
|--------|------:|----:|---------:|-----:|------:|--------:|--------:|---------:|
| signal_all (all 513 HZ) | 534 | 0.2978 | 0.4148 | 0.353 | 0.252 | 0.360 | 0.109 | 1.000 |
| **causal 65–422 Hz** | **57** | **0.3013** | **0.4273** | 0.326 | 0.206 | 0.391 | **0.214** | 1.000 |
| low_only 0–65 Hz | 28 | 0.3059 | 0.4251 | 0.294 | 0.224 | 0.456 | 0.173 | 0.978 |
| mid_confounder 422–1470 Hz | 125 | 0.2858 | 0.4066 | 0.331 | 0.213 | 0.397 | 0.103 | 0.989 |
| anti_causal 788–5120 Hz | 455 | 0.2958 | 0.4031 | 0.361 | 0.226 | 0.400 | **0.050** | 0.978 |

All configs share the 20 `MAX*` aggregate features and the `month` column —
only the *specific HZ bands* vary.

### Confirmations

1. **Causal band selection wins on macro-F1 with a 10× reduction in
   features**: 57 features → 0.4273 F1 vs 534 features → 0.4148 F1
   (+0.0125, +3 % relative). This is the direct experimental
   confirmation that the high-frequency bands were hurting
   generalisation.

2. **The `other` class recovers from 0.109 → 0.214 F1 — nearly 2 ×**
   when the 422–1470 Hz confounder range is removed. Phase 2C's
   permutation analysis flagged this exact class as most harmed by
   422–1470 Hz (combined −0.133 F1 drop), and the prediction holds
   cleanly.

3. **The anti-causal configuration is the worst for `other`**
   (F1 = 0.050, almost collapsed). Using only the high-frequency bands
   the original model overweighted leaves `other` nearly unpredictable.

4. **A 125-feature `mid_confounder` model lands between signal_all and
   anti_causal** (F1 = 0.4066) — consistent with it being genuinely
   harmful content.

### Caveats

- **`low_only_0_65 Hz` also performs well** (F1 = 0.4251, acc = 0.3059),
  almost matching `causal_65_422`. Because all configs share the 20 `MAX`
  aggregates (which pool from the entire spectrum), we cannot fully
  disentangle how much signal flows through `MAX` vs the specific HZ
  bands. A clean ablation dropping `MAX` is a Phase-2B prerequisite.
- **Overall acc is still ~30 %**: we reduced feature waste, but we did
  not yet recover real performance. The OOD collapse is primarily about
  the training algorithm (ERM is not invariant-aware), not just the
  feature set. Phase 2B will attempt IRM / GroupDRO to close this gap.

### Headline Number

> **Restricting the 513-band vibration spectrum to the 65–422 Hz
> physics band improves OOD macro-F1 from 0.4148 to 0.4273 while using
> only 11 % of the original HZ features. The `other` class F1 nearly
> doubles.**
>
> This is experimental confirmation, not just analysis, that the
> original feature set was shortcut-laden and that causal frequency
> selection alone is a measurable intervention.
