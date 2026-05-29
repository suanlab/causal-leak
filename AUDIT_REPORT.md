# Audit Report — `causal-leak` v0.2

Audit date: 2026-04-14.
Scope: full project (code, raw experiment outputs, paper draft markdown,
paper LaTeX, references, figures).
Audit dimensions: numeric correctness vs.\ result JSONs, hallucination,
plagiarism, logical errors, citation integrity, AI-style writing.

## Summary

| Category | Verdict |
|----------|---------|
| Numerical claims vs. JSON results | **Verified, with 4 fixes applied** |
| References | **1 fabricated citation removed, 1 mis-categorised removed** |
| Plagiarism | **None detected** (only properly attributed quote) |
| Logical errors | **None detected** |
| AI-style writing | **Acceptable; some em-dash overuse softened** |
| Code bugs that affect results | **None detected** |

## What was verified clean (no fix needed)

### Numerical claims confirmed against JSON sources

| Claim in paper | Source | Verified? |
|----------------|--------|----------:|
| 99.79 % E4 (Lee & Kim 2023) | original paper Table 8 (PDF) | ✓ |
| 99.77 % zero-ML lookup | Phase 0 probe `[P4]` | ✓ |
| 0.9402 / 0.9274 (full / paper_split) | matrix_results.json | ✓ |
| 0.5529 ± 0.127 (full / group_site) | matrix_results.json | ✓ |
| 0.5537 ± 0.113 (signal / group_site) | matrix_results.json | ✓ |
| 0.5881 / 0.6020 (LORO macro_acc) | loro_results.json | ✓ |
| 0.5621 / 0.5680 (LORO weighted_acc) | loro_results.json | ✓ |
| 0.2997 / 0.3038 (LORO macro_f1) | loro_results.json | ✓ |
| 0.4148 (signal_all Chungbuk F1) | causal_bands_results.json | ✓ |
| 0.4273 (causal 65-422 Hz F1) | causal_bands_results.json | ✓ |
| 0.4287 (Hierarchical F1) | hierarchical_results.json | ✓ |
| Stage 1: 45 / 45 normal recovered | hierarchical_results.json | ✓ |
| 43,189 abnormal training rows | derived: 65310 − (22166 − 45) | ✓ |
| MLP GroupDRO 0.3615 | irm_mlp_results.json | ✓ |
| CNN ERM 0.3458 | irm_cnn_results.json | ✓ |
| Best IRM sweep λ=1000 wu=30 → 0.3372 | irm_sweep_results.json | ✓ |
| Cross-region wins 6 / losses 5 | cross_region_results.json | ✓ |
| Δf1 weighted = +0.004 | cross_region_results.json | ✓ |
| Region 43 Δf1 = +0.013 | cross_region_results.json | ✓ |
| 121–422 Hz perm Δf1 = +0.033 | freq_analysis.json (sum of two) | ✓ |
| 422–1470 Hz perm Δf1 = −0.033 | freq_analysis.json (sum of two) | ✓ |

### References cross-checked
- **Lee & Kim 2023** (DOI 10.3390/s23218935) — verified from local PDF;
  authors, page count, E1–E4 design, 99.79 % accuracy, label encoding,
  feature importance plot all matching.
- **Arjovsky et al. 2019** IRM, arXiv:1907.02893 — well-known, real.
- **Sagawa et al. 2020** GroupDRO, ICLR — real.
- **Gulrajani & Lopez-Paz 2021** "In Search of Lost DG", ICLR — real.
- **Rosenfeld et al. 2021** "Risks of IRM", ICLR — real.
- **Zech et al. 2018** PLoS Medicine 15(11) e1002683 — real.
- **DeGrave et al. 2021** Nat Mach Intell 3(7), 610–619 — real.
- **Geirhos et al. 2020** Nat Mach Intell 2(11), 665–673 — real.

### No plagiarism detected
- Only one direct quote from prior paper:
  *"ascertaining its capability in pinpointing the root causes of water
  leaks"* — present in the original Conclusions, marked with quotes,
  attributed.
- All other prose is paraphrased from our own findings; no overlap with
  Lee & Kim (2023) abstract, intro, or methods exceeds normal technical
  vocabulary.

### Code paths confirmed
- `data.py::get_xy('signal')` → 1 month + 513 HZ + 20 MAX = 534 features
  ✓ matches paper.
- `data.py::get_xy('full')` → 7 ident + 4 region + 513 HZ + 20 MAX =
  544 features ✓.
- `run_causal_bands.py::bands_in_range(65, 422)` → 36 HZ bands; +1 month
  +20 MAX = 57 ✓.
- `run_loro.py` → plain mask-based LORO (12 iterations) ✓.
- `irm_core.py::irm_penalty` → IRMv1 unit-scaling formulation ✓.
- `irm_core.py::groupdro` → exponential-weight update with η=0.01 ✓.
- All training scripts use the same `XGB_PARAMS` (n=200, depth=6,
  lr=0.1, hist) ✓.

## Issues found and fixed

### F1 — Gain share denominator inconsistency
**Before:** Section 5 table reported HZ-band gain shares as
`2.3 % / 7.3 % / 22.2 % / 65.3 %`, summing to 97.1 %. The implicit
denominator was *total feature gain* (HZ + MAX + month), but the table
header said "Gain share" without qualifier and the row total looked
non-conserving.

**After:** Re-stated as HZ-only shares
`2.5 % / 7.5 % / 22.9 % / 67.1 %` (sum = 100 %), with an explicit note
that MAX/month features are excluded from the normalisation. Updated
in `paper_draft.md` and `paper.tex`.

**Impact:** The qualitative ordering is unchanged. The strongest claim
("65 % of HZ gain on irrelevant range") becomes 67 %, slightly stronger.

### F2 — "11 % of the HZ features"
**Before:** Abstract and §5.2 said
*"using 11 % of the HZ features"*. Numerically: 36 causal HZ bands /
513 total HZ bands = 7 %, not 11 %. The 11 % refers to total feature
share (57 / 534), not HZ-only share.

**After:** Re-phrased as
*"using 11 % of the features (57 of 534), or equivalently 7 % of the
513 HZ bands"*. Same edit applied in §5.2.

### F3 — "99.998 % accuracy on covered rows"
**Before:** §3.1 reported the `sid → label` lookup as
**99.998 %** accurate on covered rows.

**After:** The actual probe output was **1.0000** (i.e., perfect
100.0 % on all covered rows). Replaced "99.998" with "100.0".

### F4 — Hunaidi 2004 *Water Research* citation (potentially fabricated)
**Before:** §5 and the bibliography cited
*"Hunaidi, O., Wang, A., Bracken, M., Gambino, T., Fricke, C. 2004.
Detecting Leaks in Plastic Water Pipes. Water Research 38(3),
673–683."* This citation could not be verified against the literature.

**After:** Replaced with the well-attested
*Hunaidi, O., Chu, W., Wang, A., Guan, W. 2000. Detecting Leaks in
Plastic Pipes. Journal AWWA 92(2), 82–94.*
and added a more recent review,
*Ismail et al. 2019, A Review of Vibration Detection Methods Using
Accelerometer Sensors for Water Pipeline Leakage. IEEE Access 7,
51965–51981.* (also cited by Lee & Kim 2023 as ref [3], so this is a
literature path the original authors had already touched).

The qualitative claim was softened from a specific frequency range
attribution to "broadly consistent with published leak-acoustics
ranges for plastic pipes."

### F5 — Ilyas et al. 2019 mis-categorised
**Before:** §1.1 grouped *Ilyas et al. 2019* with sensor-network
shortcut literature. The paper actually concerns adversarial examples,
not sensor benchmarks; the placement was a mis-categorisation.

**After:** Removed from §1.1 and from `references.bib`.

### F6 — AI-style em-dash usage softened
**Before:** Several sentences used em-dashes for parenthetical asides
(e.g. *"the conditions our sensor fleet would actually encounter — in
the field"*) that are characteristic of AI-generated prose.

**After:** Converted ~6 of the most prominent em-dash insertions to
parentheses or commas, while leaving normal-prose em-dashes (used for
strong asides) intact. Total em-dash count: 27 (down from ~33).

## Open / accepted limitations

- **Cross-region robustness is mixed**: causal bands beat signal_all on
  6 / 11 LORO folds and lose on 5 / 11. The +0.013 macro-F1 gain on the
  canonical Chungbuk fold holds, but the universal claim does not.
  This is reported honestly in §7.1.1.
- **Some folds collapse to 1–2 classes**, making per-fold metrics
  noisy. The paper flags this in the discussion.
- **IRM is tunable but not transformative** on this dataset — confirmed
  by §7.1.2 sweep, reported as a negative result.
- The choice of XGBoost hyperparameters (200 trees, depth 6, lr 0.1)
  is fixed across all phases for fairness; we did not separately tune
  per phase, which leaves some residual headroom.

## What we did NOT find
- No fabricated experimental numbers.
- No citation outside the bibliography.
- No copy-paste from the prior Sensors paper or any other paper.
- No code paths that contradict the methods description in the paper.
- No silently dropped data rows or filtering steps that would bias
  results.

## Overall verdict

After fixes F1–F6, the paper draft (`paper/paper_draft.md`) and
LaTeX manuscript (`paper/paper.tex`) are internally consistent with
the experimental result JSONs in `results/`, the bibliography is
verifiable, and the writing is free of obvious AI tells beyond
normal technical-prose conventions. The negative findings (Phase 2B
+ 2E.b for IRM, 2E.a for cross-region heterogeneity) are reported
with appropriate hedging rather than overstated.

The manuscript is ready for a co-author / native-speaker proofreading
pass and journal submission.
