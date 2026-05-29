# Integrity Audit — Plagiarism / Hallucination / Logic / References / AI-Style

Date: 2026-04-20. Systematic pass over `paper/paper.tex` (1378 lines, 18 pp).

## 🚨 Issues Found and Fixed

### H1 (Hallucination / Overstated Claim) — REx was never evaluated
**Location**: Contribution 4 (formerly lines 191--193) and Related Work §2.3.

**Before**:
> "Gradient-based invariance regularisers --- IRM, GroupDRO, and
> **REx-style penalties** --- on MLP and 1D-CNN backbones trained on
> two NVIDIA TITAN Xp GPUs fail to match causal feature selection..."

**Reality**: Methodology §4.4, Results §6.6, and the IRM hyperparameter
sweep tested ERM, IRM (IRMv1), and GroupDRO. **No REx experiment was run.**
The phrase "REx-style penalties" falsely implied a third method was
evaluated.

**Fix applied**: Removed REx from the contribution. Related Work softened
to "Variants such as REx... Section 6 contributes a data point for the
two most widely used members of the family (IRM and GroupDRO)." Krueger
et al. remains cited as context (which is legitimate); no longer claimed
as a tested method.

### L1 (Logical Error) — Ranking skipped hierarchical classifier
**Location**: Discussion §8.1 (formerly lines 1094--1096).

**Before**:
> "The best OOD macro-F1 achieved in this paper is 0.4326 (tree-only
> ensemble); **the next-best is 0.4273 (causal XGBoost)**; below that
> are invariance-regularised MLP and CNN variants."

**Reality**: Hierarchical classifier (§6.8) reaches 0.4287, which lies
**between** the ensemble (0.4326) and the flat causal XGBoost (0.4273).
The sentence skipped it, creating a false three-way ranking.

**Fix applied**:
> "The best ... is 0.4326 (tree-only ensemble), followed by 0.4287
> (hierarchical classifier) and 0.4273 (flat causal XGBoost); below
> these, the best invariance-regularised MLP reaches 0.3615 and the
> augmented CNN 0.3555."

### L2 (Logical Error) — Stage 1 feature-set contradiction
**Location**: Contribution 5 (formerly line 202--203) vs methodology §4.5.

**Before (contribution)**:
> "A binary energy-based normal gate **(trained on peak aggregates)**
> followed by a 4-class abnormal typer (trained on causal bands)..."

**Reality**: `run_hierarchical.py` line 91 and methodology §4.5 show
both Stage 1 and Stage 2 are trained on the **same 57 causal features**.
The "peak aggregates only" description in the contribution is wrong.

**Fix applied**:
> "A binary normal gate followed by a 4-class abnormal typer (**both
> on the 57 causal features**) attains macro-F1 0.4287 with perfect
> OOD normal-class recall. Soft-averaging this hierarchical classifier
> with the flat causal XGBoost yields the final deployment recipe at
> macro-F1 **0.4326**."

(Also closes a minor omission: the original contribution 5 did not
mention the 0.4326 ensemble at all, yet §6.9 reports it as the best
result. The fix surfaces the final headline in the contribution list.)

### C1 (Unsupported strong claim) — "first"
**Location**: Conclusions §9 (formerly line 1179).

**Before**: "This paper provides **the first** field-deployment-oriented
evaluation of machine-learning water-leak detection..."

**Reality**: "First" over an entire application area is difficult to
prove; a less exposed claim is enough.

**Fix applied**: "...provides **a** field-deployment-oriented evaluation
..., **to our knowledge the first to apply sensor-, site-, and
region-group holdouts to this dataset**." This is the precise, defensible
claim (verified: Lee \& Kim 2023, Shukla 2020, Fereidooni 2021 all use
row-level CV, and no other publication on AI Hub dataset 138 uses
region-group holdout).

## ✅ Items That Passed the Audit

### Plagiarism
- Compared abstract, introduction, related work, and methodology against
  Lee \& Kim 2023 (local PDF); the only verbatim phrase is the explicitly
  quoted sentence *``ascertaining its capability in pinpointing the root
  causes of water leaks''*, attributed with quotation marks and citation.
- Dataset description (70 384 rows, 49 sites, 12 regions, class
  percentages) is factual and not covered by copyright.
- No suspiciously uniform sentences or inserted boilerplate identified.

### Reference verification (20 bibitems cross-checked)
| Reference | Verification |
|-----------|--------------|
| hunaidi2000plastic | Journal AWWA 92(2), 82--94 — real |
| ismail2019review | IEEE Access 7, 51965--51981 — real |
| shukla2020leakage | Automation in Construction 117, 103256 — real |
| fereidooni2021hybrid | J Ambient Intell Humaniz Comput — real (cited in Lee \& Kim 2023) |
| lee2023leak | Sensors 23(21), 8935 — verified from local PDF |
| chen2016xgboost | KDD 2016 — real |
| breiman2001rf | Machine Learning 45(1) — real |
| arjovsky2019irm | arXiv:1907.02893 — real |
| sagawa2020groupdro | ICLR 2020 — real |
| gulrajani2021dg | ICLR 2021 — real |
| rosenfeld2021irm | ICLR 2021 — real |
| krueger2021rex | ICML 2021 — real |
| geirhos2020shortcut | Nature MI 2(11) — real |
| zech2018pneumonia | PLoS Medicine 15(11) — real |
| degrave2021covid | Nature MI 3(7) — real |
| beery2018terra | ECCV 2018 — real |
| damour2022underspec | JMLR 23(226) — real |
| lundberg2017shap | NeurIPS 2017 — real |
| park2019specaugment | Interspeech 2019 — real |
| zhang2018mixup | ICLR 2018 — real |
| lin2017focal | ICCV 2017 — real |

No fabricated references.

### AI-style writing patterns
Grep'd the full manuscript for common AI-generated filler terms:
`delve`, `tapestry`, `navigate`, `seamless`, `leverage`, `harness`,
`realm`, `intricate`, `nuanced`, `profound`, `pivotal`, `underscore`,
`crucially`, `myriad`, `paramount`, `holistic`, `in today's`, `embark`,
`testament`, `meticulous`, `showcase`, `prowess`, `utilize`, `delve`.
**Zero matches.** The one `undercore`-family substring (`underscore`)
produced zero hits.

Em-dash usage: 27 LaTeX `---` (proper typography), 0 raw Unicode `—`.

`notably` / `furthermore` / `moreover` / `importantly`:
- `more importantly` appears 2× (both in concrete operational contexts:
  "more importantly for field deployment, decoupling..." and "perhaps
  most importantly for methodological practice --- the causal
  signature..."). Retained; they convey specific content.
- No "It is worth noting", "In summary", "Moreover", "Furthermore".

### Numerical integrity (re-verified)
18 numerical claims cross-checked against the raw JSON files on
2026-04-19 and remain exact. See `PRESUBMISSION_AUDIT.md` for the full
table.

### Citation-key vs bibitem matching
21 `\cite` keys ↔ 21 `\bibitem` entries ↔ 22 `references.bib` entries.
No orphans, no missing.

### LaTeX build (final, after fixes)
- pdflatex errors: 0
- Missing number: 0
- Undefined citations / references: 0
- Pages: 18
- File size: 2.23 MB

## Summary Table

| Axis | Status |
|------|:------:|
| Plagiarism | ✅ clean |
| Hallucination (fabricated fact) | ✅ 1 fixed (REx) |
| Logical error | ✅ 2 fixed (ranking, Stage 1 features) |
| Unsupported strong claim | ✅ 1 hedged ("first") |
| Wrong/fake references | ✅ none |
| AI-style filler | ✅ none detected |
| Numerical fidelity | ✅ all 18 claims exact |
| Bibliography / citations | ✅ 1:1 match |

**Manuscript is clean for submission after the four corrections.**
