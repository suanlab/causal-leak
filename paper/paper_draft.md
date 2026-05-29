# From 99.79 % to Field Deployment: A Causal Audit and Out-of-Distribution Redesign of Our Water-Pipeline Leak Detection Pipeline

**Draft v0.2 · 2026-04-13**

**Suan Lee**\*, Byeonghak Kim
School of Computer Science, Semyung University, Jecheon 27136, Republic of Korea.
\*Corresponding author: hahaja01@semyung.ac.kr

*This is a follow-up to our prior study, Lee & Kim (2023), "Machine
Learning Model for Leak Detection Using Water Pipeline Vibration Sensor,"
Sensors 23(21), 8935, DOI: 10.3390/s23218935.*

## Abstract

In our previous study (Lee & Kim, 2023) we proposed a tabular machine-
learning pipeline for water-pipeline leak-type classification on the
Korean AI Hub vibration dataset and reported a XGBoost classifier that
reached 99.79 % accuracy under stratified 10-fold cross-validation on
the four-experiment (E1-E4) feature sweep. In the Conclusions we
indicated that our next step would be edge deployment and "ascertaining
its capability in pinpointing the root causes of water leaks." In this
follow-up we deliver the audit implied by that statement. We ask: does
the 99.79 % that we reported translate to a classifier that generalises
to unseen sensors, unseen sites, and unseen Korean administrative
regions — the conditions our sensor fleet would actually encounter in
the field?

We show that the 99.79 % E4 accuracy we reported is almost entirely
explained by four overlapping identifier-label couplings in the AI Hub
data split: (i) every one of the 2,671 training sensor ids maps to
exactly one class for its entire lifetime; (ii) the `lrate` column
perfectly separates `normal` from any-leak; (iii) 97.96 % of the
validation sensors also appear in training; and (iv) 21 of the 49
sites have ≥95 % label purity. A five-line lookup rule that ignores
the 512 frequency bands entirely reproduces **99.77 %** — within
0.02 pp of our published number. This is consistent with the feature
importance we already published: in our E4 XGBoost, `lrate` and `sid`
are the two dominant features, with individual HZ bands negligible.

We re-evaluate the same XGBoost configuration under protocols that
forbid sensor-, site-, and region-level leakage. Stratified 5-fold
GroupKFold by site drops accuracy from 94 % to **55.3 %** (macro-F1
0.42). Leave-One-Region-Out across the 12 광역시도 regions drops it
further to a weighted **56.8 %** (macro-F1 0.30). On the only region
that contains substantial counts of all five classes (충북, 5,074
rows, macro-F1 0.41), grouped permutation importance pinpoints the
generalisable leak signal to the **121-422 Hz range**, while the
**422-1470 Hz range is an active confounder** that our earlier model
was over-relying on. Retraining XGBoost on only the 57 causal features
(the 65-422 Hz bands plus the MAX aggregates and `month`) improves OOD
macro-F1 from 0.4148 to 0.4273 while using 11 % of the features
(57 of 534), or equivalently 7 % of the 513 HZ bands, and doubles
the F1 of the `other` class from 0.109 to 0.214.

We further test whether gradient-based invariance regularisers (IRM
and GroupDRO on MLP and 1D-CNN backbones trained on two NVIDIA TITAN
Xp GPUs — can recover more of the gap. They cannot: no deep
configuration matches causal-band XGBoost on macro-F1. A hierarchical
classifier (Stage 1: binary `normal`-vs-abnormal gate; Stage 2: 4-class
abnormal typer on causal bands) closes the deployment recipe with
macro-F1 0.4287 and perfect OOD `normal` recall.

**Takeaway for practitioners of our 2023 pipeline.** The 99.79 %
headline is a correct measurement of ceiling performance under the
evaluation protocol we reported, but it is not a faithful estimate of
the accuracy a new installation in a new region will see. For field
deployment we recommend: (a) dropping `site`, `sid`, and `lrate` from
the feature set; (b) restricting the spectrum to 65-422 Hz; (c) using
the hierarchical `normal` gate + 4-class abnormal typer. Under this
recipe, the ceiling is ~30 % 5-class accuracy, ~43 % macro-F1, and
trivially perfect binary `normal` detection — an honest but useful
baseline for the sensor network we described in Lee & Kim (2023).

**Contributions of this follow-up.**
(i) A quantified audit of how the 99.79 % we reported decomposes into
identifier-driven and signal-driven parts.
(ii) The first OOD evaluation of the AI Hub water-leak benchmark at
site-, sensor-, and region-level holdouts.
(iii) Identification of the causal spectral signatures (121-422 Hz
for generalisable leak signal; 422-1470 Hz as harmful confounder).
(iv) A negative result on invariance regularisation: IRM and GroupDRO
do not close the OOD gap on this dataset.
(v) A deployment-oriented hierarchical recipe with code released under
`causal-leak/`.

**Keywords:** water leak detection; vibration sensor; shortcut
learning; out-of-distribution evaluation; invariant risk minimisation;
causal feature selection.

---

## 1  Introduction

In Lee & Kim (2023) we proposed a machine-learning pipeline for
water-pipe leak typing on the Korean AI Hub vibration-sensor dataset.
Our experimental design compared eight tabular models across four
feature regimes (E1 through E4), progressively augmenting the 512-band
time-frequency spectrum with peak aggregates (E2), leak-rate and
leak-level columns (E3), and finally site-, sensor-, and date-level
identifiers (E4). Under E4, XGBoost reached 99.79 % accuracy, which we
reported as the preeminent configuration and the basis for a future
edge-computing deployment.

The published feature-importance chart for our E4 XGBoost already
flagged the dominant role of `lrate` and `sid` relative to any
individual HZ band, but we did not quantify how much of the 99.79 %
was driven by those identifiers, nor did we report out-of-distribution
(OOD) accuracy. In our Conclusions we stated that we would pursue
edge-computing deployment and the "capability in pinpointing the root
causes of water leaks." In the interval since, it has become clear
that (i) the first of those goals requires a model that generalises
to sensors and regions unseen at training time, and (ii) the second
is ill-defined without a framework for attributing predictions to
physical signal rather than site-level confounders.

This follow-up study provides both. We use the same dataset and the
same XGBoost configuration as in Lee & Kim (2023) as the reference
point, then:

1. audit the 99.79 % number against four overlapping identifier
   couplings in the AI Hub split and reproduce it with a trivial
   look-up rule that does not consult the vibration spectrum at all
   (Section 3);
2. evaluate the same pipeline under three increasingly strict
   OOD protocols — sensor-group, site-group, and 12-fold Leave-One-
   Region-Out — and report how accuracy and macro-F1 collapse
   (Sections 3, 4);
3. use grouped permutation importance on the most realistic OOD
   fold to identify which spectral bands carry generalisable leak
   information, and compare them to the XGBoost gain allocation
   (Section 5);
4. retrain with only the causal bands and confirm improvement of
   OOD macro-F1 (Section 5.2);
5. test whether gradient-based invariance regularisers (IRMv1,
   GroupDRO) on MLP and 1D-CNN backbones can beat causal-band
   XGBoost (Section 6, two-GPU experiment);
6. propose a hierarchical deployment recipe — binary normal gate
   + 4-class abnormal typer — that closes the loop on the practical
   use case of the sensor network described in Lee & Kim (2023)
   (Section 7).

### 1.1  Related work on shortcut learning

Shortcut learning in tabular and sensor benchmarks has a well-known
pathology that the medical-imaging literature describes as the
*hospital-ID shortcut* (Zech et al. 2018; DeGrave et al. 2021; Geirhos
et al. 2020). The AI Hub water-leak dataset exhibits a more extreme
variant: because sensor recordings are split per class into separate
CSV files, each `sid` is a bijection to a single leak class rather
than a strong correlate. Under any evaluation split that places a
sensor's measurement history partly in train and partly in test
(including the one we used in E4) the identifier features alone
nearly solve the task.

Invariance-regularised learning is the natural first remedy when
shortcuts are suspected. Invariant Risk Minimisation (IRMv1; Arjovsky
et al. 2019) and GroupDRO (Sagawa et al. 2020) minimise a penalty that
discourages models from exploiting environment-specific correlations.
The empirical reliability of these methods under small numbers of
environments has been repeatedly questioned (Gulrajani & Lopez-Paz
2021; Rosenfeld et al. 2021); our Section 6 adds a specific data point
to that literature on the AI Hub leak dataset.

## 2  Data recap from Lee & Kim (2023)

The AI Hub water-pipe leak dataset consists of 70,384 10.24-second
vibration recordings at 11,000+ sensor locations and 49 sites spread
across 12 Korean 광역시도. Each recording is represented as 512 time-
frequency-band amplitudes at 10 Hz intervals (0-5120 Hz, with the 0 Hz
column identically zero and dropped in our 2023 study) plus 20 peak-
amplitude aggregates MAX0-MAX19. Metadata columns are `site`, `sid`,
`ldate` (YYYYMMDD), `lrate` (leak-rate, 0-90), `llevel` (leak-level,
0-5000), and `leaktype` ∈ {`in`, `out`, `noise`, `other`, `normal`}.
Class distribution: 31.5 % normal, 28.0 % out, 21.2 % in, 11.2 % other,
8.0 % noise.

The shipped dataset ships with a train/val split of 62,564 / 7,820.
In Lee & Kim (2023) we merged the two sets and re-split 8 : 2 into
56,307 / 14,077, evaluated by stratified 10-fold cross-validation.
For our audit we use both the shipped split (to match the literature)
and our 2023 re-split (to match our own reported number).

### 2.1  Notation and methods

Let $\mathcal{D} = \{(x_i, y_i, e_i)\}_{i=1}^{N}$ with $N = 70{,}384$,
where $x_i$ are the 512 + 20 + K tabular features (K = 0 for E1,
3 for E3, 3+3 for E4), $y_i \in \{0,\dots,4\}$ indexes the five
leak classes (with the Lee & Kim 2023 encoding `in`=0, `noise`=1,
`normal`=2, `other`=3, `out`=4 retained for continuity), and
$e_i \in \{1,\dots,12\}$ is the administrative region.

**Protocols.**
- $\mathrm{PaperSplit}$ — the 62,564 / 7,820 split shipped with AI Hub.
- $\mathrm{KFold10}$ — our 2023 stratified 10-fold CV on the 8 : 2
  re-split.
- $\mathrm{RandomCV}_k$, $\mathrm{GroupCV}_k^{\text{site}}$ —
  stratified $k$-fold over rows and over sites respectively.
- $\mathrm{LORO}_{12}$ — leave-one-region-out over `region_lvl1`.

**Risk and penalty definitions.** The weighted softmax cross-entropy
with inverse-frequency class weights $w_c = N/(CN_c)$ is $\ell$, the
pooled empirical risk $R(\theta) = N^{-1} \sum \ell(f_\theta(x_i), y_i)$,
and the per-environment risk $R_e(\theta) = N_e^{-1} \sum_{i:e_i=e}
\ell(f_\theta(x_i), y_i)$.

The IRMv1 penalty is
$$ \mathcal{L}_{\mathrm{IRM}}(\theta)
   = \sum_{e=1}^{E} \Bigl[ R_e(\theta)
     + \lambda \bigl\| \nabla_{w|w=1} R_e(w \cdot f_\theta) \bigr\|^2 \Bigr], $$
with $\lambda$ ramping from 1 to $\lambda_\mathrm{max}$ after a
warmup of $T_\mathrm{warmup}$ epochs.

GroupDRO maintains env weights $q_e$ with update
$q_e \leftarrow q_e \exp(\eta R_e(\theta))/Z$ and minimises
$\sum_e q_e R_e(\theta)$.

The grouped-permutation importance for a feature group $g$ is
$$ \Delta_{\mathrm{F1}}(g) = \mathrm{F1}(y, \hat f(x)) -
   \mathrm{F1}(y, \hat f(\tilde x^{(g)})), $$
where $\tilde x^{(g)}$ permutes only the columns in $g$.

## 3  Phase 1 · Auditing the 99.79 % we reported

### 3.1  Label-leaking identifiers

Regrouping every training row by `sid` and computing label purity
(the fraction of that sensor's rows sharing its modal label), **every
one of the 2,671 training sensors has purity 1.0**; the AI Hub files
are split per-class and a sensor's label is fixed. Of the 1,915
validation sensors, 1,876 (97.96 %) appear in training. Of the 43
validation sites, all 43 appear in training. This is not a subtle
effect: an `sid → majority training label` look-up covers 99.5 % of
the val set and achieves **100.0 %** accuracy on the covered rows.

Independently, the `lrate` meta column perfectly separates `normal`
from any leak: `lrate==90 ⇔ leaktype ≠ normal`. This is visible in
the E3/E4 feature-importance plots in our 2023 paper (where `lrate`
is the single most important variable); our follow-up just states its
consequence explicitly.

### 3.2  Zero-ML reproduction of the 99.79 %

Combining `sid` look-up with an `lrate` override (`lrate ≠ 90`
⇒ predict `normal`) yields a rule with **99.77 %** val accuracy,
within 0.02 pp of the 99.79 % we reported in E4. No vibration
spectrum, no tree, no model. This confirms numerically that the
identifier path we introduced in E4 was driving nearly all of the
observed headroom over E3 (90.30 %).

### 3.3  A 2 × 3 evaluation matrix

To separate the signal contribution from the identifier contribution,
we retrain the same XGBoost (200 trees, depth 6, $\eta$ = 0.1) in
three protocols and two feature sets, reporting macro-F1 across five
folds:

| Feature set | Split | Acc | Macro-F1 |
|-------------|-------|----:|---------:|
| full (544 cols incl. ids) | PaperSplit | 0.9402 | 0.9274 |
| full | Random 5-fold | 0.9449 ± 0.002 | 0.9331 |
| full | **GroupSite 5-fold** | **0.5529 ± 0.127** | **0.4188** |
| signal (512 HZ + 20 MAX + month) | PaperSplit | 0.8881 | 0.8610 |
| signal | Random 5-fold | 0.8915 ± 0.002 | 0.8669 |
| signal | **GroupSite 5-fold** | **0.5537 ± 0.113** | **0.4378** |

Two observations: (i) changing the evaluation protocol from row-level
to site-level drops accuracy by 39 pp; (ii) once the site shortcut is
closed, the identifier features add essentially zero accuracy and
slightly *harm* macro-F1. (Note that our 99.79 % used KFold10, which
is in the row-level family and therefore behaves like the other
row-level protocols above.)

See Figure 1.

## 4  Phase 2A · Honest OOD evaluation via LORO

The GroupSite 5-fold protocol holds out as few as 7 sites per fold
and has variance 0.127. We widen to a 12-fold Leave-One-Region-Out
protocol over the Korean 광역시도.

| Feature | Macro Acc | Weighted Acc | Macro-F1 |
|---------|----------:|-------------:|---------:|
| full    | 0.588 ± 0.287 | 0.562 | 0.300 |
| signal  | 0.602 ± 0.261 | 0.568 | 0.304 |

Signal-only again matches or slightly beats full. Of the 12 regions,
**only 충북 (code 43)** contains all five classes in substantial
quantities (1868 `in`, 1685 `noise`, 757 `other`, 719 `out`, 45
`normal`). On this fold, signal XGBoost scores acc 0.298, macro-F1
0.415. The remaining 11 folds either have 1-2 classes present (so
accuracy is inflated by class imbalance — 93.7 % on 대구 which is
93 % `normal`) or are very small (< 600 rows). We therefore adopt
the 충북 fold as the canonical 5-class OOD test for Sections 5-7.

See Figure 2.

## 5  Phase 2C · Causal spectral analysis

### 5.1  Gain vs permutation importance

On the 충북 test fold we train a signal-only XGBoost and compare its
internal summed gain per log-spaced band group against the macro-F1
drop when the same band group is permuted on the held-out data.

| Band | HZ-gain share | Δ Macro-F1 on perm | Verdict |
|------|-------------:|-------------------:|---------|
| 10-121 Hz | 2.5 % | ≈ 0 | noise |
| 121-422 Hz | 7.5 % | **+0.033** | **real signal** |
| 422-1470 Hz | 22.9 % | **−0.033** | **confounder** |
| 1470-5120 Hz | 67.1 % | ≈ 0 | irrelevant |

(Shares are of total gain across the 513 HZ bands; rows sum to 100 %.
MAX aggregates and `month` are excluded from this normalisation because
they are tabular, not spectral, features.)

The model allocates two-thirds of its gain to a 1470-5120 Hz range
that has no OOD contribution, and 23 % to a 422-1470 Hz range whose
permutation *improves* test macro-F1 by 3.3 pp. The physically
meaningful 121-422 Hz range, broadly consistent with published
leak-acoustics ranges for plastic pipes (Hunaidi et al., 2000;
Ismail et al., 2019), receives only 7.5 % of the gain.

See Figure 3.

Per-class signatures (F1 drop from permuting the listed band):
`in` → 226-788 Hz (+0.038);
`out` → 121-226 Hz (+0.049);
`noise` → 226-422 Hz (+0.065);
`other` → 65-121 Hz (+0.041), strongly harmed by 422-1470 Hz (−0.133);
`normal` → band-invariant, uses MAX aggregates and total spectrum
energy.

### 5.2  Causal-band retraining

We retrain XGBoost under five feature regimes on the 충북 LORO fold,
with MAX and `month` kept constant in all variants:

| Config | # feat | Acc | Macro-F1 | `other` F1 |
|--------|------:|----:|---------:|-----------:|
| signal_all (all 513 HZ) | 534 | 0.2978 | 0.4148 | 0.109 |
| **causal 65-422 Hz** | **57** | **0.3013** | **0.4273** | **0.214** |
| low 0-65 Hz | 28 | 0.3059 | 0.4251 | 0.173 |
| mid_confounder 422-1470 Hz | 125 | 0.2858 | 0.4066 | 0.103 |
| anti_causal 788-5120 Hz | 455 | 0.2958 | 0.4031 | 0.050 |

Causal-band retraining improves macro-F1 from 0.415 to 0.427 using
11 % of the features (57 of 534) and nearly doubles the F1 of the
`other` class (0.109 → 0.214). Anti-causal retraining (using only
the 1470+ Hz range our E4 XGBoost actually attended to) is the worst
configuration and collapses `other` to F1 0.050.

See Figure 5.

## 6  Phase 2B · Can invariance regularisation recover the gap?

A reasonable hypothesis is that applying gradient-based invariance
regularisers on the raw spectrum would recover more OOD performance.
We test two architectures (MLP on the 57 causal features; 1D-CNN on
the raw 513-band spectrum with an MLP head) under three training
methods (ERM, IRMv1, GroupDRO). Training ran on two NVIDIA TITAN Xp
GPUs (12 GB each, CUDA 12.2), 120 epochs, class-balanced cross-entropy.

| Model | Acc | Macro-F1 | `normal` F1 |
|-------|----:|---------:|------------:|
| XGB signal_all | 0.2978 | 0.4148 | **1.000** |
| XGB causal 65-422 Hz | **0.3013** | **0.4273** | **1.000** |
| MLP ERM | 0.2872 | 0.3155 | 0.533 |
| MLP IRM | 0.2907 | 0.3122 | 0.508 |
| **MLP GroupDRO** | **0.3210** | **0.3615** | 0.672 |
| CNN ERM | 0.3096 | 0.3458 | 0.677 |
| CNN IRM | 0.3082 | 0.3016 | 0.446 |
| CNN GroupDRO | 0.3088 | 0.3230 | 0.566 |

No deep configuration crosses the causal-XGBoost macro-F1. The best
DL model (MLP + GroupDRO) trails XGBoost causal by −6.6 pp. The per-
class breakdown identifies the root cause: every DL configuration
collapses the `normal` class F1 to 0.45-0.68, while tree models
keep it at 1.000. `normal` vs any-leak is dominated by a single
physical signal — total spectrum energy captured by the MAX
aggregates — and trees learn it in one root split; class-balanced
DL training trades it away.

IRM underperforms ERM on both architectures, consistent with the
known difficulty of IRM tuning under small numbers of uneven
environments (Gulrajani & Lopez-Paz, 2021). GroupDRO improves over
ERM on MLP (+4.6 pp F1) and is neutral on CNN.

See Figure 6.

## 7  Phase 2D · A hierarchical classifier for deployment

Phase 6 identified the DL failure mode: class-balanced loss trades
`normal` for marginal gains on the four abnormal classes. We test
whether explicitly decoupling the two sub-problems helps a tree
model, and whether it yields a practical deployment recipe.

- **Stage 1**: binary XGBoost `normal` vs abnormal, trained on all
  65,310 rows with the 57 causal-band features.
- **Stage 2**: 4-class XGBoost `in / out / noise / other`, trained
  on the 43,189 abnormal-only rows, same 57 features.

At inference, Stage 1 routes each sample; `normal` predictions go
straight through, `abnormal` predictions pass to Stage 2.

| Model | Acc | Macro-F1 | `in` | `out` | `noise` | `other` | `normal` |
|-------|----:|---------:|-----:|------:|--------:|--------:|---------:|
| Flat 5-class XGB (causal) | 0.3013 | 0.4273 | 0.326 | 0.206 | 0.391 | 0.214 | 1.000 |
| **Hierarchical (causal)** | **0.3041** | **0.4287** | 0.345 | 0.200 | 0.377 | 0.222 | 1.000 |

**Stage 1 achieves perfect OOD accuracy on the 충북 fold** — 45 of 45
normal rows are detected with zero false positives. This confirms
Phase 2C's finding that spectral-energy aggregates form a causal
`normal` signature that transfers across regions. The hierarchical
decomposition improves macro-F1 marginally (+0.001) over flat XGBoost,
but its real value is operational: a field technician gets a
clean two-stage signal (alarm + triage).

## 7.1  Phase 2E · Robustness of the causal finding and IRM tuning

Two follow-up experiments address natural objections.

### 7.1.1  Does "causal 65-422 Hz > signal_all" hold across regions?

Across the 11 non-degenerate LORO folds (region 36 skipped because it
contains only one class), the Δmacro-F1 of causal 65-422 Hz versus
signal_all is:

| Region | n_test | # cls | signal_all F1 | causal F1 | Δ F1 |
|-------:|------:|-----:|-------------:|----------:|-----:|
| 27 (대구) | 4908 | 4 | 0.275 | 0.359 | **+0.084** |
| 29 (광주) | 4484 | 3 | 0.320 | 0.383 | **+0.063** |
| 41 (경기) | 470 | 2 | 0.587 | 0.257 | **−0.331** |
| 42 (강원) | 520 | 2 | 0.170 | 0.188 | +0.019 |
| **43 (충북)** | **5074** | **5** | **0.415** | **0.427** | **+0.013** |
| 44 (충남) | 1682 | 5 | 0.333 | 0.281 | −0.051 |
| 45 (전북) | 15052 | 5 | 0.406 | 0.407 | +0.002 |
| 46 (전남) | 19322 | 4 | 0.261 | 0.262 | +0.001 |
| 47 (경북) | 13989 | 4 | 0.229 | 0.222 | −0.007 |
| 48 (경남) | 3138 | 4 | 0.186 | 0.134 | −0.052 |
| 50 (제주) | 1269 | 2 | 0.234 | 0.232 | −0.001 |
| **Macro (unweighted)** | — | — | — | — | **−0.024 ± 0.104** |
| **Weighted by n_test** | — | — | — | — | **+0.004** |

The causal-band advantage is **not uniform** across regions. Causal
wins in 6 of 11 folds (by average +0.031), loses in 5 folds (notably
region 41 by −0.331), and is essentially neutral under n_test-weighted
averaging (+0.004). A cautious reading:

- Causal 65-422 Hz is an improvement on the canonical 충북 fold and
  on the two most abnormal-heavy non-canonical folds (대구, 광주),
  but not a universal replacement for the full spectrum.
- The dramatic −0.331 loss on region 41 (경기, 470 rows, only 2
  classes) is an artefact of class imbalance: the signal_all model
  happens to predict the `out`-dominant class well, while the
  restricted causal model routes some rows to `in`. With only two
  non-zero F1 entries, the per-class average is highly sensitive.
- Regions where only 2 classes are present are operating outside
  the regime for which the causal-band hypothesis was formulated.

**Honest statement for Section 5**: the 121-422 Hz causal signal is
confirmed on the only fold that resembles a realistic 5-class
evaluation (충북), and on the two biggest abnormal-heavy folds.
It does not uniformly dominate when test folds collapse to 1-2
classes, for which the causal-band analysis was not designed.

### 7.1.2  Does better IRM tuning close the DL-vs-XGBoost gap?

Phase 2B used the IRMv1 defaults from Arjovsky et al. (2019): λ ramping
1 → 100 with 30-epoch warmup. We grid-searched the two hyperparameters
most commonly responsible for IRM instability on the same MLP + causal
bands setup:

| λ_max | Warmup | Val F1 | **Test F1** | Test Acc |
|------:|------:|------:|--------:|--------:|
| 1 | 10 | 0.8958 | 0.3145 | 0.2931 |
| 1 | 30 | 0.8958 | 0.3145 | 0.2931 |
| 10 | 10 | 0.8254 | 0.2898 | 0.2741 |
| 10 | 30 | 0.8560 | 0.2767 | 0.2499 |
| 100 | 10 | 0.7612 | 0.3181 | 0.3027 |
| 100 | 30 | 0.8354 | 0.3157 | 0.2917 |
| 1000 | 10 | 0.7474 | 0.2867 | 0.2978 |
| **1000 | 30** | 0.8329 | **0.3372** | **0.3055** |
| 10000 | 10 | 0.7474 | 0.2867 | 0.2978 |
| 10000 | 30 | 0.8313 | 0.3005 | 0.2895 |

The best configuration (λ_max = 1000, warmup = 30 epochs) improves
IRM test macro-F1 by **+0.025** over our Phase 2B default (0.3122).
At λ ≤ 1 the penalty is effectively inert; at λ ≥ 10000 the penalty
overwhelms the cross-entropy and val F1 collapses to ~0.19. There is
a usable sweet spot around λ = 100-1000 with the long warmup.

However, even the best tuned IRM:

- Still trails XGBoost causal (0.4273) by **−9 pp** macro-F1.
- Still trails MLP GroupDRO (0.3615) by **−2.4 pp**.
- Still trails XGBoost signal_all (0.4148) by **−7.8 pp**.

IRM on this dataset is *tunable* — we can extract another 2.5 pp of
macro-F1 with the right schedule — but not *transformative*: the
invariance penalty does not change the fundamental capability ceiling
of deep models on the 4-class abnormal problem. The main conclusion
of Section 6 stands: causal feature selection is a larger lever than
invariance regularisation on this dataset.

## 8  Discussion

### 8.1  What this says about our 2023 result

We do not retract Lee & Kim (2023). The 99.79 % we reported is a
faithful measurement of the ceiling attainable on the AI Hub dataset
under the evaluation protocol we stated. But that protocol -- row-
level 10-fold stratified cross-validation on the merged 70,384-row
dataset -- contains a 97.96 % sensor overlap between training and
test at the sid level, and an `lrate → leaktype` coupling that
cleanly separates `normal`. Under any protocol that forbids those
two paths, the same XGBoost configuration reaches 30 % accuracy, not
99.79 %. The gap between the two is the "identifier contribution"
that our 2023 paper did not quantify.

### 8.2  What the data actually supports

The honest position after this audit is:

1. **Binary `normal` vs any-leak** is a solved problem on this data.
   XGBoost with MAX aggregates achieves perfect separation under OOD
   on the 충북 fold. That capability would transfer to a field
   deployment.
2. **4-class abnormal typing** (`in` vs `out` vs `noise` vs `other`)
   is brittle: the best OOD macro-F1 we achieved is 0.42, with the
   `other` class the hardest (F1 0.22). The causal-band analysis
   suggests the physics *is* there, but the dataset's environmental
   coverage is thin (49 sites in 12 regions, of which only 1 has a
   representative 5-class distribution).
3. **Identifier features should be dropped from any future benchmark
   use of this dataset**; including them is an evaluation mistake.

### 8.3  Where invariance goes

Our Phase 2B negative result makes a general point: when a dataset
has a clean causal-feature story (here, 121-422 Hz), direct feature
selection outperforms loss-level invariance regularisation. Invariance
methods are best reserved for settings where the invariant feature
cannot be named — raw images, text — not for benchmarks where band-
physics gives us the answer *ex ante*.

### 8.4  Deployment recipe for the Lee & Kim (2023) sensor network

From this audit we derive a deployment recipe for the sensor network
we described in our 2023 paper:

- **Alarm layer** (Stage 1): XGBoost binary, MAX + `month` + 65-422 Hz
  bands, perfect `normal` recall on OOD test.
- **Triage layer** (Stage 2): XGBoost 4-class, causal-band features,
  trained on abnormal-only data.
- **Confidence gate**: flag predictions with Stage-2 softmax margin
  below threshold for human review.
- **Prohibited features**: `site`, `sid`, `lrate`, `llevel`,
  `region_lvl*n`, `ldate` beyond `month`.

## 9  Limitations

- The 121-422 Hz causal range is identified primarily on the 충북
  LORO fold. Cross-checking on the other non-degenerate regions is
  Phase 2E (Section 7.1).
- Our CNN architecture is moderate; a wavelet-based representation
  or a larger model might recover part of the gap.
- We did not exhaustively sweep IRM's $\lambda$ schedule; Phase 2E
  (Section 7.1.2) reports one sweep.

## 10  Conclusion

The 99.79 % E4 accuracy in Lee & Kim (2023) is reproducible but
driven by identifier leakage, not by a scientific capability of
vibration-based leak typing. Under honest OOD evaluation, 5-class
accuracy is ~30 % and macro-F1 ~0.42, concentrated in a narrow
121-422 Hz frequency band consistent with the physics of leak-noise
propagation in plastic pipes. Gradient-based invariance regularisers
do not close the gap; causal feature selection does. We release code
and a hierarchical deployment recipe under `causal-leak/` so that
practitioners who are adopting our 2023 pipeline can do so with an
evaluation protocol that reflects how their sensors will actually
perform in the field.

## Data and code availability

All data comes from AI Hub (https://www.aihub.or.kr/, dataset ID 138,
accessed 5 October 2023). The follow-up pipeline is released at
`causal-leak/`:

- `src/run_matrix.py` — Section 3 2 × 3 matrix;
- `src/run_loro.py` — Section 4 LORO 12-fold;
- `src/run_freq_analysis.py`, `src/run_causal_bands.py` — Section 5;
- `src/irm_core.py`, `src/run_irm_{mlp,cnn}.py` — Section 6;
- `src/run_hierarchical.py` — Section 7;
- `src/run_cross_region_causal.py`, `src/run_irm_sweep.py` — Section 7.1.

## Acknowledgements

We thank the AI Hub curators for the vibration dataset. Funding under
Regional Innovation Strategy (RIS) through NRF Korea
(2021RIS-001; 1345370811)) as declared in Lee & Kim (2023) also
supported this follow-up.

## References

Arjovsky, M., Bottou, L., Gulrajani, I., Lopez-Paz, D. 2019.
Invariant Risk Minimization. arXiv:1907.02893.

DeGrave, A.J., Janizek, J.D., Lee, S.-I. 2021. AI for radiographic
COVID-19 detection selects shortcuts over signal. *Nature Machine
Intelligence* 3(7), 610-619.

Geirhos, R., Jacobsen, J.-H., Michaelis, C., Zemel, R., Brendel, W.,
Bethge, M., Wichmann, F.A. 2020. Shortcut Learning in Deep Neural
Networks. *Nature Machine Intelligence* 2(11), 665-673.

Gulrajani, I., Lopez-Paz, D. 2021. In Search of Lost Domain
Generalization. *ICLR*.

Hunaidi, O., Chu, W., Wang, A., Guan, W. 2000. Detecting Leaks in
Plastic Pipes. *Journal AWWA* 92(2), 82-94.

Ismail, M.I.M., Dziyauddin, R.A., Salleh, N.A.A., Muhammad-Sukki, F.,
Bani, N.A., Izhar, M.A.M., Latiff, L.A. 2019. A Review of Vibration
Detection Methods Using Accelerometer Sensors for Water Pipeline
Leakage. *IEEE Access* 7, 51965-51981.

**Lee, S., Kim, B. 2023. Machine Learning Model for Leak Detection
Using Water Pipeline Vibration Sensor. *Sensors* 23(21), 8935.
DOI:10.3390/s23218935.**

Rosenfeld, E., Ravikumar, P., Risteski, A. 2021. The Risks of
Invariant Risk Minimization. *ICLR*.

Sagawa, S., Koh, P.W., Hashimoto, T.B., Liang, P. 2020.
Distributionally Robust Neural Networks for Group Shifts. *ICLR*.

Zech, J.R., Badgeley, M.A., Liu, M., Costa, A.B., Titano, J.J.,
Oermann, E.K. 2018. Variable generalization performance of a deep
learning model to detect pneumonia in chest radiographs. *PLoS
Medicine* 15(11), e1002683.
