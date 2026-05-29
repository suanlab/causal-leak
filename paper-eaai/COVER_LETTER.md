# Cover Letter for EAAI Submission

**To:** Editor-in-Chief, *Engineering Applications of Artificial
Intelligence*
**From:** Suan Lee, School of Computer Science, Semyung University
**Date:** 2026-05-17
**Manuscript:** *Shortcut Audit, Causal Bands, and an
Out-of-Distribution Recipe for Vibration-Based Water Leak Detection*

---

Dear Editor,

I am pleased to submit the manuscript above for consideration at
*Engineering Applications of Artificial Intelligence* as a regular
research article in the AI-for-engineering-systems track.

## Why EAAI is the right venue

The paper has a dual character that matches EAAI's stated scope of
"AI methodology...applied to engineering problems." The
methodological half formalises a reusable four-probe identifier-shortcut
audit (Definition 2), a three-tier out-of-distribution evaluation
ladder (Definition 3), and a grouped permutation importance with
multi-region bootstrap (Definition 4); these are deliberately
benchmark-agnostic and applicable to any sensor-ML pipeline with
a group hierarchy and physically groupable features
(rotating-machinery condition monitoring, gas-pipe leak detection,
partial-discharge classification, sound-based fault diagnosis).
The engineering half delivers a deployable leak-typing recipe
for municipal water-utility networks: 57 causal features
(65--422 Hz + 20 MAX aggregates + month), a hierarchical
classifier, a pre-specified tree-only soft-probability ensemble,
and a full computational-cost benchmark showing 5 s training and
3.3 ms / 1{,}000-row inference on CPU.

## Three flagship contributions

1. **Identifier shortcuts are catastrophic in sensor benchmarks.**
   On the AI Hub water-pipe vibration dataset
   (70{,}384 recordings, 49 sites, 12 administrative regions), a
   four-rule lookup ignoring the entire vibration spectrum
   reproduces 99.77\,\% validation accuracy, within 0.02
   percentage points of the published state of the art for that
   benchmark---which therefore was measuring sensor-ID memorisation
   rather than physical leak signal.

2. **The causal leak band is 121--422 Hz.** Three independent
   attribution methods (bootstrap permutation importance with
   95\,\% CIs, TreeSHAP, and four-region replication) converge on
   the same band, and the 422--1470 Hz range is exposed as an
   active confounder whose removal *improves* out-of-distribution
   macro-F1. The finding is consistent with published acoustics
   for plastic pipes, providing external corroboration.

3. **Causal feature selection beats loss-level invariance
   regularisers.** Across multi-seed evaluation of IRM, GroupDRO,
   DeepCORAL, DANN, MMD-DG, and Permutation-Guided Attention on
   MLP and CNN backbones, no method achieves a statistically
   significant lead over plain ERM at $\alpha = 0.05$, and all
   trail the causal-XGBoost baseline by 6--11 percentage points.
   The tree-only soft-probability ensemble attains OOD macro-F1
   0.4326 with perfect normal-class recall on the canonical
   region holdout, deployable at 6.8 ms / 1{,}000-row inference.

## Reproducibility

All code, multi-seed runner, 30+ result JSONs (one per claim in
\texttt{claims.csv}), and figure-generation scripts are released
under the MIT licence. A Dockerfile and a single-shell-command
replication recipe (\texttt{REPLICATION.md}) reproduce every
figure and table end-to-end.

## Anticipated reviewer questions and pre-emptive answers

- **"Does this generalise beyond AI Hub?"** Section 5 discusses a
  partial replication on an external sensor benchmark (in
  progress, planned for a follow-up technical note); the
  methodology itself (Definitions 1--4) is dataset-agnostic, and
  the applicability checklist (Section 3.10) identifies the
  pre-conditions any candidate benchmark must satisfy.
- **"Why a single-author study?"** All numerical claims are
  cross-checked against released JSONs in the integrity audit
  (supplementary material), and the released harness is designed
  for one-command external replication.
- **"Why not a Transformer-based model?"** On $n = 70{,}384$ rows,
  the Borisov et al. (2024) survey suggests tabular DL rarely
  exceeds gradient-boosted trees; this is empirically confirmed
  here (TabNet 0.329 vs XGB causal 0.427), although future work
  with a self-supervised spectrogram backbone is identified.

## Suggested reviewers

(Five names with affiliations + emails, chosen for active
publishing in EAAI-adjacent areas and no prior collaboration with
the author.)

1. **Prof. Yan Gao** — University of Macau, Faculty of Science
   and Technology. Acoustic/vibration leak detection in plastic
   water pipes. Published the 2017 Applied Acoustics paper cited
   as \texttt{gao2017model}.
   *Email: yangao@um.edu.mo*

2. **Prof. Tarek Zayed** — Hong Kong Polytechnic University,
   Department of Building and Real Estate. ML-based leak
   detection in water distribution networks
   (\texttt{tariq2022leak}).
   *Email: tarek.zayed@polyu.edu.hk*

3. **Prof. Tatsunori Hashimoto** — Stanford University, Computer
   Science. GroupDRO author (\texttt{sagawa2020groupdro});
   uniquely qualified to assess the deep-baseline comparison.
   *Email: thashim@stanford.edu*

4. **Dr. Robert Geirhos** — Google DeepMind. Shortcut-learning
   framework author (\texttt{geirhos2020shortcut}); evaluator
   for the audit-protocol contribution.
   *Email: rgeirhos@google.com*

5. **Prof. Dukhwan Yoon** — Seoul National University,
   Construction & Environmental Engineering. Korean smart-water
   and AI Hub data ecosystem. Local domain anchor.
   *Email: ydh@snu.ac.kr*

## Conflicts of interest and prior submission

The author declares no conflict of interest. This manuscript
has not been submitted to any other journal. The author's prior
work on this dataset (Lee \& Kim 2023, *Sensors*) is explicitly
identified as the motivation and is properly cited; the present
study is a methodologically distinct follow-up that audits and
restructures the earlier benchmark.

## Generative-AI disclosure

In accordance with Elsevier's 2024+ policy, generative-AI
assistance (Claude Sonnet 4.5) was used for English copy-editing
and LaTeX formatting. After using these tools the author
reviewed and edited every paragraph as needed and takes full
responsibility for the manuscript's content.

I look forward to the reviewers' feedback.

Sincerely,

Suan Lee
suanlee@semyung.ac.kr
ORCID: 0000-0002-3047-1167
