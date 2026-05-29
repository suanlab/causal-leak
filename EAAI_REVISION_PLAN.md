# EAAI Revision Plan — Substantial Upgrade for *Engineering Applications of AI*

**Target venue**: Elsevier *Engineering Applications of Artificial
Intelligence* (EAAI), IF ≈ 8.0, CiteScore ≈ 13.6.
**Current state**: 19-page Sensors-targeted manuscript (`paper.tex`),
20 figures/tables, 21 references, full reproducibility package.
**Goal**: ~30-page EAAI full paper with substantially expanded
methodology, stronger experimental rigour, broader baselines,
deeper engineering analysis, and 50–70 references.

---

## 1. Target Venue Profile

| Item | EAAI standard | Implication for us |
|------|----------------|--------------------|
| Audience | Applied-AI researchers + industrial ML practitioners | Dual content: AI methodology + engineering deployment |
| Typical length | 25–35 typeset pages | Expand from current 19 pp |
| References | 50–80 | Expand from current 21 |
| Methodology depth | Formal definitions + theoretical justification | Add formal framework for shortcut audit and OOD protocol |
| Baselines | 6–10 SOTA comparisons | Add TabNet, CORAL, DANN, MMD, multi-seed runs |
| Statistical rigour | Mean ± std over seeds, significance tests | Add multi-seed re-runs and paired tests |
| Engineering impact | Cost/benefit, deployment scenarios | Add operational analysis |
| Reproducibility | Code/data + replication instructions | Add Docker/Conda + step-by-step replication appendix |
| First-decision time | 3–5 months (typical) | Plan internal review and submission carefully |
| Format | Elsevier `elsarticle.cls` (default) | Convert from MDPI `mdpi.cls` |

---

## 2. Gap Analysis (Current vs. EAAI Bar)

| Dimension | Current | EAAI bar | Gap |
|-----------|---------|----------|----:|
| Length (pages) | 19 | 25–35 | +6–16 |
| References | 21 | 50–80 | +29–59 |
| Baselines compared | XGB, MLP+ERM/IRM/DRO, CNN+ERM/IRM/DRO, PGA | + TabNet, CORAL/DANN/MMD, additional tree models | +3–5 methods |
| Multi-seed reporting | Single seed (42) | Mean ± std over ≥3 seeds | All key tables |
| Statistical tests | None | Paired tests, CIs on every metric | Major add |
| Methodology formalisation | Practical protocols | Formal pseudocode + theory | New §3 subsection |
| 시군구-level OOD | Not evaluated | Bonus stress test | New experiment |
| Computational cost | Briefly mentioned | Full table (training, inference, memory) | New table |
| Cost-benefit analysis | None | Required for EAAI | New §8 subsection |
| External dataset replication | None | Strongly recommended | Search + try one |
| Failure-mode analysis | Brief limitations | Dedicated subsection | Expand |
| Reference expansion needs | Shortcut/OOD/leak/AI-engineering | All four | Major add |

---

## 3. Revision Goals (High-level)

1. **Position the methodology as the primary contribution.** The
   shortcut-audit protocol and the three-tier OOD evaluation must
   read as reusable AI tools applicable to *any* sensor benchmark,
   with the AI Hub leak dataset as one carefully studied case.

2. **Tighten every empirical claim** with multi-seed runs, paired
   significance tests, and explicit confidence intervals. Numbers
   that cannot survive that pass should be hedged or withdrawn.

3. **Broaden the baseline panel.** Add TabNet, CORAL/DANN/MMD-based
   domain alignment, and 1–2 modern boosting variants so reviewers
   cannot say "they only compared one tree against a basic MLP."

4. **Quantify the engineering payoff.** Translate macro-F1 numbers
   into operational metrics: detection rate at fixed false-alarm
   budget, expected daily false alarms, computational cost per
   1,000 readings, memory footprint, integration paths with
   existing SCADA/IoT layers.

5. **Strengthen reproducibility** to a level reviewers can act on —
   pinned environment, Docker image, single-command replication,
   appendix tables of fold-level numbers, expected runtime.

6. **Acknowledge limitations explicitly** with a dedicated section
   (single benchmark, Korean-specific deployment, single seed for
   some experiments, etc.) and concrete future work.

---

## 4. Work Plan in Phases

Estimated end-to-end effort: **8 weeks** of focused work for a single
author. Each phase has a clear deliverable so the project stays
checkpointable.

### Phase 0 — Week 1 — Baseline freeze and venue preparation
- [ ] Tag the current Sensors-targeted paper in git as `v0.9-sensors`
      so the EAAI revision proceeds on a separate branch.
- [ ] Read 3–5 recent EAAI papers in adjacent domains (e.g., AI for
      structural health monitoring, AI for water leak detection,
      OOD evaluation in industrial AI) to calibrate tone, length,
      figure density, and reference patterns.
- [ ] Download Elsevier `elsarticle.cls` and adapt the manuscript
      shell. Verify a minimum-viable build with one figure and one
      table before the heavy rewrite begins.
- [ ] Convert the existing fact-checked claims into a single
      `claims.csv` (claim, source JSON, value, uncertainty) so each
      can be re-verified after the multi-seed re-runs.

### Phase 1 — Weeks 2–3 — New experiments
Run them parallel-friendly on the existing 2-GPU machine.

- [ ] **E1. Multi-seed re-runs** of every stochastic experiment
      (Phase 2B IRM/GroupDRO, Phase C augmentation, Phase B PGA)
      with seeds `{42, 0, 1, 2, 3}`. Report mean ± std; replace the
      current single-seed numbers in the manuscript.
- [ ] **E2. 시군구-level OOD ablation**. Treat each 시군구 as an
      environment (≈ 40+) instead of 광역시도 (12). Re-run causal
      XGBoost and MLP-GroupDRO. Tests whether more environments
      help GroupDRO or IRM under the same dataset.
- [ ] **E3. Additional tabular baselines** on the canonical
      Chungbuk LORO fold:
      `LightGBM`, `CatBoost`, `TabNet`, `Random Forest` (already
      have).
- [ ] **E4. Domain-alignment baselines**:
      `DeepCORAL`, `DANN`, `MMD-Mixup`. These are GPU baselines but
      typically train in <30 min each.
- [ ] **E5. Cost-aware metric**. Define
      *detection-rate-at-1%-false-alarm* and re-evaluate every
      tabled model under it. Useful for the EAAI engineering
      readership.
- [ ] **E6. Computational cost table**. Train time (s), inference
      time per 1,000 rows (ms), peak RAM (MB), peak GPU memory (MB)
      for each of the top 8 models. Run on a clean machine snapshot.
- [ ] **E7. Hierarchical-threshold sensitivity sweep**. Stage-1
      threshold ∈ {0.3, 0.4, 0.5, 0.6, 0.7}; tabulate
      false-alarm vs missed-leak tradeoff.

### Phase 2 — Week 4 — Methodology formalisation
- [ ] Re-write §3 (Methodology) with formal definitions:
      - **Definition 1** (sensor benchmark with identifier shortcut)
      - **Definition 2** (four-probe shortcut audit) — algorithm box
      - **Definition 3** (three-tier OOD evaluation) — protocol box
      - **Definition 4** (grouped permutation importance) — equation
- [ ] Add a **theoretical framing** subsection: under the standard
      i.i.d. assumption, identifier shortcuts produce zero excess
      risk under row-CV but bounded-below excess risk under
      group-CV; we make this informal argument explicit.
- [ ] Add an **applicability checklist** identifying when our
      methodology applies (any sensor benchmark with id-like
      column + group hierarchy + groupable physical features).

### Phase 3 — Week 5 — Manuscript restructure
- [ ] Restructure section flow for EAAI's expected ordering:
      1. Introduction
      2. Related Work (expanded; see Phase 4)
      3. Problem Formulation and Methodology
      4. Dataset and Experimental Setup
      5. Results
      6. Engineering Implications and Deployment Recipe
      7. Discussion: Threats to Validity, Failure Modes
      8. Conclusions and Future Work
- [ ] Move Figure 1 (Overview) right after the Introduction's
      contributions list (already done — keep).
- [ ] Add a *Practitioner Recommendations* boxed callout near the
      end of §6.
- [ ] Consolidate the strengthening phases (current §6.4–§6.9) into
      a clean Results structure where each subsection answers a
      named research question (RQ1–RQ7).

### Phase 4 — Week 6 — Reference expansion (target: 50–70)
- [ ] AI methodology citations to add (≈15):
      - Domain generalisation: DeepCORAL, DANN, MMD-DG, fishr,
        recent surveys.
      - Spurious correlation: Sagawa 2020 (already), Liu et al.
        2021 worst-group, JTT.
      - Tabular DL: TabNet, FT-Transformer, NODE, recent surveys.
      - Causal feature attribution: Janzing et al., causal Shapley.
- [ ] Engineering-domain citations to add (≈20):
      - Recent (2023–2026) leak-detection ML papers from MSSP,
        Water Research, Automation in Construction, Sensors.
      - Pipe acoustics references that pin the 121–422 Hz physical
        interpretation (Hunaidi follow-ups, Gao 2017,
        Almeida 2018, etc.).
      - Korean smart-water-grid policy references for engineering
        impact framing.
- [ ] Reproducibility/benchmarking citations to add (≈8):
      - DomainBed, WILDS, Underspecification, ML reproducibility
        crisis surveys.
- [ ] Cross-check every existing reference for venue/year accuracy
      one more time.

### Phase 5 — Week 7 — Engineering implications expansion
- [ ] Add §6 (or §7) "Engineering Implications and Deployment Recipe":
      - Operational metric translation
      - Cost-benefit estimate (e.g., for a 50,000-sensor utility:
        baseline false-alarm cost vs deploying our recipe)
      - Computational-cost table (from E6)
      - Integration scenarios with SCADA / IoT MQTT pipelines
      - Failure modes and mitigations (sensor drift, seasonality,
        new pipe materials)
- [ ] Add a *Compliance and standards* paragraph: how the audit
      protocol relates to AI risk-management frameworks (NIST
      AI RMF, EU AI Act high-risk classification of infrastructure
      AI).

### Phase 6 — Week 8 — Polish and submission
- [ ] Internal review pass: read the whole paper aloud, fix
      awkward phrasing.
- [ ] Run `chktex` and `lacheck` on the LaTeX source.
- [ ] Run a fresh integrity audit (extend `INTEGRITY_AUDIT.md`):
      plagiarism scan against newly added text, hallucination
      check on new references, AI-style filler check on rewritten
      sections.
- [ ] Prepare cover letter highlighting:
      - Why EAAI is the right venue
      - Three flagship contributions
      - How we addressed reviewer concerns from the prior
        Sensors-track preparation (anticipate)
- [ ] Prepare suggested reviewers (3–5 EAAI-published authors in
      adjacent domains).
- [ ] Submit via EM (Elsevier Editorial Manager).

---

## 5. New Experiments — Detailed Specifications

| ID | Experiment | Compute | Days |
|----|-------------|---------|-----:|
| E1 | Multi-seed re-runs of 12 stochastic configs × 5 seeds | 2 GPU, ~12 h total | 1 |
| E2 | 시군구-level OOD (XGBoost causal, MLP-DRO, CNN-best) | mostly CPU, ~6 h | 1 |
| E3 | TabNet, LightGBM, CatBoost, RF on Chungbuk fold | CPU/GPU, ~2 h each | 1 |
| E4 | DeepCORAL, DANN, MMD on causal features | 2 GPU, ~3 h each | 2 |
| E5 | Detection-rate-at-1%-FAR re-evaluation | CPU, <1 h (post-hoc) | 0.5 |
| E6 | Wall-clock + memory benchmark (top 8 models) | clean snapshot, ~3 h | 1 |
| E7 | Stage-1 threshold sweep with FN/FP curves | CPU, <1 h | 0.5 |

Total new compute: ≈ 28 h wall-clock on the 2-GPU machine, runnable
overnight in two batches.

Deliverables per experiment: one new JSON in `results/`, one new row
or panel in the manuscript, fold-level numbers in `appendix.tex`.

---

## 6. Manuscript Restructure Map

| Current section | EAAI section | Action |
|-----------------|---------------|--------|
| §1 Introduction | §1 Introduction | Tighten; add EAAI-relevant motivation |
| §2 Related Work | §2 Related Work | Expand 3× (see Phase 4) |
| §3 Dataset | §4 Dataset | Move after Methodology |
| §4 Methodology | §3 Problem Formulation and Methodology | Expand with formal definitions |
| §5 Experimental Setup | §4.2 Experimental Setup | Merge with dataset, add E6 cost table |
| §6 Results | §5 Results | Restructure as RQ1–RQ7; integrate multi-seed |
| §7 Discussion | §6 Engineering Implications | Reframe as engineering-impact section |
| (new) | §7 Threats to Validity | Pull current limitations + expand |
| §8 Conclusions | §8 Conclusions and Future Work | Add 3–4 future directions |
| (new) Appendix A | Hyperparameters and seed-level tables | Move appendix-worthy detail |
| (new) Appendix B | Step-by-step replication recipe | New |

---

## 7. Reference Expansion Targets

```
Existing (21)
├── Lee & Kim 2023 [keep]
├── Domain generalisation (4): Arjovsky, Sagawa, Gulrajani, Rosenfeld, Krueger [keep]
├── Shortcut/spurious (5): Geirhos, Zech, DeGrave, Beery, D'Amour [keep]
├── Leak detection (4): Hunaidi 2000, Ismail 2019, Shukla 2020, Fereidooni 2021 [keep]
├── ML basics (3): XGBoost, RF, LIME or SHAP [keep]
├── Tabular DL (3): SpecAugment, Mixup, Focal [keep]

To add (~30+)
├── Tabular DL: TabNet (Arik 2021), FT-Transformer (Gorishniy 2021),
│   NODE (Popov 2020), tabular DL surveys (Borisov 2024)
├── Domain alignment: DeepCORAL (Sun 2016), DANN (Ganin 2016),
│   MMD-DG, Fishr (Rame 2022), JTT (Liu 2021)
├── Recent leak-detection ML: 5–8 papers from MSSP/Water
│   Research/Automation in Construction (2023–2026)
├── Pipe acoustics: Almeida 2018, Gao 2017, additional Hunaidi work
├── ML benchmarking: WILDS (Koh 2021), DomainBed (Gulrajani 2021,
│   already), reproducibility surveys (Pineau 2019, Henderson 2018)
├── AI risk frameworks: NIST AI RMF, EU AI Act high-risk article
├── Korean smart-water policy / KICT or K-water reports (2–3)
├── Industrial AI deployment: 2–3 EAAI papers in adjacent domains
   for the "AI for engineering" framing
```

Target: 55–65 references at submission.

---

## 8. Figure / Table Plan Update

Existing figures: 8 (Overview + 7 results plots). All vector.

To add:
- **Fig. 9** — Cost-accuracy tradeoff curve (E5/E6 results).
- **Fig. 10** — Stage-1 threshold sweep with operating-point
  selection (E7).
- **Table 11** — Computational-cost benchmark (E6).
- **Table 12** — Multi-seed mean ± std for the headline 12 configs.
- **Table 13** — Domain-alignment baseline comparison (E4).
- **Appendix Table A1** — Full per-fold per-seed results.
- **Appendix Table A2** — Hyperparameter grid.

---

## 9. Quality Assurance Checklist (pre-submission)

- [ ] LaTeX builds with 0 errors / 0 undefined references
- [ ] Every numeric claim re-verified against the post-revision JSONs
- [ ] All references have working DOI / URL
- [ ] No verbatim phrase ≥ 8 words copied from any reference
- [ ] All figures vector (PDF or TikZ); 300+ DPI raster only when
      essential (heatmap colour blocks, etc.)
- [ ] Each table has thousand separators, consistent decimal precision
- [ ] All hyperparameters reported (training, model, evaluation)
- [ ] Code repository: pinned environment, Dockerfile, replication
      shell script, MIT licence
- [ ] Cover letter, suggested reviewers, declarations of interest
- [ ] Author Contributions (single-author: combined statement)
- [ ] CRediT taxonomy (single-author: all roles)
- [ ] Data Availability Statement points to AI Hub + GitHub
- [ ] GenAI usage disclosed in Acknowledgements (per Elsevier 2025+)

---

## 10. Risk Register

| Risk | Likelihood | Mitigation |
|------|:---------:|------------|
| Multi-seed re-runs change the headline numbers | Medium | Lock the strongest claims to the *direction* (sign of Δ), not the exact value |
| Domain-alignment baselines beat causal-XGBoost | Low | If they do, re-frame the paper around them — it's still our methodology |
| EAAI desk-rejects on scope mismatch | Low–Med | Include a paragraph in the cover letter explicitly mapping our work to EAAI scope statement |
| Reference list grows too unfocused | Medium | Cap related work at 65; require each citation to support an explicit claim |
| Reviewer requests 외부 데이터셋 검증 (external dataset) | High | Pre-emptively try MIMII or ToyADMOS in the *limitations* — even a partial replication strengthens the paper |
| Reviewer questions "Why not Transformers / TabPFN?" | High | Add a short ablation or explicit justification (n=70k is small for transformer pre-training) |
| Korean labels / locale issues in TikZ | Low | Already romanised; double-check before submission |

---

## 11. Top-line Schedule

| Week | Focus | Output |
|:----:|-------|--------|
| 1 | Plan, freeze, EAAI shell, claims.csv | Branch + minimal-build PDF |
| 2 | E1 (multi-seed) + E3 (tabular baselines) | New JSONs, updated tables |
| 3 | E2 (시군구) + E4 (domain alignment) | New JSONs, updated figures |
| 4 | Methodology formalisation, theoretical framing | Rewritten §3 |
| 5 | Restructure §1–§5 to EAAI flow; add E5/E6/E7 | First full revised draft |
| 6 | Reference expansion + figure additions | Reference list at 55+ |
| 7 | §6 engineering implications + §7 threats | Second full draft |
| 8 | Polish, integrity audit, cover letter, submit | Submission |

---

## 12. Definition of Done for the Revision

The revision is **ready to submit to EAAI** when *all* of the
following are true:

1. The manuscript is 25–35 typeset pages in `elsarticle` format,
   compiles cleanly under `pdflatex` with 0 errors and 0 undefined
   references.
2. Every headline claim is supported by ≥3-seed evidence with a
   reported confidence interval.
3. The methodology section includes formal definitions, an
   algorithm box, and an applicability checklist.
4. The Results section answers ≥7 named research questions, each
   with at least one figure or table.
5. The Engineering Implications section includes a
   computational-cost table and a cost-benefit estimate.
6. References are ≥55, all verified, no fabricated entries.
7. Reproducibility appendix lists every hyperparameter and an
   end-to-end replication script.
8. Pre-submission integrity audit passes (plagiarism / hallucination
   / logic / AI-style).
9. Cover letter, suggested reviewers, and declarations are drafted.
10. A senior peer (or self-review with at least 48 hours' distance)
    has read and signed off.

---

## 13. Immediate Next Steps (this session or next)

1. Decide the working branch / git tag for the EAAI revision and
   create a clean working folder structure (`paper-eaai/`).
2. Download Elsevier `elsarticle.cls` and produce a minimum-build
   PDF with placeholder content to confirm the toolchain.
3. Lock the multi-seed list `{42, 0, 1, 2, 3}` and build a single
   shell script that re-runs E1 in one command.
4. Draft the new Section 3 outline (formal Methodology) on paper
   first; submit a 1-page outline for review before rewriting.

