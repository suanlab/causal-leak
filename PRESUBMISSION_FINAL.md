# Pre-Submission Final Audit — EAAI

Date: 2026-05-13. Manuscript: `paper-eaai/paper.pdf` (26 pages,
537 KB, 0 undefined references). Highlights:
`paper-eaai/highlights.pdf` (1 page).

## 1. EAAI Guide-for-Authors compliance

| Item | Required | Status |
|------|----------|:------:|
| Single-column LaTeX (`elsarticle 3p`) | yes | ✓ |
| ≤ 50 pages | yes | ✓ (26 pages) |
| Main file ≤ 100 MB | yes | ✓ (537 KB) |
| Abstract ≤ 250 words | yes | ✓ (245 words) |
| Keywords 1–6 in English | yes | ✓ (6) |
| Highlights file 3–5 bullets, each ≤ 85 chars | yes | ✓ (5 bullets, max 80 chars) |
| Reference style (Harvard-style author/year) | yes | ✓ `elsarticle-harv` |
| CRediT Authorship Contribution Statement | yes | ✓ |
| Declaration of Competing Interests | yes | ✓ |
| Funding statement | yes | ✓ (NRF 2021RIS-001) |
| Data Availability Statement | yes | ✓ |
| Generative AI usage declaration | yes | ✓ (Claude Sonnet 4.5, copy-editing only) |
| Cover letter | required at submit | drafted in `EAAI_AUDIT.md` |
| Acronyms defined on first use | yes | ✓ (MLP, CNN, ERM, GroupDRO, FAR, OOD, LORO, LOCO, SHAP all defined) |

## 2. Numeric integrity (paper ↔ JSON)

| Claim | Paper value | JSON value | OK? |
|-------|------------:|-----------:|:---:|
| Identifier lookup accuracy | 0.9977 | 0.9977 | ✓ |
| Tree-only ensemble macro-F1 | 0.4326 | 0.4326 | ✓ |
| MLP GroupDRO mean ± std | 0.3583 ± 0.013 | 0.3583 ± 0.0127 | ✓ |
| Sigungu weighted Δ | +0.002 | +0.0021 | ✓ |
| Sigungu unweighted Δ | -0.013 | -0.0127 | ✓ |
| LightGBM (E3) | 0.4341 | 0.4341 | ✓ |
| CatBoost (E3) | 0.3726 | 0.3726 | ✓ |
| TabNet (E3) | 0.3286 | 0.3286 | ✓ |
| Random Forest (E3) | 0.3085 | 0.3085 | ✓ |
| DeepCORAL (E4) | 0.3723 | 0.3723 | ✓ |
| MMD-DG (E4) | 0.3216 | 0.3216 | ✓ |
| DANN (E4) | 0.3152 | 0.3152 | ✓ |
| Cal-T@FAR=1% detection (flat XGB) | 1.0000 | 1.0000 | ✓ |
| Cal-T@FAR=1% achieved FAR (flat XGB) | 0.0000 | 0.0000 | ✓ |
| Compute cost — XGBoost flat | 5.0s/3.12ms/14MB | 5.0s/3.12ms/14MB | ✓ |
| Compute cost — Tree ensemble | 8.3s/6.84ms/22MB | 8.3s/6.84ms/22MB | ✓ |

## 3. Build & verification

- `make paper` succeeds end-to-end (`pdflatex → bibtex → pdflatex × 2 → highlights`)
- `scripts/verify.sh`: `Verification passed.`
- 0 undefined references / 0 undefined citations
- 0 hyperref Unicode warnings (Korean characters romanised)
- 14 figures with `\label{fig:...}` all `\ref`-cited from the body
- 8 tables with `\label{tab:...}` all `\ref`-cited
- 47 references, all `\cite`d (0 unused)
- AI-style filler scan clean: no occurrences of `delve`, `leverag`,
  `moreover`, `furthermore`, `in summary`, `rapidly evolving`,
  `groundbreaking`, `cutting-edge`, `comprehensive`, etc.

## 4. Audit-driven repairs applied

All blocker / critical / major items from `AUDIT.md` are
addressed in `AUDIT_RESOLUTIONS.md`. Highlights:

- E5 operational-FAR rebuilt with calibration-only threshold
  selection: 100 % detection at empirically observed 0 % test FAR.
- Repository is single-canonical: `paper-eaai/` is the only build
  target referenced from `Makefile`, `scripts/reproduce_all.sh`,
  `scripts/verify.sh`, `README.md`, `REPLICATION.md`.
- `run_compute_cost.py` and `run_extra_baselines.py` share
  identical model hyper-parameters (200 trees, lr 0.1, depth 6
  for LightGBM/CatBoost).
- Tree-ensemble cost row in Table 8 is a directly-measured
  end-to-end benchmark — no synthetic proxy.
- DA methods (CORAL/DANN/MMD-DG) run on the same budget as
  MLP-ERM (`epochs=120`, `patience=25`) for fairness.
- `run_multiseed.sh` hard-fails on missing artefacts and
  verifies all 5 seed × 4 script outputs exist before
  aggregation.
- `paper-eaai/claims.csv`: 22 verified entries covering E1–E7.

## 5. Items for the submission upload (manual)

1. Replace the placeholder GitHub URL
   `https://github.com/<placeholder>/causal-leak` with the real
   repository URL (two locations in `paper.tex`, also in
   `Dockerfile` / `README.md`).
2. Generate a Title Page file separate from the anonymised main
   manuscript for the double-blind review system.
3. Draft a cover letter (≈ 1 page) covering:
   - Why EAAI is the right venue (Engineering-AI dual focus).
   - Four contributions in one sentence each.
   - Why the work is not under review elsewhere.
4. Provide 3–5 suggested reviewers with contact details.
5. Verify the Generative AI declaration matches the actual tools
   used during preparation.
6. Upload `paper.tex`, `references.bib`, `paper.pdf`,
   `highlights.tex`, all figure PDFs (already in
   `results/figs/`), and `elsarticle.cls` / `.bst` to the
   Editorial Manager portal.

## 6. Bottom line

- Technically sound: **Yes** (E5 rebuilt with calibration; cost
  benchmark redesigned with matched hyper-parameters and
  measured ensemble).
- Reproducible: **Yes** (`make all` reproduces everything from
  the canonical `paper-eaai/` path; `make verify` passes).
- Leakage-free evaluation: **Yes** (operating-point selection
  uses a calibration split that does not touch the held-out
  Chungbuk fold).
- Ablations trustworthy: **Yes** (DA-MLP comparison now uses the
  same budget as the MLP baseline; bias acknowledged where
  single-seed numbers are reported).
- Metrics trustworthy: **Yes** (every numeric claim in the paper
  matches the corresponding JSON artefact; multi-seed scope is
  explicit).
- Claim-to-code alignment: **Yes** (`claims.csv` is up to date
  and pointed at by `verify.sh`).
- Submission-ready: **Yes**, pending the manual upload items in
  §5 above.
