# EAAI Guide-for-Authors Audit (2026-05-13)

Compared the current `paper-eaai/paper.tex` against the official
EAAI Guide for Authors fetched from
`https://www.sciencedirect.com/journal/engineering-applications-of-artificial-intelligence/publish/guide-for-authors`.

## CRITICAL — must fix before submission

| # | Requirement | Current state | Fix |
|---|-------------|---------------|-----|
| C1 | "Papers must be formatted in single-column format" (desk-rejection criterion) | `\documentclass[5p,times,twocolumn]` — double-column | Switch to `[3p,times]` (single-column preprint) |
| C2 | Abstract ≤ 250 words | Current ~370 words | Trim to ≤ 250 |
| C3 | Highlights file (3–5 bullet points, ≤ 85 chars each) | Missing | Create `highlights.tex` |
| C4 | Declaration of Competing Interests (mandatory) | Missing | Add explicit section |
| C5 | Funding statement (mandatory) | Missing (MDPI one removed during port) | Add `\section*{Funding}` |
| C6 | CRediT author contributions statement | Missing | Add CRediT block |
| C7 | No undefined acronyms in title/abstract (desk-rejection) | "MMD-DG", "DANN", "DeepCORAL" used without expansion | Expand or remove from abstract |
| C8 | Abstract must clearly distinguish AI contribution vs engineering application (desk-rejection) | Implicit | Add a clarifying sentence |

## MEDIUM — should fix

| # | Requirement | State | Fix |
|---|-------------|-------|-----|
| M1 | Title page separate from anonymized manuscript (double-blind) | Single combined file | Build `title_page.tex` (deferred until submission upload) |
| M2 | Data availability statement | Present (\section*{Code and Data Availability}) | Rename to "Data Availability" + register dataset DOI |
| M3 | Suggested reviewers (cover letter) | Not yet drafted | Defer to submission cover letter |
| M4 | References Harvard style (author-year) | Using elsarticle-harv | OK |
| M5 | Manuscript ≤ 50 pages | 10 pages | OK |
| M6 | Keywords 1–6 in English, no multi-word phrases with "and"/"of" | "AI for engineering systems" violates | Replace |

## CONFIRMED OK

- LaTeX template: `elsarticle.cls` provided by Elsevier
- Reference list: 41 verified entries
- Vector-only figures (PDF / TikZ)
- All figures have captions and are cited in text
- Data partitioning fixed (split-seed=42) with multi-seed model-seed sweep
- No PNG / raster dependencies in main figures
- DOIs included for cite-checkable entries
- 5-seed mean ± std reported for stochastic experiments
- Single author (Suan Lee), all CRediT roles will fall on him
- GenAI declaration already present (acknowledgements section)

## Action items applied in this commit

1. Switched `documentclass` from `[5p,times,twocolumn]` to
   `[3p,times]` (single-column).
2. Compressed abstract to 244 words while preserving the four
   contribution statements.
3. Expanded MMD-DG / DANN / DeepCORAL on first abstract use.
4. Added "AI contribution / engineering application" distinguishing
   sentence to the abstract per desk-rejection criterion 2.
5. Created `paper-eaai/highlights.tex` (5 bullets, all ≤ 85 chars).
6. Added explicit `\section*{Funding}`, `\section*{Declaration of
   Competing Interests}`, `\section*{CRediT Authorship Contribution
   Statement}` sections.
7. Renamed `\section*{Code and Data Availability}` to `\section*{Data
   Availability}` to match guide phrasing.
8. Replaced keyword "AI for engineering systems" with single-word
   alternative "smart-water-grid".
