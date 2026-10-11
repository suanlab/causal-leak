# Binary Detection Confound Check — Site, Campaign, and `lrate`

Raised by the independent review of Paper B (issue 7): no sensor carries both
labels (0 of 2,710), normal recordings exist at only 8 of 49 installations, and
`lrate` is 90 for every leak recording and 30-70 for every normal one. Could
binary normal-vs-leak detection be recognising *which kind of installation* a
recording came from, rather than leak noise?

- Script: `src/run_lrate_confound.py` (all tests use the full 534-column physical
  input, the most capable set, so the conservative choice for a confound test)
- Raw results: `results/lrate_confound_results.json`; log `results/run_lrate_confound.log`

## What `lrate` is

`DATASET_REPORT.md` established that `lrate` is assigned after labelling
(`lrate == 90 <=> leak` is a perfect rule). It is not a model input. It is also not
a site proxy: every one of the 8 normal-bearing installations carries all five
normal values (30-70) in the same decreasing proportions. Collection dates of
normal recordings (2019-10 to 2021-01) overlap those of leak-only installations
(2019-02 to 2021-02), so there is no sign of a separate collection campaign.

## A. Separation inside one installation

For each installation holding at least 30 normal and 30 leak recordings, the model
is trained with the installation's **entire region** held out and scored inside the
installation. A site signature is constant within an installation, so it cannot
separate the classes there.

| Installation | Region | Normal / leak recs | Normal / leak sensors | AUC (rec) | AUC (sensor) |
|---|---:|---:|---:|---:|---:|
| S-2711010100 | 27 | 4,566 / 342 | 169 / 10 | 1.0000 | 1.0000 |
| S-4372025021 | 43 | 45 / 4,753 | 5 / 135 | 1.0000 | 1.0000 |
| S-4413110100 | 44 | 1,151 / 531 | 110 / 13 | 1.0000 | 1.0000 |
| S-4521010100 | 45 | 1,624 / 594 | 207 / 19 | 1.0000 | 1.0000 |
| S-4571025029 | 45 | 10,675 / 396 | 584 / 8 | 1.0000 | 1.0000 |

The perfect separation is **not** a single loudness or gain threshold. Without any
training, the best single feature (`MAX1`) reaches AUC 0.84-0.91 inside these
installations and the mean spectral level only 0.57-0.86. The model separates on a
multivariate spectral pattern. Normal recordings look alike everywhere (median mean
level 6.68-6.92 across the five sites); leak recordings vary widely (7.25-16.27).

## B. Campaign signature

Leak recordings only: can the spectrum tell those at the 8 normal-bearing
installations (6,682) from those at the 41 leak-only installations (41,536)?
Grouped 5-fold by installation.

Fold AUCs 0.463, 0.457, 0.497, 0.793, 0.602 — mean 0.56, with one outlying fold.
(The pooled out-of-fold AUC of 0.367 is not meaningful: it pools scores from five
differently calibrated models.) There is no consistent signature a detector could
use to recognise "a normal-bearing installation".

## C. `lrate` dose-response

Among 18,061 held-out normal recordings from test A (regions 27, 43, 44, 45 — not
region 29), the detector's leak score rises monotonically with `lrate`
(Spearman rho = 0.61):

| lrate | n | mean leak score | share scored >= 0.5 |
|---:|---:|---:|---:|
| 30 | 6,094 | 0.00005 | 0 |
| 40 | 4,087 | 0.00005 | 0 |
| 50 | 3,053 | 0.00011 | 0 |
| 60 | 2,493 | 0.00045 | 0 |
| 70 | 2,334 | 0.00921 | 0.043 % |

The detector never sees `lrate`, so this is agreement, not leakage. The most
consistent reading is that `lrate` is a graded, annotation-time leak-likelihood
score, and that normals graded 70 are acoustically closest to leaks. The only
normal recordings scored above 0.5 in this test were graded 70.

## Conclusion

- **Site confound: not supported.** Perfect separation persists inside every mixed
  installation with the whole region held out.
- **Campaign confound: not supported.** Leak recordings at normal-bearing
  installations are not consistently distinguishable from other leak recordings.
- **`lrate`: not a confound.** It agrees with the detector's score rather than
  driving it.
- **Residual limit that this data cannot resolve:** no sensor ever carries both
  labels, so a difference in how leak sensors and normal sensors were recorded
  cannot be excluded. The tests above rule out the site-level and campaign-level
  versions of that explanation, and a single gain difference, but not a
  sensor-level one. This must be stated as a limitation.
