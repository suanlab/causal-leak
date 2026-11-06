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

> **Correction (November 2026).** An earlier version of this note read `lrate`
> as an annotation-time leak-likelihood grade. The AI Hub dataset page
> (dataset 138, v1.0, 2021-06-25) defines it differently:
>
> *lrate (누수 확률): "일일 새벽 시간대 2시간동안 총 10번의 누수 감지 여부를
> 체크하고 일정 기준치이상 반응한 결과를 확률로 표시"* -- the sensor checks for
> a leak 10 times during a two-hour early-morning window each day, and `lrate`
> is the share of those checks whose response exceeded a fixed threshold.
>
> `llevel` is the mean magnitude of the above-threshold responses, and the 20
> `MAX` columns are the peak frequency and peak amplitude of each of the 10
> checks. Labels are documented as confirmed by field leak-survey specialists
> through detection and excavation.

So `lrate` is the **device's own detector output**, computed from the signal,
and it is not an annotation. Every `in`, `out`, `noise` and `other` recording
has `lrate = 90`; every `normal` recording has 30-70.

The 20 `MAX` columns alternate: even index = peak frequency, odd index = peak
amplitude, one pair per check. Counting the checks whose peak amplitude is at
least 130 (`src/run_device_rule.py`) recovers `lrate` for normal recordings:
the count never exceeds `lrate` and equals it for 78 % of them. For non-normal
recordings `lrate` is 90 whatever the recovered count (8, 9 or 10; 45,486 of
48,218 are at 10). Whether 90 is a cap or was assigned afterwards cannot be
determined from the release. What can be determined is that the class
boundary coincides with the trigger count.

That boundary needs no learning. The rule "non-normal iff at least k of the 10
per-check peak amplitudes are >= T", with (T, k) chosen by balanced accuracy on
the training regions under leave-one-region-out, selects T = 130 and k = 8 in
every fold. It flags 0 of 22,166 normal recordings and misses 84 of 48,218
non-normal ones. On the four regions that can measure false alarms (27, 29,
44, 45) that is 0 of 22,121 normals (0 of 1,239 sensors) and 8 of 4,005
non-normals missed. Any learned normal-vs-rest gate on the 534-column signal
set has these columns as input, which explains the AUC of 1.000 in every
region and every mixed installation.

`lrate` is excluded from the 534-column `signal` set used for every detector
here. It is one of the identifier and metadata columns in the 544-column
`full` set. It is also not a site proxy: every one of the 8 normal-bearing
installations carries all five normal values (30-70) in the same decreasing
proportions. Collection dates of normal recordings (2019-10 to 2021-01) overlap
those of leak-only installations (2019-02 to 2021-02).

The page documents 50,000 recordings (in/out 10,000 each, noise/other 5,000
each, normal 20,000). The release used here has 70,384, with 22,166 normal. The
page gives no sampling rate, recording duration or cadence, and does not say
how normal recordings were selected.

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

The perfect separation is not a single-feature threshold: the best single
feature (`MAX1`) reaches AUC 0.84-0.91 inside these installations. It is a
threshold on a count over the 10 per-check amplitudes (see *What `lrate` is*),
which is constant across sites, so this test cannot detect it.

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

The detector never sees `lrate`. It does see the spectrum and the per-check
maxima from which the device computed `lrate`. The dose-response is what one
expects if the detector has re-learned the device's threshold rule: normals
whose checks fired 7 of 10 times score highest. The only normal recordings
scored above 0.5 had `lrate = 70`.

## Conclusion

- **Site confound: not supported.** Perfect separation persists inside every mixed
  installation with the whole region held out.
- **Campaign confound: not supported.** Leak recordings at normal-bearing
  installations are not consistently distinguishable from other leak recordings.
- **Device-trigger shortcut: confirmed.** `lrate` is the device detector's hit
  rate. A two-parameter count rule on the device's own per-check amplitudes,
  chosen on training regions only, reproduces the normal/non-normal boundary to
  within 84 of 70,384 recordings with no training. Normal-vs-rest "detection"
  on this benchmark reproduces the device's trigger logic. It cannot show
  that anything detects leaks among recordings the device did not pre-select.
- **Residual sensor-level limit:** no sensor ever carries both labels, so a
  difference in how leak sensors and normal sensors were recorded cannot be
  excluded either.
