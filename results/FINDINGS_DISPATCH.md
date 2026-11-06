# Crew Dispatch at Training-Calibrated Operating Points

Supersedes the argmax false-alarm comparison in `run_operational_folds.py`.
That comparison found 4 normal sensors with false alarms for the tree ensemble
against 14 for LightGBM (McNemar p = 0.021), and Paper B recommended the
ensemble on that basis. A simulated review (three reviewers and an editor,
November 2026) raised two objections, and both were verified:

1. A "false alarm" counted a normal recording typed as noise or other, and a
   leak typed as noise did not count as a miss. A utility sends a crew for an
   in/out leak, not for electrical noise.
2. Every candidate separates normal from the rest with AUC 1.000. The 4-vs-14
   gap therefore reflects where each model's argmax boundary falls under shift.
   It is not a property of the models.

Both are moot for the normal class: the device's own trigger count separates
it (`FINDINGS_LRATE_CONFOUND.md`).

- Scripts: `src/run_dispatch_operating_points.py <region>` (outer fit plus three
  region-grouped inner folds; per-recording probabilities in `results/dispatch_op/`),
  then `src/analyze_dispatch.py`.
- Results: `results/dispatch_operating_points_results.json`, `results/analyze_dispatch.log`;
  figure `results/figs/figB2_dispatch.pdf`.
- Outer-fold argmax predictions reproduce `operational_folds_results.json`
  confusion matrices exactly. Checked on region 45 by `paper-b/verify_claims.py`.

## Protocol

- **Dispatch:** a recording typed `in` or `out`. **Missed leak:** true in/out
  not dispatched. **False dispatch:** true normal dispatched. **Nuisance
  dispatch:** true noise/other dispatched.
- **Score:** `s = p(in) + p(out)`, the same for every model.
- **Operating points:** argmax, plus thresholds on `s` chosen on the training
  pool to hit 1, 5, 10 and 20 % miss. The training scores are out-of-fold from
  three region-grouped inner folds. GroupKFold is used, except for held-out
  region 45, where it left an inner training split with no normals, so regions
  were re-dealt with the normal-bearing ones spread across folds. Inner folds
  reuse the outer fold's bands, so calibration is mildly optimistic but never
  sees the test region.
- **Regions:** eight typing regions (27, 29, 43–48) give 31,928 in/out and
  13,555 noise/other recordings. False dispatches on normals are counted in
  regions 27, 29, 44 and 45 (22,121 recordings, 1,239 sensors, 7 installations).

## Results

| Operating point | Model | Missed (%) | Nuisance (%) | False dispatches (sensors) |
|---|---|---:|---:|---:|
| argmax | soft average | 15.8 | 73.8 | 3 (3) |
| argmax | LightGBM | 25.5 | 61.2 | 7 (7) |
| 5 % | soft average | 4.5 | 89.4 | 13 (11) |
| 5 % | LightGBM | 4.4 | 88.5 | 41 (32) |
| 10 % | soft average | 9.9 | 80.1 | 4 (4) |
| 10 % | LightGBM | 10.8 | 78.5 | 17 (15) |
| 20 % | soft average | 22.6 | 63.1 | 1 (1) |
| 20 % | LightGBM | 24.7 | 60.7 | 6 (6) |

The full table, with flat and hierarchical rows, is in the paper and the JSON.

- **Leaks are barely separated from noise.** For in/out against noise/other
  among non-normal recordings, every model scores AUC 0.594–0.598 pooled. That
  is 0.57–0.66 in five regions, and below 0.5 in regions 44 (0.36–0.41) and 48
  (0.47–0.48). In against out among true leaks scores 0.54 pooled, and below
  0.5 in regions 27, 29 and 45.
- **Calibration transfers only in the pool.** Targets of 1/5/10/20 % give
  0.8 / 4.2–4.6 / 9.6–10.8 / 22.5–24.7 % missed, pooled over the held-out
  regions. Regions 46 and 47 hold 81 % of in/out recordings, and region by
  region the 10 % target misses 2.4–50.2 % (21–23 % on region 43). At 10 %
  miss, 78.5–81.0 % of noise/other recordings also send a crew.
- **Per location.** With scores averaged per sensor, leak vs noise AUC is
  0.589–0.592 and in vs out 0.579–0.583. At the 10 % target, 938–944 of 969
  leak sensors and 383–386 of 418 noise/other sensors get at least one crew.
- **Against acting on every alarm.** The 10 % target avoids 19–21.5 % of
  nuisance visits, against 10 % for a class-unrelated score. Dispatch
  precision is 72.4–72.8 %, against a base rate of 70.2 %, which is set by
  curation.
- **Multiplicity.** Of the 18 normal false-dispatch McNemar tests (3 pairs ×
  6 operating points), only the 5 % comparisons survive Holm.
- **No consistent model ranking.**
  - At argmax the soft average is cheaper than LightGBM if a missed leak
    costs more than 0.55 nuisance dispatches; LightGBM is cheaper otherwise.
  - At 5 % LightGBM has fewer misses and fewer dispatches.
  - At 10 % the soft average misses fewer leaks and dispatches to more noise.
- **Normal false dispatches.** The hierarchical-gate models are lower at the
  5 % and 10 % targets (10 %: 4 against 15 sensors, p = 0.003; lower at all
  7 installations, sign test p = 0.016). They are level at 1 % (soft average
  72 against 76, p = 0.69), and the order reverses at a test-fitted zero-miss
  threshold (164 against 142 recordings). The device trigger flags 0 of the
  22,121 normals, so gating on it removes every normal false dispatch for
  every model.

## Conclusion

The ensemble recommendation is withdrawn. On this benchmark the dispatch
decision is limited by telling leaks from noise, where every candidate is near
AUC 0.6. Model choice moves the operating point by a few points, and the
direction depends on the miss target.
