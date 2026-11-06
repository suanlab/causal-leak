"""Paired tests whose unit is the sensor (or installation), not the recording.

Recordings from one sensor are not independent, so a paired comparison of two
models' errors is made over sensors:

  mcnemar_exact      exact McNemar test on discordant units: units where only
                     model A errs against units where only model B errs
  cluster_bootstrap  percentile interval for the difference in error counts,
                     resampling units with replacement
  sign_test          exact two-sided sign test over coarser units (e.g.
                     installations); ties are dropped

A "unit with an error" is whatever the caller defines (a normal sensor with at
least one false dispatch, a leak sensor never dispatched, ...). Units are
hashable keys; (region, sid) tuples keep sensors from different folds distinct.

Run as a script to recompute the sensor-level comparison reported from
results/operational_folds_results.json.
"""
import json
import os
from collections import Counter

import numpy as np
from scipy.stats import binomtest


def mcnemar_exact(err_a, err_b):
    """Exact McNemar test from the sets of units on which each model errs."""
    err_a, err_b = set(err_a), set(err_b)
    only_a, only_b = len(err_a - err_b), len(err_b - err_a)
    n = only_a + only_b
    p = binomtest(only_a, n, 0.5).pvalue if n else 1.0
    return {'only_a': only_a, 'only_b': only_b, 'discordant': n,
            'p_value': float(p)}


def cluster_bootstrap(count_a, count_b, units, n_boot=10000, seed=42,
                      level=0.95):
    """Interval for sum(count_b - count_a), resampling units with replacement.

    count_a, count_b: mappings unit -> error count (missing units count 0).
    units: every unit at risk, including those neither model erred on.
    """
    diff = np.array([count_b.get(u, 0) - count_a.get(u, 0) for u in units])
    rng = np.random.default_rng(seed)
    boots = np.array([diff[rng.integers(0, len(diff), len(diff))].sum()
                      for _ in range(n_boot)])
    a = (1 - level) / 2 * 100
    lo, hi = np.percentile(boots, [a, 100 - a])
    return {'observed': int(diff.sum()), 'lo': float(lo), 'hi': float(hi),
            'n_units': len(diff), 'n_boot': n_boot, 'seed': seed}


def sign_test(pairs):
    """Exact two-sided sign test over (a, b) pairs; ties are dropped."""
    a_worse = sum(1 for a, b in pairs if a > b)
    b_worse = sum(1 for a, b in pairs if b > a)
    n = a_worse + b_worse
    p = binomtest(a_worse, n, 0.5).pvalue if n else 1.0
    return {'a_worse': a_worse, 'b_worse': b_worse,
            'ties': len(pairs) - n, 'p_value': float(p)}


def main():
    path = os.path.join(os.path.dirname(__file__), '..', 'results',
                        'operational_folds_results.json')
    op = json.load(open(path))
    est = [f for f in op['folds'] if f['far_estimable']]
    units = sorted({(f['region'], s) for f in est for s in f['normal_sensors']})
    a, b = 'tree_only_soft_avg', 'lightgbm'
    cnt = {m: Counter((f['region'], s) for f in est
                      for s in f['false_alarm_sensors'][m]) for m in (a, b)}
    print(f'{len(units)} normal sensors over regions '
          f'{[f["region"] for f in est]}; error = any non-normal prediction')
    print('McNemar  ', mcnemar_exact(cnt[a], cnt[b]))
    print('bootstrap', cluster_bootstrap(cnt[a], cnt[b], units))


if __name__ == '__main__':
    main()
