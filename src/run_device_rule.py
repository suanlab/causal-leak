"""Can normal-vs-rest be separated by the sensor's own threshold rule?

The AI Hub page for this dataset defines `lrate` as the share of 10 daily leak
checks (two-hour early-morning window) whose response exceeded a fixed
threshold. The 20 MAX columns are, alternately, the peak frequency (even
index) and peak amplitude (odd index) of each of those 10 checks. Every
in/out/noise/other recording has lrate = 90 and every normal one 30-70, so
class membership is conditioned on how often the device's threshold fired.

This script fits no classifier. Under leave-one-region-out it

1. identifies the device threshold T from the training regions' device fields
   alone: the T for which lrate == 10 * #{checks with peak amplitude >= T}
   holds on every training row with lrate < 90 (an arithmetic identity, not a
   fit; the fold fails if the solution is not unique);
2. takes k from the same fields: k_lo is one more than the largest count on
   those rows, and k_hi the largest k that leaves the same lrate = 90 rows
   below it (no recording has a count of exactly 8, so k is identified only
   up to that gap);
3. applies  non-normal  iff  #{checks with peak amplitude >= T} >= k
   unchanged to the held-out region.

In this release lrate < 90 coincides with the normal class, so "no
classifier" means T and k come from logged device fields, not that they are
independent of the label partition.

For comparison it also reports the balanced-accuracy grid search used in the
first version (T in steps of 10, which never tested 125; archived in
results/superseded_grid10/) and the same search in steps of 1.

The rule uses only MAX columns, which are in the 534-column signal set, so any
learned normal-vs-rest gate on that set has it available.

Output: results/device_rule_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from audit_tests import depletion_test, identify_k as _identify_k, identify_threshold  # noqa: E402
from data import load_all  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
NORMAL = 4
ALARM_CODE = 90
AMP_COLS = [f'MAX{i}' for i in range(1, 20, 2)]     # odd index = amplitude
T_SEARCH = np.arange(20, 1001, 1)
GRIDS = {'grid10': np.arange(20, 1001, 10), 'grid1': T_SEARCH}
K_GRID = np.arange(1, 11)


def count_at(amp, T):
    return (amp >= T).sum(1)


def identify_T(amp, lrate):
    """Every T with lrate == 10 * count on all rows with lrate < ALARM_CODE."""
    return identify_threshold(amp, lrate, ALARM_CODE, scale=10, grid=T_SEARCH)


def identify_k(amp, lrate, T):
    return _identify_k(amp, lrate, ALARM_CODE, T)


def fit_rule_grid(amp, is_norm, T_grid):
    """Balanced-accuracy search over (T, k); ties keep the first found."""
    best = (-1.0, None, None)
    n_norm, n_rest = is_norm.sum(), (~is_norm).sum()
    for T in T_grid:
        c = count_at(amp, T)
        h_norm = np.bincount(c[is_norm], minlength=11)
        h_rest = np.bincount(c[~is_norm], minlength=11)
        for k in K_GRID:
            tpr = h_rest[k:].sum() / n_rest
            tnr = h_norm[:k].sum() / n_norm
            ba = float((tpr + tnr) / 2)
            if ba > best[0]:
                best = (ba, int(T), int(k))
    return best


def score(amp, is_norm, sid, T, k):
    flagged = count_at(amp, T) >= k
    fa = is_norm & flagged
    miss = ~is_norm & ~flagged
    return {
        'n_normal': int(is_norm.sum()), 'n_rest': int((~is_norm).sum()),
        'false_flags': int(fa.sum()), 'missed': int(miss.sum()),
        'normal_sensors': len(set(sid[is_norm].tolist())),
        'false_flag_sensors': len(set(sid[fa].tolist())),
        'acc': float((flagged != is_norm).mean()),
    }


def pool(folds, regions=None):
    keys = ('n_normal', 'n_rest', 'false_flags', 'missed', 'normal_sensors',
            'false_flag_sensors')
    sel = [f for f in folds if regions is None or f['region'] in regions]
    return {k: sum(f[k] for f in sel) for k in keys}


def main():
    df = load_all()
    y = df['label'].to_numpy()
    is_norm = y == NORMAL
    amp = df[AMP_COLS].to_numpy(dtype=np.int64)
    reg = df['region_lvl1'].to_numpy()
    sid = df['sid'].astype(str).to_numpy()
    lrate = df['lrate'].to_numpy()
    far = (27, 29, 44, 45)

    folds, grid_folds = [], {g: [] for g in GRIDS}
    for r in sorted(set(reg.tolist())):
        tr, te = reg != r, reg == r
        Ts = identify_T(amp[tr], lrate[tr])
        assert len(Ts) == 1, f'region {r}: T not unique ({Ts})'
        T = Ts[0]
        k_lo, k_hi = identify_k(amp[tr], lrate[tr], T)
        c_te = count_at(amp[te], T)
        same = all(np.array_equal(c_te >= k_lo, c_te >= k)
                   for k in range(k_lo, k_hi + 1))
        rec = {'region': int(r), 'T': T, 'k': k_lo, 'k_range': [k_lo, k_hi],
               'test_partition_same_over_k_range': bool(same),
               **score(amp[te], is_norm[te], sid[te], T, k_lo)}
        folds.append(rec)
        print(f"region {r}: T={T} k={k_lo}-{k_hi}  normal {rec['n_normal']:>6} "
              f"false {rec['false_flags']:>3}  rest {rec['n_rest']:>6} "
              f"missed {rec['missed']:>4}", flush=True)
        for g, grid in GRIDS.items():
            ba, Tg, kg = fit_rule_grid(amp[tr], is_norm[tr], grid)
            grid_folds[g].append({'region': int(r), 'T': Tg, 'k': kg,
                                  'train_balanced_acc': ba,
                                  **score(amp[te], is_norm[te], sid[te], Tg, kg)})

    # Pooled identification and the count histogram that shows the gaps.
    T_all = identify_T(amp, lrate)
    assert len(T_all) == 1
    T_all = T_all[0]
    k_lo, k_hi = identify_k(amp, lrate, T_all)
    c = count_at(amp, T_all)
    hits = c * 10
    miss = ~is_norm & (c < k_lo)
    anomalies = [{'region': int(reg[i]), 'site': str(df['site'].iloc[i]),
                  'sid': str(sid[i]), 'date': str(df['ldate'].iloc[i]),
                  'label': int(y[i]), 'count': int(c[i])}
                 for i in np.flatnonzero(miss)]

    out = {
        'rule': 'non-normal iff #(per-check peak amplitude >= T) >= k',
        'identification': ('T: unique solution of lrate == 10*count on training '
                           'rows with lrate < 90; k: gap between the largest such '
                           'count and the counts of lrate = 90 rows'),
        'amp_cols': AMP_COLS, 'folds': folds,
        'pooled_loro': pool(folds), 'far_regions_loro': pool(folds, far),
        'pooled_fit': {'T': T_all, 'k': k_lo, 'k_range': [k_lo, k_hi]},
        'lrate_recovery_at_pooled_T': {
            'normal_hits_le_lrate': float((hits[is_norm] <= lrate[is_norm]).mean()),
            'normal_hits_eq_lrate': float((hits[is_norm] == lrate[is_norm]).mean()),
            'lrate_lt90_hits_eq_lrate': float(
                (hits[lrate < ALARM_CODE] == lrate[lrate < ALARM_CODE]).mean())},
        'count_histogram_at_pooled_T': {
            'normal': np.bincount(c[is_norm], minlength=11).tolist(),
            'non_normal': np.bincount(c[~is_norm], minlength=11).tolist()},
        'anomalies': anomalies,
        'depletion_at_pooled_T': depletion_test(
            np.bincount(c, minlength=11), boundary=k_lo, floor=int(c[lrate < ALARM_CODE].min())),
        'grid_search': {g: {'folds': f, 'pooled_loro': pool(f),
                            'far_regions_loro': pool(f, far)}
                        for g, f in grid_folds.items()},
    }
    with open(os.path.join(RES, 'device_rule_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    tot, tot_far = out['pooled_loro'], out['far_regions_loro']
    print(f"\nidentified rule: LORO pooled false flags "
          f"{tot['false_flags']}/{tot['n_normal']}, missed "
          f"{tot['missed']}/{tot['n_rest']}; FAR false flags "
          f"{tot_far['false_flags']}/{tot_far['n_normal']}, missed "
          f"{tot_far['missed']}/{tot_far['n_rest']}")
    print(f"pooled T={T_all} k={k_lo}-{k_hi}; count histogram normal "
          f"{out['count_histogram_at_pooled_T']['normal']}, non-normal "
          f"{out['count_histogram_at_pooled_T']['non_normal']}")
    for g, v in out['grid_search'].items():
        Tk = sorted({(f['T'], f['k']) for f in v['folds']})
        print(f"{g}: (T,k) {Tk}  missed {v['pooled_loro']['missed']}, "
              f"false flags {v['pooled_loro']['false_flags']}")


if __name__ == '__main__':
    main()
