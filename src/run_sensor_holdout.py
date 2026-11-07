"""Where does typing collapse: at a new sensor, or at a new site?

Adds a "new sensor at a known site" rung to the evaluation ladder of
run_matrix.py: the same XGBoost configuration under 5-fold
StratifiedGroupKFold grouped by sensor identifier (sid), so every test sensor
is unseen but its site usually is not. Run on both feature sets.

Then, inside the three regions with the most alarm recordings (43, 46, 47),
compares grouping by sensor with grouping by site on the signal-only set, so
the region is held fixed and only the unit of novelty changes.

Alarm-class macro-F1 (mean F1 over in/out/noise/other present) is computed
from the stored confusion matrices by summarize_leak_f1.leak_f1.

Output: results/sensor_holdout_results.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from data import get_xy, load_all  # noqa: E402
from run_matrix import N_SPLITS, SEED, fit_eval  # noqa: E402
from summarize_leak_f1 import leak_f1, present_from_cm  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
REGIONS = (43, 46, 47)


def splits_by(X, y, groups):
    n = min(N_SPLITS, len(np.unique(groups)))
    try:
        cv = StratifiedGroupKFold(n_splits=n, shuffle=True, random_state=SEED)
        return list(cv.split(X, y, groups=groups)), n
    except ValueError:
        return list(GroupKFold(n_splits=n).split(X, y, groups=groups)), n


def run(X, y, groups, label):
    folds = []
    splits, n = splits_by(X, y, groups)
    for i, (tr, te) in enumerate(splits):
        m, _ = fit_eval(X, y, tr, te)
        m['alarm_f1'] = leak_f1(m['f1_per_class'], present_from_cm(m['cm']))
        m['n_test_groups'] = int(len(np.unique(groups[te])))
        folds.append(m)
        print(f'  {label} fold {i}: alarm-F1 {m["alarm_f1"]:.3f} '
              f'acc {m["acc"]:.3f} test groups {m["n_test_groups"]}', flush=True)
    vals = [f['alarm_f1'] for f in folds if f['alarm_f1'] is not None]
    chance = [1 / len(present_from_cm(f['cm'])) for f in folds
              if present_from_cm(f['cm'])]
    return {'n_splits': n, 'folds': folds,
            'alarm_f1_mean': float(np.mean(vals)),
            'chance_mean': float(np.mean(chance)),
            'acc_mean': float(np.mean([f['acc'] for f in folds]))}


def main():
    t0 = time.time()
    df = load_all()
    out = {'ladder_group_sensor': {}, 'within_region': {}}
    for feat in ('full', 'signal'):
        X, y, _ = get_xy(df, feat)
        X, y = np.asarray(X), np.asarray(y)
        print(f'=== {feat} / group_sensor  X={X.shape} ===', flush=True)
        out['ladder_group_sensor'][feat] = run(
            X, y, df['sid'].astype(str).to_numpy(), f'{feat}/sensor')
    X, y, _ = get_xy(df, 'signal')
    X, y = np.asarray(X), np.asarray(y)
    reg = df['region_lvl1'].to_numpy()
    for r in REGIONS:
        m = reg == r
        rec = {'n_rows': int(m.sum()), 'n_sites': int(df.loc[m, 'site'].nunique()),
               'n_sensors': int(df.loc[m, 'sid'].nunique())}
        for unit, col in (('sensor', 'sid'), ('site', 'site')):
            print(f'=== region {r} / group by {unit} ===', flush=True)
            rec[unit] = run(X[m], y[m], df.loc[m, col].astype(str).to_numpy(),
                            f'r{r}/{unit}')
        out['within_region'][str(r)] = rec
    out['time_s'] = time.time() - t0
    with open(os.path.join(RES, 'sensor_holdout_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    for feat, v in out['ladder_group_sensor'].items():
        print(f'{feat} group_sensor: alarm-F1 {v["alarm_f1_mean"]:.3f} '
              f'(chance {v["chance_mean"]:.3f})')
    for r, v in out['within_region'].items():
        print(f'region {r}: sensor {v["sensor"]["alarm_f1_mean"]:.3f}  '
              f'site {v["site"]["alarm_f1_mean"]:.3f}  ({v["n_sites"]} sites)')


if __name__ == '__main__':
    main()
