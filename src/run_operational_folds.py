"""Operational comparison of the deployment candidates on folds that can
actually measure a false-alarm rate.

The canonical region-43 fold carries 45 normal recordings out of 5,074, so its
false-alarm estimate rests on 45 negatives: an observed FAR of 0/45 has a 95%
upper bound near 6.5% by the rule of three. Every model tested reaches normal
recall 1.000 there, so the fold cannot separate them on the operational axis at
all.

Regions 27, 29, 44 and 45 hold 1,151-12,299 normal recordings each. This script
evaluates the deployment candidates on those folds plus 43, and reports the two
quantities a utility actually budgets against:

  false_alarm_rate  fraction of true-normal recordings predicted as any leak
  missed_leak_rate  fraction of true-leak recordings predicted as normal

Each model uses the configuration the manuscript reports for it, so the
comparison is between the candidates as published rather than between one tuned
and one untuned learner.

Bands are selected per fold with the evaluation region excluded
(band_selection.nested_bands_cached).

Output: results/operational_folds_results.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import nested_bands_cached  # noqa: E402
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = int(os.environ.get('CAUSAL_SEED', '42'))
NORMAL = 4
REGIONS = [27, 29, 43, 44, 45]
MIN_NORMALS = 100         # region 43 (45 normals) cannot estimate a FAR

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    tree_method='hist', n_jobs=-1, random_state=SEED, verbosity=0,
)


def operational(y_true, pred):
    """Macro-F1 plus the two rates a utility budgets against."""
    is_norm = y_true == NORMAL
    fa = float((pred[is_norm] != NORMAL).mean()) if is_norm.sum() else None
    miss = float((pred[~is_norm] == NORMAL).mean()) if (~is_norm).sum() else None
    return {
        'f1_macro': float(f1_score(y_true, pred, average='macro',
                                   zero_division=0)),
        'acc': float(accuracy_score(y_true, pred)),
        'normal_recall': float((pred[is_norm] == NORMAL).mean())
                         if is_norm.sum() else None,
        'false_alarm_rate': fa,
        'missed_leak_rate': miss,
        'n_normal': int(is_norm.sum()),
        'n_leak': int((~is_norm).sum()),
        'cm': confusion_matrix(y_true, pred, labels=list(range(5))).tolist(),
    }


def fit_flat(X, y, tr, te):
    clf = XGBClassifier(objective='multi:softprob', num_class=5,
                        eval_metric='mlogloss', **XGB_PARAMS)
    clf.fit(X[tr], y[tr])
    return clf.predict_proba(X[te])


def fit_hier(X, y, tr, te):
    """Stage-1 normal gate + stage-2 four-class typer, as in run_ensemble."""
    y_bin = (y == NORMAL).astype(np.int32)
    s1 = XGBClassifier(objective='binary:logistic', eval_metric='logloss',
                       **XGB_PARAMS)
    s1.fit(X[tr], y_bin[tr])
    p_norm = s1.predict_proba(X[te])[:, 1]

    abn = tr[y[tr] != NORMAL]
    s2 = XGBClassifier(objective='multi:softprob', num_class=4,
                       eval_metric='mlogloss', **XGB_PARAMS)
    s2.fit(X[abn], y[abn])
    p_ab = s2.predict_proba(X[te])

    p = np.zeros((len(te), 5), dtype=np.float32)
    p[:, NORMAL] = p_norm
    for k in range(4):
        p[:, k] = (1 - p_norm) * p_ab[:, k]
    return p


def fit_lightgbm(X, y, tr, te):
    """LightGBM exactly as configured in run_extra_baselines.py.

    That configuration -- num_leaves=63 with inverse-frequency sample weights --
    is what produced the 0.4577 macro-F1 that outranks the ensemble. Comparing
    against a default-parameter LightGBM would understate the competitor and
    flatter the incumbent.
    """
    from lightgbm import LGBMClassifier
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    clf = LGBMClassifier(n_estimators=200, learning_rate=0.1,
                         num_leaves=63, max_depth=6,
                         objective='multiclass', num_class=5,
                         random_state=SEED, n_jobs=4, verbose=-1)
    clf.fit(X[tr], y[tr], sample_weight=cls_w[y[tr]])
    return clf.predict_proba(X[te])


def main():
    t_all = time.time()
    df = load_all()
    y = df['label'].to_numpy(dtype=np.int32)
    regions = df['region_lvl1'].values
    sel_cols = ['month'] + HZ_COLS + MAX_COLS
    X_sel = df[sel_cols].to_numpy(dtype=np.float32, copy=False)

    folds = []
    for r in REGIONS:
        te = np.where(regions == r)[0]
        tr = np.where(regions != r)[0]
        n_norm = int((y[te] == NORMAL).sum())
        print(f'=== region {r}: n_test={len(te)} n_normal={n_norm} ===',
              flush=True)

        cols, man = nested_bands_cached(df, X_sel, y, sel_cols, r,
                                        kind='region')
        X = df[cols].to_numpy(dtype=np.float32, copy=False)

        t0 = time.time()
        p_flat = fit_flat(X, y, tr, te)
        p_hier = fit_hier(X, y, tr, te)
        p_ens = 0.5 * p_flat + 0.5 * p_hier      # pre-specified tree-only avg
        p_lgbm = fit_lightgbm(X, y, tr, te)

        models = {
            'xgb_flat': p_flat.argmax(1),
            'xgb_hier': p_hier.argmax(1),
            'tree_only_soft_avg': p_ens.argmax(1),
            'lightgbm': p_lgbm.argmax(1),
        }
        # Recordings from one sensor are not independent, so a false-alarm
        # count over recordings overstates the evidence. Keep the sensor of
        # every false alarm, and every normal sensor, for a clustered test.
        sid_te = df['sid'].astype(str).values[te]
        is_norm_te = y[te] == NORMAL
        normal_sensors = sorted(set(sid_te[is_norm_te].tolist()))
        rec = {
            'region': int(r),
            'n_test': int(len(te)),
            'n_normal': n_norm,
            'far_estimable': n_norm >= MIN_NORMALS,
            'selected_groups': man['selected_groups'],
            'n_features': len(cols),
            'models': {k: operational(y[te], v) for k, v in models.items()},
            'normal_sensors': normal_sensors,
            'false_alarm_sensors': {
                k: sid_te[is_norm_te & (v != NORMAL)].tolist()
                for k, v in models.items()},
            'time_s': float(time.time() - t0),
        }
        folds.append(rec)

        print(f"  {'model':<22}{'macroF1':>9}{'FAR':>9}{'miss':>9}", flush=True)
        for k, m in rec['models'].items():
            fa = f"{m['false_alarm_rate']:.4f}" if m['false_alarm_rate'] is not None else '--'
            ms = f"{m['missed_leak_rate']:.4f}" if m['missed_leak_rate'] is not None else '--'
            print(f"  {k:<22}{m['f1_macro']:>9.4f}{fa:>9}{ms:>9}", flush=True)
        print(f'  ({rec["time_s"]:.0f}s)\n', flush=True)

    # aggregate over folds where a FAR estimate is meaningful
    est = [f for f in folds if f['far_estimable']]
    summary = {}
    for k in ['xgb_flat', 'xgb_hier', 'tree_only_soft_avg', 'lightgbm']:
        w = np.array([f['n_normal'] for f in est], dtype=float)
        far = np.array([f['models'][k]['false_alarm_rate'] for f in est])
        f1 = np.array([f['models'][k]['f1_macro'] for f in est])
        miss = np.array([f['models'][k]['missed_leak_rate'] for f in est])
        summary[k] = {
            'far_weighted': float(np.average(far, weights=w)),
            'far_mean': float(far.mean()),
            'far_max': float(far.max()),
            'missed_leak_weighted': float(np.average(
                miss, weights=[f['n_test'] - f['n_normal'] for f in est])),
            'f1_macro_mean': float(f1.mean()),
            'n_folds': len(est),
        }

    out = {'seed': SEED, 'regions': REGIONS, 'min_normals': MIN_NORMALS,
           'folds': folds, 'summary_over_far_estimable_folds': summary}
    path = os.path.join(RESULTS_DIR, 'operational_folds_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)

    print('=== SUMMARY over folds with >= %d normals (n=%d) ==='
          % (MIN_NORMALS, len(est)), flush=True)
    print(f"  {'model':<22}{'FAR(w)':>10}{'FAR max':>10}{'miss(w)':>10}{'macroF1':>10}",
          flush=True)
    for k, s in summary.items():
        print(f"  {k:<22}{s['far_weighted']:>10.4f}{s['far_max']:>10.4f}"
              f"{s['missed_leak_weighted']:>10.4f}{s['f1_macro_mean']:>10.4f}",
              flush=True)
    print(f'\nSaved {path}\nTotal {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
