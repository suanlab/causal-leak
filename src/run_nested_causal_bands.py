"""Leakage-free replacement for the causal-band headline experiment.

``run_causal_bands.py`` evaluates a 65-422 Hz subset that was chosen from
permutation importance measured on region 43 -- the same region it then
reports as the held-out test fold. This script re-runs the comparison with
band selection nested strictly inside the training regions (see
``band_selection.py``), and repeats it for every region with enough label
coverage instead of the single canonical fold.

Three feature sets are compared on each evaluation region:

  signal_all     all 513 HZ bands + 20 MAX + month (no selection at all)
  nested_causal  bands chosen by inner-fold permutation, eval region unseen
  fixed_65_422   the previously published band, carried forward for reference
                 and explicitly flagged as test-derived

The gap between ``nested_causal`` and ``fixed_65_422`` is the size of the
optimism the old protocol introduced.

Output: results/nested_causal_bands_results.json
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import (  # noqa: E402
    XGB_PARAMS, bands_in_range, select_bands_nested, usable_regions,
    verify_manifest,
)
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

SEED = int(os.environ.get('CAUSAL_SEED', '42'))


def fit_eval(df, feature_cols, tr, te, y):
    X = df[feature_cols].to_numpy(dtype=np.float32, copy=False)
    clf = XGBClassifier(**{**XGB_PARAMS, 'random_state': SEED})
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return {
        'n_features': len(feature_cols),
        'acc': float(accuracy_score(y[te], pred)),
        'f1_macro': float(f1_score(y[te], pred, average='macro',
                                   zero_division=0)),
        'f1_per_class': [
            float(f1_score(y[te], pred, labels=[i], average='macro',
                           zero_division=0)) for i in range(5)
        ],
        'cm': confusion_matrix(y[te], pred, labels=list(range(5))).tolist(),
    }


def main():
    t_all = time.time()
    df = load_all()
    y = df['label'].to_numpy(dtype=np.int32)
    regions = df['region_lvl1'].values

    # Full signal matrix is what band selection searches over.
    sel_cols = ['month'] + HZ_COLS + MAX_COLS
    X_sel = df[sel_cols].to_numpy(dtype=np.float32, copy=False)

    outer_regions = usable_regions(df, y)
    print(f'Evaluation regions (n={len(outer_regions)}): {outer_regions}',
          flush=True)
    print(f'Seed: {SEED}\n', flush=True)

    fixed_cols = ['month'] + bands_in_range(65, 422) + MAX_COLS

    folds = []
    for region in outer_regions:
        t0 = time.time()
        te = np.where(regions == region)[0]
        tr = np.where(regions != region)[0]
        print(f'=== eval region {region}  train={len(tr)}  test={len(te)} ===',
              flush=True)

        selected, nested_cols, manifest = select_bands_nested(
            df, X_sel, y, sel_cols, region, seed=SEED)
        verify_manifest(manifest)
        print(f'  selected groups: {selected}  '
              f'({len(nested_cols)} features)', flush=True)

        record = {
            'region': int(region),
            'n_train': int(len(tr)),
            'n_test': int(len(te)),
            'n_classes_in_test': int(len(np.unique(y[te]))),
            'selected_groups': selected,
            'manifest': manifest,
            'models': {},
        }
        for name, cols in [
            ('signal_all', sel_cols),
            ('nested_causal', nested_cols),
            ('fixed_65_422', fixed_cols),
        ]:
            m = fit_eval(df, cols, tr, te, y)
            record['models'][name] = m
            print(f'    {name:>14s}: n_feat={m["n_features"]:4d}  '
                  f'acc={m["acc"]:.4f}  f1={m["f1_macro"]:.4f}', flush=True)

        record['time_s'] = float(time.time() - t0)
        folds.append(record)
        print(f'  fold done in {record["time_s"]:.0f}s\n', flush=True)

    # ---- aggregate ----
    summary = {}
    for name in ['signal_all', 'nested_causal', 'fixed_65_422']:
        accs = np.array([f['models'][name]['acc'] for f in folds])
        f1s = np.array([f['models'][name]['f1_macro'] for f in folds])
        w = np.array([f['n_test'] for f in folds], dtype=float)
        summary[name] = {
            'macro_f1_mean': float(f1s.mean()),
            'macro_f1_std': float(f1s.std()),
            'acc_mean': float(accs.mean()),
            'weighted_acc': float(np.average(accs, weights=w)),
            'weighted_f1': float(np.average(f1s, weights=w)),
        }

    # How often does each band group survive honest selection?
    stability = {}
    for f in folds:
        for g in f['selected_groups']:
            stability[g] = stability.get(g, 0) + 1
    stability = {g: {'selected_in_folds': c, 'of_total': len(folds)}
                 for g, c in sorted(stability.items())}

    optimism = (summary['fixed_65_422']['macro_f1_mean']
                - summary['nested_causal']['macro_f1_mean'])

    out = {
        'protocol': 'nested band selection; eval region excluded from '
                    'all selection inputs (verified per fold)',
        'seed': SEED,
        'eval_regions': outer_regions,
        'folds': folds,
        'summary': summary,
        'band_group_stability': stability,
        'selection_optimism_macro_f1': float(optimism),
    }
    path = os.path.join(RESULTS_DIR, 'nested_causal_bands_results.json')
    with open(path, 'w') as fh:
        json.dump(out, fh, indent=2)

    print('=== SUMMARY (mean over evaluation regions) ===', flush=True)
    print(f'{"model":>16s}{"macro_F1":>14s}{"acc":>10s}{"weighted_F1":>14s}',
          flush=True)
    for name, s in summary.items():
        print(f'{name:>16s}{s["macro_f1_mean"]:>10.4f}'
              f'±{s["macro_f1_std"]:.3f}{s["acc_mean"]:>10.4f}'
              f'{s["weighted_f1"]:>14.4f}', flush=True)
    print(f'\nBand-group selection stability: {stability}', flush=True)
    print(f'Selection optimism (fixed - nested) macro-F1: {optimism:+.4f}',
          flush=True)
    print(f'\nSaved {path}', flush=True)
    print(f'Total elapsed: {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
