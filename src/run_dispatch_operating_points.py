"""Dispatch-rule comparison of the deployment candidates at calibrated
operating points.

run_operational_folds.py counts a false alarm whenever a normal recording is
predicted as any non-normal class, and compares models only at the argmax
decision. Both choices flatter whichever model happens to place its argmax
boundary conservatively under shift. A utility dispatches a crew for an in/out
leak, not for electrical noise, so this script scores every candidate under an
explicit dispatch rule:

  dispatch        prediction is 'in' or 'out'
  missed leak     true in/out recording that is not dispatched
  false dispatch  true normal recording that is dispatched
                  (true noise/other recordings that are dispatched are counted
                  separately)

Every model is scored on the same dispatch score s = p(in) + p(out), and is
compared at three kinds of operating point:

  argmax          the decision as published (dispatch iff argmax is in/out)
  calibrated      threshold on s chosen on the TRAINING pool only, from
                  region-grouped inner cross-validation, to hit a target miss
                  rate; then applied unchanged to the held-out region
  (curve)         the full miss / false-dispatch curve on the held-out region,
                  kept for diagnosis only -- a threshold read off it is fitted
                  on the test region and is not a protocol-valid comparison

Model configurations are imported from run_operational_folds, so the outer-fold
argmax predictions reproduce that script's confusion matrices exactly. Inner
calibration folds reuse the outer fold's band set, which was selected with the
held-out region excluded; the inner held-out regions did enter that selection,
so calibration is mildly optimistic, never test-informed.

One region per invocation (regions run in parallel):

  python src/run_dispatch_operating_points.py 45

Output: results/dispatch_op/region<R>.npz (per-recording probabilities) and
results/dispatch_op/region<R>.json (fold metadata). analyze_dispatch.py turns
these into results/dispatch_operating_points_results.json.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.model_selection import GroupKFold

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import nested_bands_cached  # noqa: E402
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402
from run_operational_folds import (  # noqa: E402
    NORMAL, SEED, fit_flat, fit_hier, fit_lightgbm)

OUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'dispatch_op')
os.makedirs(OUT_DIR, exist_ok=True)

N_INNER = 3
MODELS = ['xgb_flat', 'xgb_hier', 'tree_only_soft_avg', 'lightgbm']


def inner_splits(tr, y, regions):
    """Region-grouped inner folds over the training pool.

    GroupKFold by region, unless that leaves some inner training split without
    a class the pool contains (with only four normal-bearing regions left, all
    can land in one held-out fold). Then regions are re-dealt: regions holding
    a class held by few regions go round-robin, largest first, so no fold takes
    all of them; the rest go to whichever fold has fewest rows. Only training
    labels and region sizes are used.
    """
    classes = set(y[tr].tolist())
    gkf = list(GroupKFold(n_splits=N_INNER).split(tr, y[tr], groups=regions[tr]))
    if all(set(y[tr[a]].tolist()) == classes for a, _ in gkf):
        return gkf, 'groupkfold'

    regs, cnt = np.unique(regions[tr], return_counts=True)
    size = dict(zip(regs.tolist(), cnt.tolist()))
    has = {r: set(y[tr][regions[tr] == r].tolist()) for r in size}
    holders = {c: [r for r in size if c in has[r]] for c in classes}
    rare = min(classes, key=lambda c: len(holders[c]))
    fold_of, rows = {}, [0] * N_INNER
    for i, r in enumerate(sorted(holders[rare], key=lambda r: -size[r])):
        fold_of[r] = i % N_INNER
        rows[i % N_INNER] += size[r]
    for r in sorted((r for r in size if r not in fold_of), key=lambda r: -size[r]):
        f = int(np.argmin(rows))
        fold_of[r] = f
        rows[f] += size[r]
    fid = np.array([fold_of[r] for r in regions[tr].tolist()])
    splits = [(np.where(fid != f)[0], np.where(fid == f)[0])
              for f in range(N_INNER)]
    for a, _ in splits:
        missing = classes - set(y[tr[a]].tolist())
        if missing:
            raise RuntimeError(f'inner training split lacks classes {missing}')
    return splits, 'rebalanced'


def fit_all(X, y, tr, te):
    p_flat = fit_flat(X, y, tr, te)
    p_hier = fit_hier(X, y, tr, te)
    return {
        'xgb_flat': p_flat,
        'xgb_hier': p_hier,
        'tree_only_soft_avg': 0.5 * p_flat + 0.5 * p_hier,
        'lightgbm': fit_lightgbm(X, y, tr, te),
    }


def main(r):
    t_all = time.time()
    df = load_all()
    y = df['label'].to_numpy(dtype=np.int32)
    regions = df['region_lvl1'].values
    sel_cols = ['month'] + HZ_COLS + MAX_COLS
    X_sel = df[sel_cols].to_numpy(dtype=np.float32, copy=False)

    te = np.where(regions == r)[0]
    tr = np.where(regions != r)[0]
    cols, man = nested_bands_cached(df, X_sel, y, sel_cols, r, kind='region')
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    print(f'=== region {r}: n_test={len(te)} features={len(cols)} ===', flush=True)

    t0 = time.time()
    outer = fit_all(X, y, tr, te)
    print(f'  outer fit {time.time() - t0:.0f}s', flush=True)

    # Region-grouped inner CV on the training pool: out-of-fold probabilities
    # for every training recording, used only to calibrate thresholds.
    oof = {k: np.full((len(tr), 5), np.nan, dtype=np.float32) for k in MODELS}
    inner_fold = np.full(len(tr), -1, dtype=np.int8)
    splits, split_rule = inner_splits(tr, y, regions)
    print(f'  inner split rule: {split_rule}', flush=True)
    for i, (a, b) in enumerate(splits):
        t0 = time.time()
        p = fit_all(X, y, tr[a], tr[b])
        for k in MODELS:
            oof[k][b] = p[k]
        inner_fold[b] = i
        print(f'  inner {i}: held-out regions '
              f'{sorted(set(regions[tr[b]].tolist()))} '
              f'classes_in_train={sorted(set(y[tr[a]].tolist()))} '
              f'({time.time() - t0:.0f}s)', flush=True)

    sid = df['sid'].astype(str).values
    site = df['site_int'].values
    np.savez_compressed(
        os.path.join(OUT_DIR, f'region{r}.npz'),
        te_y=y[te], te_sid=sid[te], te_site=site[te],
        tr_y=y[tr], tr_region=regions[tr], tr_inner_fold=inner_fold,
        **{f'te_{k}': outer[k].astype(np.float32) for k in MODELS},
        **{f'oof_{k}': oof[k] for k in MODELS},
    )
    meta = {
        'region': int(r), 'seed': SEED, 'n_test': int(len(te)),
        'n_train': int(len(tr)), 'n_inner': N_INNER,
        'selected_groups': man['selected_groups'], 'n_features': len(cols),
        'inner_split_rule': split_rule,
        'inner_heldout_regions': [sorted(set(regions[tr[b]].tolist()))
                                  for _, b in splits],
        'models': MODELS, 'time_s': float(time.time() - t_all),
    }
    with open(os.path.join(OUT_DIR, f'region{r}.json'), 'w') as fh:
        json.dump(meta, fh, indent=2)
    print(f'Saved region {r} ({time.time() - t_all:.0f}s)', flush=True)


if __name__ == '__main__':
    main(int(sys.argv[1]))
