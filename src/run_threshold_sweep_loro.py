"""A2 / EAAI revision — calibrated detection-rate sweep across all 11 LORO folds.

The original E5/E7 evaluation on the single Chungbuk fold is uninformative:
Chungbuk has only 45 normal vs 5{,}029 abnormal samples, so the Stage-1
binary gate trivially achieves 100\,\% detection at near-zero false-alarm
rate. This stronger redesign evaluates the binary normal-vs-abnormal gate
across all 11 LORO folds (region 36 is skipped, single-class).

Protocol (deployment-honest):
  - For each held-out region r:
      tr_pool  = rows from other 11 regions
      cal      = random 20\% of tr_pool (threshold-calibration split)
      tr_main  = remaining 80\% of tr_pool
      te       = rows from region r
  - Train Stage-1 binary XGBoost on tr_main, with `normal` as negative
    class and \{in, out, noise, other\} as positive (leak) class.
  - For each cal-FAR target $\in$ \{0.005, 0.010, 0.020, 0.050\}:
      pick T_cal as the (1 - far_target) quantile of leak-prob on cal
      negatives, evaluate (detection_rate, achieved_far) on te.
  - Aggregate via weighted-by-$n_\mathrm{test}$ mean across folds.

Writes results/threshold_sweep_loro_results.json. Compute: ~5 min on CPU.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42
NORMAL_LABEL = 4
CAL_FRAC = 0.20
FAR_TARGETS = [0.005, 0.010, 0.020, 0.050]


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def train_stage1(X, y, tr):
    """Binary normal-vs-abnormal XGBoost on causal features."""
    is_normal = (y[tr] == NORMAL_LABEL).astype(np.int64)
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', n_jobs=-1, random_state=SEED,
        eval_metric='logloss', verbosity=0,
    )
    clf.fit(X[tr], is_normal)
    return clf


def threshold_at_far_on_cal(p_leak_cal, y_cal_bin, far):
    """Quantile-based threshold: smallest T such that
    Pr_cal(p_leak >= T | normal) <= far."""
    neg = p_leak_cal[y_cal_bin == 0]
    if len(neg) == 0:
        return None
    return float(np.quantile(neg, 1.0 - far))


def evaluate_op(p_leak_te, y_te_bin, T):
    if T is None:
        return None
    neg = p_leak_te[y_te_bin == 0]
    pos = p_leak_te[y_te_bin == 1]
    return {
        'threshold': T,
        'detection_rate':
            float((pos >= T).mean()) if len(pos) > 0 else None,
        'achieved_far':
            float((neg >= T).mean()) if len(neg) > 0 else None,
        'n_pos': int(len(pos)),
        'n_neg': int(len(neg)),
    }


def main():
    t0 = time.time()
    df = load_all()
    cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    X_all = df[cols].to_numpy(dtype=np.float32, copy=False)
    y_all = df['label'].to_numpy(dtype=np.int64)

    regions = sorted(df['region_lvl1'].unique().tolist())
    print(f'Regions: {regions}', flush=True)

    folds = []
    for r in regions:
        if r == 36:
            continue  # single class
        te_mask = (df['region_lvl1'].values == r)
        tr_pool = np.where(~te_mask)[0]
        te = np.where(te_mask)[0]
        # cal/tr split
        rng = np.random.RandomState(SEED + int(r))
        perm = rng.permutation(len(tr_pool))
        n_cal = int(CAL_FRAC * len(tr_pool))
        cal = tr_pool[perm[:n_cal]]
        tr_main = tr_pool[perm[n_cal:]]

        # standardise on tr_main only
        sc = StandardScaler().fit(X_all[tr_main])
        X = sc.transform(X_all).astype(np.float32)

        y_te_bin = (y_all[te] != NORMAL_LABEL).astype(np.int64)
        y_cal_bin = (y_all[cal] != NORMAL_LABEL).astype(np.int64)
        n_normal_te = int((y_all[te] == NORMAL_LABEL).sum())
        n_normal_cal = int((y_all[cal] == NORMAL_LABEL).sum())

        if n_normal_te == 0 or n_normal_cal == 0:
            print(f'  skip region {r}: no normal samples '
                  f'(te={n_normal_te}, cal={n_normal_cal})',
                  flush=True)
            continue

        clf = train_stage1(X, y_all, tr_main)
        p_leak_cal = 1.0 - clf.predict_proba(X[cal])[:, 1]
        p_leak_te = 1.0 - clf.predict_proba(X[te])[:, 1]

        ops = {}
        for far in FAR_TARGETS:
            T = threshold_at_far_on_cal(p_leak_cal, y_cal_bin, far)
            ops[f'far_{far:.3f}'] = evaluate_op(p_leak_te, y_te_bin, T)

        rec = {
            'region': int(r),
            'n_test': int(len(te)),
            'n_normal_test': n_normal_te,
            'n_abnormal_test': int((y_all[te] != NORMAL_LABEL).sum()),
            'n_normal_cal': n_normal_cal,
            'operating_points': ops,
        }
        folds.append(rec)
        print(f'  region {r:>3d}: n_te={len(te):>5d}  '
              f'normal_te={n_normal_te:>4d}  '
              f'cal-T@0.01={ops["far_0.010"]["threshold"]:.4f}  '
              f'det={ops["far_0.010"]["detection_rate"]:.3f}  '
              f'achieved-FAR={ops["far_0.010"]["achieved_far"]:.4f}',
              flush=True)

    # Aggregate: weighted-by-n_test mean of detection_rate and achieved_far
    # for each cal-FAR target
    aggregates = {}
    for far in FAR_TARGETS:
        key = f'far_{far:.3f}'
        ops_list = [(f['n_test'], f['operating_points'][key]) for f in folds
                    if f['operating_points'].get(key) is not None]
        if not ops_list:
            continue
        total_n = sum(n for n, _ in ops_list)
        wm_det = sum(n * op['detection_rate'] for n, op in ops_list) / total_n
        wm_far = sum(n * op['achieved_far'] for n, op in ops_list) / total_n
        # Macro (unweighted) mean
        m_det = np.mean([op['detection_rate'] for _, op in ops_list])
        m_far = np.mean([op['achieved_far'] for _, op in ops_list])
        aggregates[key] = {
            'far_target': far,
            'n_folds': len(ops_list),
            'weighted_mean_detection': float(wm_det),
            'weighted_mean_achieved_far': float(wm_far),
            'macro_mean_detection': float(m_det),
            'macro_mean_achieved_far': float(m_far),
            'min_detection': float(min(op['detection_rate']
                                        for _, op in ops_list)),
            'max_detection': float(max(op['detection_rate']
                                        for _, op in ops_list)),
        }

    print('\n=== LORO calibrated-FAR aggregate ===')
    print(f'{"target":>10s}  {"n_folds":>8s}  {"weighted det":>14s}  '
          f'{"weighted FAR":>14s}  {"macro det":>10s}  {"min det":>8s}')
    for k, a in aggregates.items():
        print(f'{k:>10s}  {a["n_folds"]:>8d}  '
              f'{a["weighted_mean_detection"]:>14.4f}  '
              f'{a["weighted_mean_achieved_far"]:>14.4f}  '
              f'{a["macro_mean_detection"]:>10.4f}  '
              f'{a["min_detection"]:>8.4f}')

    out_path = os.path.join(RESULTS_DIR, 'threshold_sweep_loro_results.json')
    with open(out_path, 'w') as f:
        json.dump({'folds': folds, 'aggregates': aggregates,
                   'seed': SEED, 'cal_frac': CAL_FRAC,
                   'far_targets': FAR_TARGETS,
                   'time_s': time.time() - t0}, f, indent=2)
    print(f'\nSaved {out_path}  ({time.time() - t0:.0f}s)')


if __name__ == '__main__':
    main()
