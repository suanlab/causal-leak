"""Phase 1 / E5+E7 — Cost-aware metrics with calibration-only threshold
selection.

E5: detection-rate-at-target-FAR
    Treat 'normal' as the negative class and any of {in, out, noise,
    other} as the positive class. The decision threshold is selected
    on a *training-side* calibration split that the test fold has
    never been exposed to, then evaluated once on the held-out
    Chungbuk LORO test fold. This avoids test-set tuning and gives
    an honest deployment-side operating point.

E7: hierarchical Stage-1 threshold sweep
    For the binary 'normal vs abnormal' gate of the hierarchical
    classifier, sweep T in {0.3, 0.4, 0.5, 0.6, 0.7} on the test
    fold and report (missed-leak rate, false-alarm rate, downstream
    4-class macro-F1, full-set 5-class macro-F1). The sweep is
    diagnostic, not operating-point selection.

The two flat / hierarchical XGBoost classifiers are re-trained inside
this script (no cached predictions are read), so the script is
self-contained and reproducible from `make_split` alone.

Writes results/cost_metrics_results.json.
Compute: ~30 seconds on CPU.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             f1_score)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, LABEL_NAMES, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = 42
NORMAL_LABEL = 4   # data.LABEL_MAP['normal'] == 4
CAL_FRAC = 0.20    # fraction of training rows reserved for threshold calibration


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def make_split():
    """Three-way split: tr_main (train) / cal (threshold calibration) / te.

    tr_main and cal both come from the non-Chungbuk regions, so the
    held-out fold is never touched during calibration.
    """
    df = load_all()
    cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)
    te_mask = df['region_lvl1'].values == HOLDOUT_REGION
    tr_pool = np.where(~te_mask)[0]
    te = np.where(te_mask)[0]

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_pool))
    n_cal = int(CAL_FRAC * len(tr_pool))
    cal = tr_pool[perm[:n_cal]]
    tr_main = tr_pool[perm[n_cal:]]
    return X, y, tr_main, cal, te, cols


def train_xgb_flat(X, y, tr):
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=5,
        tree_method='hist', n_jobs=-1, random_state=SEED,
        eval_metric='mlogloss', verbosity=0,
    )
    clf.fit(X[tr], y[tr])
    return clf


def train_hierarchical(X, y, tr):
    """Stage 1: binary 'normal vs abnormal'. Stage 2: 4-class abnormal."""
    is_normal = (y[tr] == NORMAL_LABEL).astype(np.int64)
    s1 = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', n_jobs=-1, random_state=SEED,
        eval_metric='logloss', verbosity=0,
    )
    s1.fit(X[tr], is_normal)

    abn_mask = y[tr] != NORMAL_LABEL
    s2 = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=4,
        tree_method='hist', n_jobs=-1, random_state=SEED,
        eval_metric='mlogloss', verbosity=0,
    )
    s2.fit(X[tr][abn_mask], y[tr][abn_mask])
    return s1, s2


def threshold_at_far_on_cal(prob_leak_cal, y_cal_binary, far_target):
    """Pick the lowest leak-probability threshold T such that the
    calibration-side false-alarm rate <= far_target.

    Returns None if no normal samples are present in the calibration
    split (cannot calibrate FAR).
    """
    neg = prob_leak_cal[y_cal_binary == 0]
    if len(neg) == 0:
        return None
    return float(np.quantile(neg, 1.0 - far_target))


def evaluate_operating_point(prob_leak_te, y_te_binary, T):
    neg = prob_leak_te[y_te_binary == 0]
    pos = prob_leak_te[y_te_binary == 1]
    detection_rate = float((pos >= T).mean()) if len(pos) > 0 else None
    achieved_far = float((neg >= T).mean()) if len(neg) > 0 else None
    return {'threshold': T, 'detection_rate': detection_rate,
            'achieved_far': achieved_far,
            'n_pos': int(len(pos)), 'n_neg': int(len(neg))}


def main():
    t0 = time.time()
    print('Loading data...', flush=True)
    X, y, tr_main, cal, te, cols = make_split()
    sc = StandardScaler().fit(X[tr_main])
    X_s = sc.transform(X).astype(np.float32)
    print(f'  X_s={X_s.shape}  tr_main={len(tr_main)}  '
          f'cal={len(cal)}  te={len(te)}', flush=True)

    cal_normal = int((y[cal] == NORMAL_LABEL).sum())
    cal_abn = int((y[cal] != NORMAL_LABEL).sum())
    print(f'  calibration split: {cal_normal} normal / {cal_abn} abnormal',
          flush=True)

    # =============== E5: train on tr_main, calibrate on cal, eval on te ===
    print('\n--- E5: detection rate at calibration-derived FAR ---',
          flush=True)
    print('  training flat causal-XGBoost on tr_main...', flush=True)
    flat = train_xgb_flat(X_s, y, tr_main)
    p_cal_flat = 1.0 - flat.predict_proba(X_s[cal])[:, NORMAL_LABEL]
    p_te_flat = 1.0 - flat.predict_proba(X_s[te])[:, NORMAL_LABEL]
    y_cal_bin = (y[cal] != NORMAL_LABEL).astype(np.int64)
    y_te_bin = (y[te] != NORMAL_LABEL).astype(np.int64)

    e5_flat = {}
    for far in [0.01, 0.02, 0.05, 0.10]:
        T_cal = threshold_at_far_on_cal(p_cal_flat, y_cal_bin, far)
        if T_cal is None:
            e5_flat[f'far_{far:.2f}'] = {'error':
                                          'no normal samples in calibration split'}
            continue
        op = evaluate_operating_point(p_te_flat, y_te_bin, T_cal)
        op['far_target'] = far
        op['threshold_source'] = 'calibration_split'
        op['cal_n_normal'] = cal_normal
        e5_flat[f'far_{far:.2f}'] = op
        print(f'  flat XGB cal-T@FAR={far:.2f} -> T={T_cal:.4f}  '
              f'test detection={op["detection_rate"]:.4f}  '
              f'test achieved FAR={op["achieved_far"]:.4f}', flush=True)

    print('\n  training hierarchical on tr_main...', flush=True)
    s1, s2 = train_hierarchical(X_s, y, tr_main)
    p_cal_norm = s1.predict_proba(X_s[cal])[:, 1]
    p_te_norm = s1.predict_proba(X_s[te])[:, 1]
    p_cal_h = 1.0 - p_cal_norm
    p_te_h = 1.0 - p_te_norm

    e5_hier = {}
    for far in [0.01, 0.02, 0.05, 0.10]:
        T_cal = threshold_at_far_on_cal(p_cal_h, y_cal_bin, far)
        if T_cal is None:
            e5_hier[f'far_{far:.2f}'] = {'error':
                                          'no normal samples in calibration split'}
            continue
        op = evaluate_operating_point(p_te_h, y_te_bin, T_cal)
        op['far_target'] = far
        op['threshold_source'] = 'calibration_split'
        op['cal_n_normal'] = cal_normal
        e5_hier[f'far_{far:.2f}'] = op
        print(f'  hier  cal-T@FAR={far:.2f} -> T={T_cal:.4f}  '
              f'test detection={op["detection_rate"]:.4f}  '
              f'test achieved FAR={op["achieved_far"]:.4f}', flush=True)

    # =============== E7: stage-1 threshold sweep (test-only diagnostic) ===
    print('\n--- E7: hierarchical Stage-1 threshold sweep (diagnostic) ---',
          flush=True)
    p_abn_test = s2.predict_proba(X_s[te])
    abn_pred_test = p_abn_test.argmax(1)
    e7_records = []
    for T in [0.3, 0.4, 0.5, 0.6, 0.7]:
        pred_normal = p_te_norm >= T
        pred = np.where(pred_normal, NORMAL_LABEL, abn_pred_test).astype(np.int64)
        cm = confusion_matrix(y[te], pred, labels=list(range(5)))
        tp_normal = cm[NORMAL_LABEL, NORMAL_LABEL]
        fp_normal = cm[:, NORMAL_LABEL].sum() - tp_normal
        fn_normal = cm[NORMAL_LABEL, :].sum() - tp_normal
        n_abn_te = (y[te] != NORMAL_LABEL).sum()
        n_normal_te = (y[te] == NORMAL_LABEL).sum()
        miss_leak = float(fp_normal / max(1, n_abn_te))
        false_alarm = float(fn_normal / max(1, n_normal_te))
        f1_4cls = f1_score(
            y[te][y[te] != NORMAL_LABEL],
            pred[y[te] != NORMAL_LABEL],
            labels=[0, 1, 2, 3], average='macro', zero_division=0,
        )
        f1_5cls = f1_score(y[te], pred, average='macro', zero_division=0)
        rec = {
            'threshold': T,
            'missed_leak_rate': miss_leak,
            'false_alarm_rate': false_alarm,
            'f1_macro_4cls_on_abnormal': float(f1_4cls),
            'f1_macro_5cls': float(f1_5cls),
            'cm': cm.tolist(),
        }
        e7_records.append(rec)
        print(f'  T={T:.1f}  miss={miss_leak:.3f}  '
              f'FAR={false_alarm:.3f}  '
              f'4cls_F1={f1_4cls:.3f}  5cls_F1={f1_5cls:.3f}',
              flush=True)

    out = {
        'features': cols,
        'holdout_region': HOLDOUT_REGION,
        'seed': SEED,
        'cal_split_fraction': CAL_FRAC,
        'cal_n_total': int(len(cal)),
        'cal_n_normal': cal_normal,
        'cal_n_abnormal': cal_abn,
        'e5_flat_xgb_detection_at_far': e5_flat,
        'e5_hier_stage1_detection_at_far': e5_hier,
        'e7_threshold_sweep': e7_records,
        'time_s': time.time() - t0,
    }
    out_path = os.path.join(RESULTS_DIR, 'cost_metrics_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}  ({out["time_s"]:.0f}s)')


if __name__ == '__main__':
    main()
