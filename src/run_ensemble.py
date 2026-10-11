"""Phase D (Strengthening 2 final) — Ensemble of model families.

Combines soft predictions (class probabilities) from the strongest
tree and deep configurations:

  1. XGBoost causal 65-422 Hz                  (F1 0.4273 from Phase 2C)
  2. XGBoost hierarchical (normal gate +       (F1 0.4287 from Phase 2D)
     causal abnormal typer)
  3. Validation-selected augmented CNN (from Phase C)
  4. PGA MLP diagnostics (reported only; no stored probabilities)

Ensemble rules:
  - "soft_avg": diagnostic unweighted mean of available probabilities
  - "tree_only_soft_avg": pre-specified causal-tree average (flat + hier)

All on LORO region 43 test fold.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import causal_feature_cols
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = 42

XGB_PARAMS = dict(
    n_estimators=200, max_depth=6, learning_rate=0.1,
    objective='multi:softprob', num_class=5,
    tree_method='hist', n_jobs=-1, random_state=SEED,
    eval_metric='mlogloss', verbosity=0,
)


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def softmax_from_preds(preds, n_classes=5):
    """Convert hard predictions to one-hot softmax (degraded fallback)."""
    out = np.zeros((len(preds), n_classes), dtype=np.float32)
    out[np.arange(len(preds)), preds] = 1.0
    return out


def main():
    t_all = time.time()
    df = load_all()
    mask_te = df['region_lvl1'].values == HOLDOUT_REGION
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]
    y = df['label'].to_numpy(dtype=np.int32)
    print(f'train={len(tr)} test={len(te)}', flush=True)

    causal_cols = causal_feature_cols(43)
    X_causal = df[causal_cols].to_numpy(dtype=np.float32, copy=False)

    # ---- 1. XGB causal (flat) — get soft probs ----
    print('\n[1/4] XGBoost causal flat ...', flush=True)
    t0 = time.time()
    xgb_causal = XGBClassifier(**XGB_PARAMS)
    xgb_causal.fit(X_causal[tr], y[tr])
    p1 = xgb_causal.predict_proba(X_causal[te])
    f1_1 = f1_score(y[te], p1.argmax(1), average='macro', zero_division=0)
    acc_1 = accuracy_score(y[te], p1.argmax(1))
    print(f'  f1={f1_1:.4f} acc={acc_1:.4f} ({time.time() - t0:.0f}s)',
          flush=True)

    # ---- 2. XGB hierarchical ----
    print('\n[2/4] XGBoost hierarchical ...', flush=True)
    t0 = time.time()
    y_bin = (y == 4).astype(np.int32)
    stage1 = XGBClassifier(
        objective='binary:logistic', eval_metric='logloss',
        **{k: v for k, v in XGB_PARAMS.items()
           if k not in ['objective', 'num_class', 'eval_metric']})
    stage1.fit(X_causal[tr], y_bin[tr])
    stage1_prob_normal = stage1.predict_proba(X_causal[te])[:, 1]

    abn_tr = tr[y[tr] != 4]
    stage2 = XGBClassifier(
        objective='multi:softprob', num_class=4,
        eval_metric='mlogloss',
        **{k: v for k, v in XGB_PARAMS.items()
           if k not in ['objective', 'num_class', 'eval_metric']})
    stage2.fit(X_causal[abn_tr], y[abn_tr])
    stage2_prob_ab = stage2.predict_proba(X_causal[te])  # [N,4], classes 0..3

    # Combine into 5-class soft prob:
    # P(normal) = stage1_prob_normal
    # P(abnormal class k) = (1 - stage1_prob_normal) * stage2_prob_ab[:, k]
    p2 = np.zeros((len(te), 5), dtype=np.float32)
    p2[:, 4] = stage1_prob_normal
    for k in range(4):
        p2[:, k] = (1 - stage1_prob_normal) * stage2_prob_ab[:, k]
    f1_2 = f1_score(y[te], p2.argmax(1), average='macro', zero_division=0)
    acc_2 = accuracy_score(y[te], p2.argmax(1))
    print(f'  f1={f1_2:.4f} acc={acc_2:.4f} ({time.time() - t0:.0f}s)',
          flush=True)

    # ---- 3. Augmentation-trained CNN (use stored preds from Phase C) ----
    print('\n[3/4] Augmented CNN (best recipe) ...', flush=True)
    aug_path = os.path.join(RESULTS_DIR, 'augmentation_results.json')
    if not os.path.exists(aug_path):
        print('  !! no augmentation_results.json, skipping CNN', flush=True)
        p3 = softmax_from_preds(p1.argmax(1))  # fallback to xgb
        f1_3 = f1_1
        val_f1_3 = f1_1
    else:
        aug = json.load(open(aug_path))
        # Select a CNN recipe without looking at the Chungbuk test fold.
        # The augmentation file also contains an MLP ablation, which is kept out
        # of this CNN ensemble comparison.
        cnn_runs = [r for r in aug['runs'] if r['name'].startswith('cnn_')]
        best_aug_run = max(cnn_runs, key=lambda r: r['val_f1'])
        best_aug_name = best_aug_run['name']
        best_aug_pred = np.array(
            aug['test_preds'][best_aug_name], dtype=np.int32)
        p3 = softmax_from_preds(best_aug_pred)
        f1_3 = f1_score(y[te], best_aug_pred, average='macro',
                        zero_division=0)
        val_f1_3 = float(best_aug_run['val_f1'])
        print(f'  using {best_aug_name} val_f1={val_f1_3:.4f} '
              f'test_f1={f1_3:.4f}', flush=True)

    # ---- 4. PGA MLP best (from Phase B) ----
    print('\n[4/4] Best PGA MLP ...', flush=True)
    pga_path = os.path.join(RESULTS_DIR, 'pga_results.json')
    if not os.path.exists(pga_path):
        print('  !! no pga_results.json, skipping PGA', flush=True)
        p4 = p1   # fallback
        f1_4 = f1_1
    else:
        pga = json.load(open(pga_path))
        # PGA probabilities are not stored, so keep it out of ensemble
        # construction and report diagnostic results only.
        runs = pga['runs']
        best_pga_test = max(runs, key=lambda r: r['test_f1'])
        if all('val_f1' in r for r in runs):
            best_pga_val = max(runs, key=lambda r: r['val_f1'])
            f1_4 = float(best_pga_val['test_f1'])
            print(f'  PGA val-selected={best_pga_val["tag"]} '
                  f'val_f1={best_pga_val["val_f1"]:.4f} '
                  f'test_f1={f1_4:.4f}; soft probs unavailable, '
                  f'skipping ensemble contribution', flush=True)
        else:
            f1_4 = float(best_pga_test['test_f1'])
            print(f'  PGA val_f1 unavailable; best recorded test_f1='
                  f'{f1_4:.4f}; soft probs unavailable, skipping ensemble '
                  f'contribution', flush=True)
        pga_best_recorded_test_f1 = float(best_pga_test['test_f1'])
        p4 = None

    # ---- Ensemble ----
    ensembles = {}
    contribs = {'xgb_causal': (p1, f1_1, f1_1),
                'xgb_hier':   (p2, f1_2, f1_2)}
    if p3 is not None:
        contribs['cnn_val_selected'] = (p3, f1_3, val_f1_3)

    for name, (p, _test_f, _val_f) in contribs.items():
        ensembles[name] = {'f1': f1_score(y[te], p.argmax(1),
                                          average='macro', zero_division=0),
                           'acc': accuracy_score(y[te], p.argmax(1))}

    # Soft-average
    ps = np.stack([v[0] for v in contribs.values()], axis=0)
    mean_p = ps.mean(axis=0)
    ensembles['soft_avg'] = {
        'f1': float(f1_score(y[te], mean_p.argmax(1),
                             average='macro', zero_division=0)),
        'acc': float(accuracy_score(y[te], mean_p.argmax(1))),
        'members': list(contribs.keys()),
    }

    # XGB + hierarchical only (since CNN one-hot may degrade)
    tree_only = (p1 + p2) / 2.0
    ensembles['tree_only_soft_avg'] = {
        'f1': float(f1_score(y[te], tree_only.argmax(1),
                             average='macro', zero_division=0)),
        'acc': float(accuracy_score(y[te], tree_only.argmax(1))),
        'members': ['xgb_causal', 'xgb_hier'],
        'selection': 'pre-specified causal-tree average',
    }

    print('\n=== ENSEMBLE SUMMARY ===')
    print(f'{"method":>22s}  {"f1":>8s}  {"acc":>8s}')
    for name, m in ensembles.items():
        print(f'{name:>22s}  {m["f1"]:8.4f}  {m["acc"]:8.4f}')

    with open(os.path.join(RESULTS_DIR, 'ensemble_results.json'),
              'w') as f:
        json.dump({'ensembles': ensembles,
                   'best_individual_f1': max(f1_1, f1_2, f1_3),
                   'pga_val_selected_test_f1': f1_4,
                   'pga_best_recorded_test_f1':
                       locals().get('pga_best_recorded_test_f1', f1_4)},
                  f, indent=2)
    print(f'\nTotal {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
