"""Temporal-split validation: train on earlier recordings, test on later ones.

Sorts all data by ldate (inspection date), splits at the 70th-percentile
date, and evaluates the deployment recipe (causal XGBoost flat,
hierarchical, tree-ensemble) on the chronologically later portion.

This simulates a realistic deployment scenario where a model trained on
historical data must generalise to future recordings — complementary to
the spatial LORO split.

Outputs results/temporal_split_results.json.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import nested_bands_cached
from data import HZ_COLS, MAX_COLS, LABEL_NAMES, load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42
TRAIN_FRAC = 0.70


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def make_temporal_split():
    df = load_all()
    cols_signal = ['month'] + HZ_COLS + MAX_COLS

    df_sorted = df.sort_values('ldate_int').reset_index(drop=True)

    cutoff_idx = int(len(df_sorted) * TRAIN_FRAC)
    cutoff_date = int(df_sorted.iloc[cutoff_idx]['ldate_int'])

    tr_mask = df_sorted['ldate_int'] < cutoff_date
    te_mask = df_sorted['ldate_int'] >= cutoff_date

    # Bands are selected from pre-cutoff rows only, split by region for the
    # inner loop, so nothing after the cutoff influences the feature set.
    # The retired 65-422 Hz subset was selected on region 43 across the whole
    # time range; see results/FINDINGS_NESTED_BANDS.md.
    X_sig_raw = df_sorted[cols_signal].to_numpy(dtype=np.float32, copy=False)
    y_raw = df_sorted['label'].to_numpy(dtype=np.int32)
    era = np.where(te_mask.values, 'post', 'pre')
    cols_causal, _tman = nested_bands_cached(
        df_sorted, X_sig_raw, y_raw, cols_signal, 'post', kind='temporal',
        env=era, inner_env=df_sorted['region_lvl1'].values)
    tr_idx = np.where(tr_mask)[0]
    te_idx = np.where(te_mask)[0]

    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_idx))
    n_val = int(0.10 * len(tr_idx))
    va_idx = tr_idx[perm[:n_val]]
    tr_idx_final = tr_idx[perm[n_val:]]

    X_causal = df_sorted[cols_causal].to_numpy(dtype=np.float32)
    X_signal = df_sorted[cols_signal].to_numpy(dtype=np.float32)
    y = df_sorted['label'].to_numpy(dtype=np.int64)

    sc_c = StandardScaler().fit(X_causal[tr_idx_final])
    sc_s = StandardScaler().fit(X_signal[tr_idx_final])
    X_causal_s = sc_c.transform(X_causal).astype(np.float32)
    X_signal_s = sc_s.transform(X_signal).astype(np.float32)

    tr_regions = df_sorted.loc[tr_mask, 'region_lvl1'].unique().tolist()
    te_regions = df_sorted.loc[te_mask, 'region_lvl1'].unique().tolist()
    tr_dates = (int(df_sorted.loc[tr_mask, 'ldate_int'].min()),
                int(df_sorted.loc[tr_mask, 'ldate_int'].max()))
    te_dates = (int(df_sorted.loc[te_mask, 'ldate_int'].min()),
                int(df_sorted.loc[te_mask, 'ldate_int'].max()))

    tr_class_dist = [int((y[tr_idx_final] == c).sum()) for c in range(5)]
    te_class_dist = [int((y[te_idx] == c).sum()) for c in range(5)]

    info = {
        'cutoff_date': cutoff_date,
        'train_size': len(tr_idx_final),
        'val_size': len(va_idx),
        'test_size': len(te_idx),
        'train_date_range': tr_dates,
        'test_date_range': te_dates,
        'train_regions': sorted(tr_regions),
        'test_regions': sorted(te_regions),
        'train_class_dist': dict(zip(LABEL_NAMES, tr_class_dist)),
        'test_class_dist': dict(zip(LABEL_NAMES, te_class_dist)),
    }

    return (X_causal_s, X_signal_s, y, tr_idx_final, va_idx, te_idx,
            cols_causal, cols_signal, info)


def compute_metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'weighted_acc': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro',
                                   zero_division=0)),
        'f1_weighted': float(f1_score(y_true, y_pred, average='weighted',
                                      zero_division=0)),
        'f1_per_class': {LABEL_NAMES[i]: float(f1_score(
            y_true, y_pred, labels=[i], average='macro', zero_division=0))
            for i in range(5)},
        'cm': confusion_matrix(y_true, y_pred, labels=list(range(5))).tolist(),
    }


def run_xgb_flat(X, y, tr, va, te, feature_name):
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sw = cls_w[y[tr]]
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=5,
        tree_method='hist', random_state=SEED, n_jobs=-1,
        use_label_encoder=False, eval_metric='mlogloss',
    )
    clf.fit(X[tr], y[tr], sample_weight=sw)
    pred = clf.predict(X[te])
    prob = clf.predict_proba(X[te])
    return compute_metrics(y[te], pred), prob, clf


def run_xgb_hierarchical(X, y, tr, va, te):
    y_bin = (y != 4).astype(np.int64)
    cls_w_bin = np.bincount(y_bin[tr]).astype(np.float64)
    cls_w_bin = cls_w_bin.sum() / (2 * cls_w_bin)
    sw_bin = cls_w_bin[y_bin[tr]]

    stage1 = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='binary:logistic',
        tree_method='hist', random_state=SEED, n_jobs=-1,
        use_label_encoder=False, eval_metric='logloss',
    )
    stage1.fit(X[tr], y_bin[tr], sample_weight=sw_bin)

    abn_mask_tr = y[tr] != 4
    tr_abn = tr[abn_mask_tr]
    y_abn = y[tr_abn]
    cls_w_4 = np.bincount(y_abn, minlength=5).astype(np.float64)[:4]
    cls_w_4 = cls_w_4.sum() / (4 * np.maximum(cls_w_4, 1))
    sw_4 = cls_w_4[y_abn]

    stage2 = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=4,
        tree_method='hist', random_state=SEED, n_jobs=-1,
        use_label_encoder=False, eval_metric='mlogloss',
    )
    stage2.fit(X[tr_abn], y_abn, sample_weight=sw_4)

    prob_abn = stage1.predict_proba(X[te])[:, 1]
    prob_4class = stage2.predict_proba(X[te])
    prob_5 = np.zeros((len(te), 5), dtype=np.float64)
    for i in range(4):
        prob_5[:, i] = prob_abn * prob_4class[:, i]
    prob_5[:, 4] = 1.0 - prob_abn
    pred = prob_5.argmax(axis=1)

    return compute_metrics(y[te], pred), prob_5


def main():
    print('=== Temporal Split Validation ===', flush=True)
    (X_c, X_s, y, tr, va, te,
     cols_c, cols_s, split_info) = make_temporal_split()

    print(f'  Cutoff date: {split_info["cutoff_date"]}')
    print(f'  Train: {split_info["train_size"]}  '
          f'Val: {split_info["val_size"]}  '
          f'Test: {split_info["test_size"]}')
    print(f'  Train dates: {split_info["train_date_range"]}')
    print(f'  Test  dates: {split_info["test_date_range"]}')
    print(f'  Train class dist: {split_info["train_class_dist"]}')
    print(f'  Test  class dist: {split_info["test_class_dist"]}')

    results = {'split_info': split_info, 'models': {}}

    # 1. XGBoost flat on causal features
    print('\n--- XGBoost flat (causal 57 features) ---', flush=True)
    t0 = time.time()
    m_flat_c, prob_flat_c, _ = run_xgb_flat(X_c, y, tr, va, te, 'causal')
    results['models']['xgb_flat_causal'] = {
        **m_flat_c, 'time_s': round(time.time() - t0, 1),
        'n_features': len(cols_c),
    }
    print(f'  F1={m_flat_c["f1_macro"]:.4f}  Acc={m_flat_c["acc"]:.4f}')

    # 2. XGBoost flat on signal_all features
    print('\n--- XGBoost flat (signal_all 534 features) ---', flush=True)
    t0 = time.time()
    m_flat_s, prob_flat_s, _ = run_xgb_flat(X_s, y, tr, va, te, 'signal')
    results['models']['xgb_flat_signal'] = {
        **m_flat_s, 'time_s': round(time.time() - t0, 1),
        'n_features': len(cols_s),
    }
    print(f'  F1={m_flat_s["f1_macro"]:.4f}  Acc={m_flat_s["acc"]:.4f}')

    # 3. XGBoost hierarchical on causal features
    print('\n--- XGBoost hierarchical (causal) ---', flush=True)
    t0 = time.time()
    m_hier, prob_hier = run_xgb_hierarchical(X_c, y, tr, va, te)
    results['models']['xgb_hier_causal'] = {
        **m_hier, 'time_s': round(time.time() - t0, 1),
        'n_features': len(cols_c),
    }
    print(f'  F1={m_hier["f1_macro"]:.4f}  Acc={m_hier["acc"]:.4f}')

    # 4. Tree-only ensemble (soft average of flat causal + hier)
    print('\n--- Tree-only ensemble (causal) ---', flush=True)
    prob_ens = 0.5 * prob_flat_c + 0.5 * prob_hier
    pred_ens = prob_ens.argmax(axis=1)
    m_ens = compute_metrics(y[te], pred_ens)
    results['models']['tree_ensemble_causal'] = {
        **m_ens, 'n_features': len(cols_c),
    }
    print(f'  F1={m_ens["f1_macro"]:.4f}  Acc={m_ens["acc"]:.4f}')

    # 5. LightGBM on causal features
    print('\n--- LightGBM (causal) ---', flush=True)
    import lightgbm as lgb
    t0 = time.time()
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sw = cls_w[y[tr]]
    lgb_clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1,
        num_leaves=63, max_depth=6,
        objective='multiclass', num_class=5,
        random_state=SEED, n_jobs=4, verbose=-1,
    )
    lgb_clf.fit(X_c[tr], y[tr], sample_weight=sw)
    pred_lgb = lgb_clf.predict(X_c[te])
    m_lgb = compute_metrics(y[te], pred_lgb)
    results['models']['lightgbm_causal'] = {
        **m_lgb, 'time_s': round(time.time() - t0, 1),
        'n_features': len(cols_c),
    }
    print(f'  F1={m_lgb["f1_macro"]:.4f}  Acc={m_lgb["acc"]:.4f}')

    # 6. CatBoost on causal features
    print('\n--- CatBoost (causal) ---', flush=True)
    from catboost import CatBoostClassifier
    t0 = time.time()
    cb_clf = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        loss_function='MultiClass', random_seed=SEED,
        class_weights=cls_w.tolist(),
        thread_count=4, verbose=False,
    )
    cb_clf.fit(X_c[tr], y[tr])
    pred_cb = cb_clf.predict(X_c[te]).reshape(-1).astype(np.int64)
    m_cb = compute_metrics(y[te], pred_cb)
    results['models']['catboost_causal'] = {
        **m_cb, 'time_s': round(time.time() - t0, 1),
        'n_features': len(cols_c),
    }
    print(f'  F1={m_cb["f1_macro"]:.4f}  Acc={m_cb["acc"]:.4f}')

    # Summary
    print('\n=== TEMPORAL SPLIT SUMMARY ===')
    print(f'{"Model":>25s}  {"F1 macro":>10s}  {"Acc":>8s}')
    for name, m in results['models'].items():
        print(f'{name:>25s}  {m["f1_macro"]:10.4f}  {m["acc"]:8.4f}')

    # Causal vs signal delta
    delta_f1 = (results['models']['xgb_flat_causal']['f1_macro']
                - results['models']['xgb_flat_signal']['f1_macro'])
    results['causal_vs_signal_delta_f1'] = round(delta_f1, 4)
    print(f'\nCausal vs Signal delta F1: {delta_f1:+.4f}')

    out_path = os.path.join(RESULTS_DIR, 'temporal_split_results.json')
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
