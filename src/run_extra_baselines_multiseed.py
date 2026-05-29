"""Multi-seed evaluation of additional tabular baselines on the
canonical LORO Chungbuk fold (region_lvl1 == 43).

Runs LightGBM, CatBoost, RandomForest, and TabNet across 5 seeds
{42, 0, 1, 2, 3} to enable statistical comparison with XGBoost.

Outputs results/extra_baselines_multiseed.json with per-seed and
aggregate (mean ± std) results plus paired Wilcoxon tests.
"""
import json
import os
import sys
import time

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, LABEL_NAMES, load_all

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEEDS = [42, 0, 1, 2, 3]
SPLIT_SEED = 42


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def make_split():
    df = load_all()
    cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    te_mask = df['region_lvl1'].values == HOLDOUT_REGION
    tr_pool = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(tr_pool))
    n_val = int(0.10 * len(tr_pool))
    va = tr_pool[perm[:n_val]]
    tr = tr_pool[perm[n_val:]]

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)
    return Xs, y, tr, va, te_idx, cols


def make_split_full():
    """Full 534-feature split for TabNet full-feature experiment."""
    df = load_all()
    cols = ['month'] + HZ_COLS + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    te_mask = df['region_lvl1'].values == HOLDOUT_REGION
    tr_pool = np.where(~te_mask)[0]
    te_idx = np.where(te_mask)[0]

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(tr_pool))
    n_val = int(0.10 * len(tr_pool))
    va = tr_pool[perm[:n_val]]
    tr = tr_pool[perm[n_val:]]

    sc = StandardScaler().fit(X[tr])
    Xs = sc.transform(X).astype(np.float32)
    return Xs, y, tr, va, te_idx, cols


def metrics(y_true, y_pred):
    return {
        'acc': float(accuracy_score(y_true, y_pred)),
        'f1_macro': float(f1_score(y_true, y_pred, average='macro',
                                   zero_division=0)),
        'f1_per_class': [float(f1_score(y_true, y_pred, labels=[i],
                                        average='macro', zero_division=0))
                         for i in range(5)],
    }


def run_random_forest(X, y, tr, va, te, seed):
    clf = RandomForestClassifier(n_estimators=400, max_depth=20,
                                 n_jobs=4, random_state=seed,
                                 class_weight='balanced')
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te])
    return metrics(y[te], pred)


def run_lightgbm(X, y, tr, va, te, seed):
    import lightgbm as lgb
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sample_w = cls_w[y[tr]]
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1,
        num_leaves=63, max_depth=6,
        objective='multiclass', num_class=5,
        random_state=seed, n_jobs=4, verbose=-1,
    )
    clf.fit(X[tr], y[tr], sample_weight=sample_w)
    pred = clf.predict(X[te])
    return metrics(y[te], pred)


def run_catboost(X, y, tr, va, te, seed):
    from catboost import CatBoostClassifier
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    clf = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        loss_function='MultiClass', random_seed=seed,
        class_weights=cls_w.tolist(),
        thread_count=4, verbose=False,
    )
    clf.fit(X[tr], y[tr])
    pred = clf.predict(X[te]).reshape(-1).astype(np.int64)
    return metrics(y[te], pred)


def run_tabnet(X, y, tr, va, te, seed):
    from pytorch_tabnet.tab_model import TabNetClassifier
    cls_w = np.bincount(y[tr]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sample_w = cls_w[y[tr]]
    clf = TabNetClassifier(
        n_d=32, n_a=32, n_steps=4,
        gamma=1.5, lambda_sparse=1e-4,
        seed=seed, verbose=0,
        device_name='cpu',
    )
    clf.fit(X[tr], y[tr],
            eval_set=[(X[va], y[va])], eval_metric=['accuracy'],
            max_epochs=80, patience=15, batch_size=2048,
            virtual_batch_size=512, num_workers=0,
            weights=sample_w.astype(np.float32))
    pred = clf.predict(X[te])
    return metrics(y[te], pred)


def paired_wilcoxon(a, b):
    if len(a) != len(b) or len(a) < 3:
        return float('nan')
    try:
        from scipy.stats import wilcoxon
        res = wilcoxon(a, b, zero_method='wilcox', alternative='two-sided')
        return float(res.pvalue)
    except Exception:
        return float('nan')


def main():
    print('=== Multi-seed Extra Baselines ===', flush=True)
    print('Building causal-feature split...', flush=True)
    X, y, tr, va, te, cols = make_split()
    print(f'  Causal: X={X.shape}  tr={len(tr)}  va={len(va)}  te={len(te)}',
          flush=True)

    print('Building full-feature split for TabNet...', flush=True)
    X_full, y_full, tr_f, va_f, te_f, cols_full = make_split_full()
    print(f'  Full: X={X_full.shape}', flush=True)

    models = {
        'lightgbm': run_lightgbm,
        'catboost': run_catboost,
        'random_forest': run_random_forest,
        'tabnet_causal': run_tabnet,
    }

    all_runs = {}
    for name, fn in models.items():
        all_runs[name] = []
        for seed in SEEDS:
            print(f'\n  {name} seed={seed}...', end=' ', flush=True)
            t0 = time.time()
            try:
                m = fn(X, y, tr, va, te, seed)
                m['seed'] = seed
                m['time_s'] = round(time.time() - t0, 1)
                all_runs[name].append(m)
                print(f'f1={m["f1_macro"]:.4f} acc={m["acc"]:.4f} '
                      f'({m["time_s"]:.0f}s)', flush=True)
            except Exception as exc:
                all_runs[name].append({'seed': seed, 'error': str(exc)})
                print(f'FAILED: {exc}', flush=True)

    # TabNet on full features
    print('\n\n=== TabNet Full-Feature (534) ===', flush=True)
    all_runs['tabnet_full'] = []
    for seed in SEEDS:
        print(f'  tabnet_full seed={seed}...', end=' ', flush=True)
        t0 = time.time()
        try:
            m = run_tabnet(X_full, y_full, tr_f, va_f, te_f, seed)
            m['seed'] = seed
            m['time_s'] = round(time.time() - t0, 1)
            all_runs['tabnet_full'].append(m)
            print(f'f1={m["f1_macro"]:.4f} acc={m["acc"]:.4f} '
                  f'({m["time_s"]:.0f}s)', flush=True)
        except Exception as exc:
            all_runs['tabnet_full'].append({'seed': seed, 'error': str(exc)})
            print(f'FAILED: {exc}', flush=True)

    # Aggregate
    summary = {}
    for name, runs in all_runs.items():
        valid = [r for r in runs if 'f1_macro' in r]
        if not valid:
            summary[name] = {'error': 'all seeds failed'}
            continue
        f1s = [r['f1_macro'] for r in valid]
        accs = [r['acc'] for r in valid]
        summary[name] = {
            'n_seeds': len(valid),
            'seeds': [r['seed'] for r in valid],
            'f1_macro_mean': round(float(np.mean(f1s)), 4),
            'f1_macro_std': round(float(np.std(f1s, ddof=1)) if len(f1s) > 1 else 0.0, 4),
            'f1_macro_per_seed': {r['seed']: r['f1_macro'] for r in valid},
            'acc_mean': round(float(np.mean(accs)), 4),
            'acc_std': round(float(np.std(accs, ddof=1)) if len(accs) > 1 else 0.0, 4),
            'f1_per_class_mean': [
                round(float(np.mean([r['f1_per_class'][c] for r in valid])), 4)
                for c in range(5)
            ],
            'per_seed': valid,
        }

    # Paired Wilcoxon: each model vs XGBoost causal (deterministic 0.4273)
    xgb_ref = 0.4273
    tests = []
    for name, s in summary.items():
        if 'f1_macro_per_seed' not in s or s['n_seeds'] < 5:
            continue
        f1s = [s['f1_macro_per_seed'][seed] for seed in SEEDS
               if seed in s['f1_macro_per_seed']]
        xgb_vals = [xgb_ref] * len(f1s)
        p = paired_wilcoxon(f1s, xgb_vals)
        tests.append({
            'comparison': f'{name} vs XGBoost_causal',
            'model_mean': s['f1_macro_mean'],
            'xgb_ref': xgb_ref,
            'diff': round(s['f1_macro_mean'] - xgb_ref, 4),
            'wilcoxon_p': round(p, 4) if not np.isnan(p) else 'n/a',
            'n_seeds': len(f1s),
        })

    # Inter-model tests
    for a_name in ['lightgbm', 'catboost']:
        for b_name in ['random_forest', 'tabnet_causal']:
            if a_name in summary and b_name in summary:
                a_s = summary[a_name]
                b_s = summary[b_name]
                if a_s.get('n_seeds', 0) >= 5 and b_s.get('n_seeds', 0) >= 5:
                    a_f1 = [a_s['f1_macro_per_seed'][s] for s in SEEDS]
                    b_f1 = [b_s['f1_macro_per_seed'][s] for s in SEEDS]
                    p = paired_wilcoxon(a_f1, b_f1)
                    tests.append({
                        'comparison': f'{a_name} vs {b_name}',
                        'a_mean': a_s['f1_macro_mean'],
                        'b_mean': b_s['f1_macro_mean'],
                        'diff': round(a_s['f1_macro_mean'] - b_s['f1_macro_mean'], 4),
                        'wilcoxon_p': round(p, 4) if not np.isnan(p) else 'n/a',
                    })

    result = {
        'description': 'Multi-seed extra baselines on Chungbuk LORO fold',
        'seeds': SEEDS,
        'split_seed': SPLIT_SEED,
        'holdout_region': HOLDOUT_REGION,
        'n_causal_features': len(cols),
        'n_full_features': len(cols_full),
        'summary': summary,
        'paired_tests': tests,
    }

    # Print summary
    print('\n\n=== MULTI-SEED SUMMARY ===')
    print(f'{"Model":>20s}  {"n":>3s}  {"F1 mean":>10s}  {"F1 std":>8s}  '
          f'{"Acc mean":>10s}')
    for name, s in summary.items():
        if 'f1_macro_mean' in s:
            print(f'{name:>20s}  {s["n_seeds"]:>3d}  '
                  f'{s["f1_macro_mean"]:10.4f}  {s["f1_macro_std"]:8.4f}  '
                  f'{s["acc_mean"]:10.4f}')

    print('\n=== PAIRED TESTS ===')
    for t in tests:
        print(f'  {t["comparison"]}: diff={t.get("diff", "?")} '
              f'p={t.get("wilcoxon_p", "?")}')

    out_path = os.path.join(RESULTS_DIR, 'extra_baselines_multiseed.json')
    with open(out_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    main()
