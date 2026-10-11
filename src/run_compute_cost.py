"""Phase 1 / E6 — Computational cost benchmark.

Measures wall-clock training time, inference time per 1{,}000 rows,
peak RAM, and peak GPU memory for the strongest models on the
canonical Chungbuk LORO fold.

Models benchmarked (all on the 57 causal features unless noted):
  - XGBoost flat              (200 trees, depth 6, lr 0.1)
  - XGBoost hierarchical      (binary stage1 + 4-class stage2)
  - Tree ensemble             (soft average of flat + hierarchical;
                               deployment recipe — measured end-to-end)
  - LightGBM                  (200 trees, lr 0.1 — matches RQ5 baseline)
  - CatBoost                  (200 iter, lr 0.1 — matches RQ5 baseline)
  - RandomForest              (sklearn, 400 trees)
  - MLP ERM                   (causal-feature 57->256->128->5)
  - 1D-CNN ERM                (raw 513-band spectrum + 21 tabular)

Each model is re-trained from scratch in this script (no warm-cache
artefacts), and inference is timed on a 5,074-row test set then
normalised per 1k rows. Peak RAM is measured via tracemalloc and
peak GPU memory via torch.cuda.max_memory_allocated.

Compute: ~10 min total (CPU-only models ~30s each, deep models ~3 min).
"""
import gc
import json
import os
import sys
import time
import tracemalloc

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import causal_feature_cols
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = 42
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def make_split():
    df = load_all()
    cols_causal = causal_feature_cols(43)
    cols_tab = ['month'] + MAX_COLS
    cols_all_hz = HZ_COLS

    X_causal = df[cols_causal].to_numpy(dtype=np.float32, copy=False)
    X_tab = df[cols_tab].to_numpy(dtype=np.float32, copy=False)
    X_hz = df[cols_all_hz].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    te = np.where(df['region_lvl1'].values == HOLDOUT_REGION)[0]
    tr_pool = np.where(df['region_lvl1'].values != HOLDOUT_REGION)[0]
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(tr_pool))
    n_val = int(0.1 * len(tr_pool))
    va = tr_pool[perm[:n_val]]
    tr = tr_pool[perm[n_val:]]

    sc1 = StandardScaler().fit(X_causal[tr])
    sc2 = StandardScaler().fit(X_tab[tr])
    sc3 = StandardScaler().fit(X_hz[tr])
    X_causal_s = sc1.transform(X_causal).astype(np.float32)
    X_tab_s = sc2.transform(X_tab).astype(np.float32)
    X_hz_s = sc3.transform(X_hz).astype(np.float32)
    return {
        'X_causal': X_causal_s, 'X_tab': X_tab_s, 'X_hz': X_hz_s,
        'y': y, 'tr': tr, 'va': va, 'te': te,
    }


def reset_torch():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    gc.collect()


def benchmark_train_infer(name, train_fn, infer_fn, sp, n_warmup=2):
    reset_torch()
    tracemalloc.start()

    t0 = time.time()
    artefact = train_fn(sp)
    t_train = time.time() - t0

    # warmup
    for _ in range(n_warmup):
        _ = infer_fn(artefact, sp)

    # timed inference (3 runs, take median)
    times = []
    for _ in range(3):
        t1 = time.time()
        pred = infer_fn(artefact, sp)
        times.append(time.time() - t1)
    t_infer = float(np.median(times))
    n_te = len(sp['te'])
    t_per_1k = t_infer / max(1, n_te) * 1000.0

    cur, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_ram_mb = peak / (1024 ** 2)

    if torch.cuda.is_available():
        peak_gpu_mb = torch.cuda.max_memory_allocated() / (1024 ** 2)
    else:
        peak_gpu_mb = 0.0

    f1m = float(f1_score(sp['y'][sp['te']], pred, average='macro',
                         zero_division=0))
    acc = float(accuracy_score(sp['y'][sp['te']], pred))

    rec = {
        'model': name,
        'train_time_s': t_train,
        'inference_time_total_s': t_infer,
        'inference_ms_per_1k_rows': t_per_1k * 1000.0,   # ms
        'peak_ram_mb': peak_ram_mb,
        'peak_gpu_mb': peak_gpu_mb,
        'test_f1_macro': f1m,
        'test_acc': acc,
    }
    print(f'  {name:>22s}  train={t_train:6.1f}s  '
          f'infer/1k={t_per_1k*1000.0:6.2f}ms  '
          f'RAM={peak_ram_mb:6.0f}MB  GPU={peak_gpu_mb:6.0f}MB  '
          f'F1={f1m:.4f}  acc={acc:.4f}', flush=True)
    return rec


# ----- Train/Infer functions -----
def train_xgb_flat(sp):
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=5,
        tree_method='hist', n_jobs=-1, random_state=SEED,
        eval_metric='mlogloss', verbosity=0,
    )
    clf.fit(sp['X_causal'][sp['tr']], sp['y'][sp['tr']])
    return clf


def infer_xgb_flat(clf, sp):
    return clf.predict(sp['X_causal'][sp['te']])


def train_xgb_hier(sp):
    is_n = (sp['y'][sp['tr']] == 4).astype(np.int64)
    s1 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       objective='binary:logistic',
                       tree_method='hist', n_jobs=-1, random_state=SEED,
                       eval_metric='logloss', verbosity=0)
    s1.fit(sp['X_causal'][sp['tr']], is_n)
    abn = sp['y'][sp['tr']] != 4
    s2 = XGBClassifier(n_estimators=200, max_depth=6, learning_rate=0.1,
                       objective='multi:softprob', num_class=4,
                       tree_method='hist', n_jobs=-1, random_state=SEED,
                       eval_metric='mlogloss', verbosity=0)
    s2.fit(sp['X_causal'][sp['tr']][abn], sp['y'][sp['tr']][abn])
    return s1, s2


def infer_xgb_hier(clfs, sp):
    s1, s2 = clfs
    p_norm = s1.predict_proba(sp['X_causal'][sp['te']])[:, 1]
    p_abn = s2.predict_proba(sp['X_causal'][sp['te']])
    abn_pred = p_abn.argmax(1)
    pred = np.where(p_norm >= 0.5, 4, abn_pred)
    return pred.astype(np.int64)


def train_tree_ensemble(sp):
    """Pre-specified deployment recipe: tree-only soft average of the flat
    causal XGBoost and the hierarchical (binary gate + 4-class typer).

    Cost is measured for the entire inference path that produces the
    ensemble's probability vector, not for a synthetic proxy.
    """
    flat = train_xgb_flat(sp)
    hier = train_xgb_hier(sp)
    return flat, hier


def infer_tree_ensemble(arts, sp):
    flat, (s1, s2) = arts
    Xte = sp['X_causal'][sp['te']]
    p_flat = flat.predict_proba(Xte)          # (n, 5)
    p_norm = s1.predict_proba(Xte)[:, 1]      # (n,)
    p_abn = s2.predict_proba(Xte)              # (n, 4) over [in, out, noise, other]
    # Reconstruct hierarchical 5-class probability: weight Stage-2 by P(abn)
    p_hier = np.zeros_like(p_flat)
    p_hier[:, 4] = p_norm
    p_hier[:, :4] = p_abn * (1.0 - p_norm)[:, None]
    p_avg = 0.5 * (p_flat + p_hier)
    return p_avg.argmax(1).astype(np.int64)


def train_lgbm(sp):
    import lightgbm as lgb
    cls_w = np.bincount(sp['y'][sp['tr']]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    sw = cls_w[sp['y'][sp['tr']]]
    # Hyper-parameters held identical to run_extra_baselines.py so cost
    # is measured for the exact same model whose accuracy is reported in
    # the baselines table (RQ5 / Table 6).
    clf = lgb.LGBMClassifier(
        n_estimators=200, learning_rate=0.1, num_leaves=63, max_depth=6,
        objective='multiclass', num_class=5,
        random_state=SEED, n_jobs=-1, verbose=-1,
    )
    clf.fit(sp['X_causal'][sp['tr']], sp['y'][sp['tr']], sample_weight=sw)
    return clf


def infer_lgbm(clf, sp):
    return clf.predict(sp['X_causal'][sp['te']])


def train_catboost(sp):
    from catboost import CatBoostClassifier
    cls_w = np.bincount(sp['y'][sp['tr']]).astype(np.float64)
    cls_w = cls_w.sum() / (5 * cls_w)
    # Same config as run_extra_baselines.py (RQ5) so cost and accuracy
    # apply to the exact same model.
    clf = CatBoostClassifier(
        iterations=200, depth=6, learning_rate=0.1,
        loss_function='MultiClass', random_seed=SEED,
        class_weights=cls_w.tolist(), verbose=False,
    )
    clf.fit(sp['X_causal'][sp['tr']], sp['y'][sp['tr']])
    return clf


def infer_catboost(clf, sp):
    return clf.predict(sp['X_causal'][sp['te']]).reshape(-1).astype(np.int64)


def train_rf(sp):
    clf = RandomForestClassifier(n_estimators=400, max_depth=20,
                                 n_jobs=-1, random_state=SEED,
                                 class_weight='balanced')
    clf.fit(sp['X_causal'][sp['tr']], sp['y'][sp['tr']])
    return clf


def infer_rf(clf, sp):
    return clf.predict(sp['X_causal'][sp['te']])


# Deep models
class MLP(nn.Module):
    def __init__(self, n_in, n_classes=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_in, 256), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x):
        return self.net(x)


class CNN(nn.Module):
    def __init__(self, tab_dim, n_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, 9, padding=4), nn.BatchNorm1d(32),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, 7, padding=3), nn.BatchNorm1d(64),
            nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, 5, padding=2), nn.BatchNorm1d(128),
            nn.ReLU(), nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + tab_dim, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_tab, x_raw):
        if x_raw.dim() == 2:
            x_raw = x_raw.unsqueeze(1)
        h = self.conv(x_raw).squeeze(-1)
        h = torch.cat([h, x_tab], dim=1)
        return self.head(h)


def train_mlp_erm(sp, epochs=40):
    torch.manual_seed(SEED)
    model = MLP(n_in=sp['X_causal'].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_w = np.bincount(sp['y'][sp['tr']]).astype(np.float32)
    cls_w = cls_w.sum() / (5 * cls_w)
    cls_w_t = torch.from_numpy(cls_w).to(DEVICE)

    Xtr = torch.from_numpy(sp['X_causal'][sp['tr']]).to(DEVICE)
    ytr = torch.from_numpy(sp['y'][sp['tr']]).to(DEVICE)
    bs = 528
    n = len(sp['tr'])
    rng = np.random.RandomState(SEED)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits = model(Xtr[b])
            loss = F.cross_entropy(logits, ytr[b], weight=cls_w_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def infer_mlp(model, sp):
    model.eval()
    with torch.no_grad():
        Xte = torch.from_numpy(sp['X_causal'][sp['te']]).to(DEVICE)
        return model(Xte).argmax(1).cpu().numpy()


def train_cnn_erm(sp, epochs=20):
    torch.manual_seed(SEED)
    model = CNN(tab_dim=sp['X_tab'].shape[1]).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    cls_w = np.bincount(sp['y'][sp['tr']]).astype(np.float32)
    cls_w = cls_w.sum() / (5 * cls_w)
    cls_w_t = torch.from_numpy(cls_w).to(DEVICE)

    Xhz_tr = torch.from_numpy(sp['X_hz'][sp['tr']]).to(DEVICE)
    Xtab_tr = torch.from_numpy(sp['X_tab'][sp['tr']]).to(DEVICE)
    ytr = torch.from_numpy(sp['y'][sp['tr']]).to(DEVICE)
    bs = 256
    n = len(sp['tr'])
    rng = np.random.RandomState(SEED)
    for ep in range(epochs):
        model.train()
        perm = rng.permutation(n)
        for i in range(0, n, bs):
            b = perm[i:i + bs]
            logits = model(Xtab_tr[b], Xhz_tr[b])
            loss = F.cross_entropy(logits, ytr[b], weight=cls_w_t)
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def infer_cnn(model, sp):
    model.eval()
    preds = []
    with torch.no_grad():
        Xhz = torch.from_numpy(sp['X_hz'][sp['te']]).to(DEVICE)
        Xtab = torch.from_numpy(sp['X_tab'][sp['te']]).to(DEVICE)
        for i in range(0, len(sp['te']), 1024):
            logits = model(Xtab[i:i + 1024], Xhz[i:i + 1024])
            preds.append(logits.argmax(1).cpu().numpy())
    return np.concatenate(preds)


def main():
    print(f'DEVICE = {DEVICE}', flush=True)
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f'  GPU {i}: {torch.cuda.get_device_name(i)}', flush=True)

    sp = make_split()
    print(f'  X_causal={sp["X_causal"].shape}  X_hz={sp["X_hz"].shape}  '
          f'tr={len(sp["tr"])}  te={len(sp["te"])}', flush=True)

    print('\n=== Compute-cost benchmark ===', flush=True)
    bench = []
    bench.append(benchmark_train_infer('xgboost_flat',
                                       train_xgb_flat, infer_xgb_flat, sp))
    bench.append(benchmark_train_infer('xgboost_hier',
                                       train_xgb_hier, infer_xgb_hier, sp))
    bench.append(benchmark_train_infer('tree_ensemble',
                                       train_tree_ensemble,
                                       infer_tree_ensemble, sp))
    bench.append(benchmark_train_infer('lightgbm',
                                       train_lgbm, infer_lgbm, sp))
    bench.append(benchmark_train_infer('catboost',
                                       train_catboost, infer_catboost, sp))
    bench.append(benchmark_train_infer('random_forest',
                                       train_rf, infer_rf, sp))
    bench.append(benchmark_train_infer('mlp_erm',
                                       train_mlp_erm, infer_mlp, sp))
    bench.append(benchmark_train_infer('cnn_erm',
                                       train_cnn_erm, infer_cnn, sp))

    out_path = os.path.join(RESULTS_DIR, 'compute_cost_results.json')
    with open(out_path, 'w') as f:
        json.dump({'device': DEVICE, 'seed': SEED,
                   'n_train': len(sp['tr']), 'n_test': len(sp['te']),
                   'benchmarks': bench}, f, indent=2)
    print(f'\nSaved {out_path}')

    print('\n=== COMPUTE COST SUMMARY ===')
    print(f'{"model":>22s} {"train_s":>8s} {"infer/1k_ms":>12s} '
          f'{"RAM_MB":>8s} {"GPU_MB":>8s} {"F1":>7s} {"acc":>7s}')
    for r in bench:
        print(f'{r["model"]:>22s} {r["train_time_s"]:>8.1f} '
              f'{r["inference_ms_per_1k_rows"]:>12.2f} '
              f'{r["peak_ram_mb"]:>8.0f} {r["peak_gpu_mb"]:>8.0f} '
              f'{r["test_f1_macro"]:>7.4f} {r["test_acc"]:>7.4f}')


if __name__ == '__main__':
    main()
