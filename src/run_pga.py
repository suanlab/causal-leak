"""Phase B (Strengthening 3) — Permutation-Guided Attention (PGA).

A novel method contribution of this paper: band-level attention weights
learned jointly with the classifier, regularised to match a permutation-
importance prior computed on training-only internal validation.

Loss:  L = weighted_CE(f(attn * x), y)
         + lambda_c * KL(softmax(attn) || target_bands)

where target_bands is the soft prior over the 513 HZ bands derived from
ReLU-rectified permutation Δf1 measured inside the training 11 regions
(no test-set leakage), normalised to a distribution.

Evaluated under LORO region 43, compared against:
  - causal-band XGBoost (strongest baseline so far)
  - MLP GroupDRO (best plain deep model)
  - PGA MLP at several lambda_c values
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = int(os.environ.get('CAUSAL_SEED', '42'))   # variable for multi-seed
SPLIT_SEED = 42                                    # fixed val/train split
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed=None):
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------------------------------------------------------------
# Step 1: Compute permutation-importance prior on INTERNAL train CV
# ---------------------------------------------------------------

def build_train_cv_band_prior(df, tr_idx, n_bands=513):
    """Compute per-band permutation Δf1 using a held-out slice of
    training data (no region 43 involvement)."""
    rng = np.random.RandomState(SPLIT_SEED)
    # Internal 80/20 split on tr_idx
    perm = rng.permutation(len(tr_idx))
    n_val = int(0.2 * len(tr_idx))
    tr_inner = tr_idx[perm[n_val:]]
    va_inner = tr_idx[perm[:n_val]]

    cols = ['month'] + HZ_COLS + MAX_COLS
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)

    print(f'  [prior] inner train={len(tr_inner)} inner val={len(va_inner)}',
          flush=True)
    clf = XGBClassifier(
        n_estimators=200, max_depth=6, learning_rate=0.1,
        objective='multi:softprob', num_class=5,
        tree_method='hist', n_jobs=-1, random_state=SPLIT_SEED,
        eval_metric='mlogloss', verbosity=0,
    )
    clf.fit(X[tr_inner], y[tr_inner])
    base = clf.predict(X[va_inner])
    base_f1 = f1_score(y[va_inner], base, average='macro', zero_division=0)
    print(f'  [prior] base internal-val F1={base_f1:.4f}', flush=True)

    # Per-band permutation (coarse: group bands into 10 log bins for speed,
    # then assign each band the group-level importance)
    edges = np.geomspace(10, 5120, 11)
    band_f = np.array([i for i in range(0, 5130, 10)])
    hz_col_indices = [cols.index(c) for c in HZ_COLS]
    band_importance = np.zeros(len(HZ_COLS), dtype=np.float64)
    X_val = X[va_inner].copy()
    for i in range(10):
        lo, hi = edges[i], edges[i + 1]
        mask = (band_f >= lo) & (band_f < hi)
        if not mask.any():
            continue
        group_cols = [hz_col_indices[j] for j, m in enumerate(mask) if m]
        order = rng.permutation(len(va_inner))
        Xp = X_val.copy()
        Xp[:, group_cols] = X_val[order][:, group_cols]
        pred = clf.predict(Xp)
        p_f1 = f1_score(y[va_inner], pred, average='macro',
                        zero_division=0)
        drop = base_f1 - p_f1
        band_importance[mask] = drop

    # Prior: ReLU-rectified, normalised
    prior = np.maximum(band_importance, 0.0)
    if prior.sum() > 0:
        prior = prior / prior.sum()
    else:
        prior = np.ones_like(prior) / len(prior)
    print(f'  [prior] rectified sum = {band_importance.clip(min=0).sum():.4f}',
          flush=True)
    print(f'  [prior] band groups with positive Δf1: '
          f'{(band_importance > 0).sum()} / {len(band_importance)}',
          flush=True)
    return prior, band_importance


# ---------------------------------------------------------------
# Step 2: PGA model
# ---------------------------------------------------------------

class PGAClassifier(nn.Module):
    def __init__(self, n_bands=513, n_extra=21, n_classes=5,
                 hidden=(256, 128), dropout=0.2,
                 init_attn_prior: np.ndarray | None = None):
        super().__init__()
        init = torch.zeros(n_bands, dtype=torch.float32)
        if init_attn_prior is not None:
            # Initialise as logits whose softmax ≈ prior
            p = np.clip(init_attn_prior, 1e-6, 1.0)
            init = torch.tensor(np.log(p / p.mean()), dtype=torch.float32)
        self.band_logits = nn.Parameter(init)
        layers = []
        prev = n_bands + n_extra
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.head = nn.Sequential(*layers)

    def forward(self, x_bands, x_extra):
        attn = F.softmax(self.band_logits, dim=0)  # [n_bands]
        x = x_bands * attn.unsqueeze(0) * attn.numel()  # preserve scale
        return self.head(torch.cat([x, x_extra], dim=1)), attn


# ---------------------------------------------------------------
# Step 3: training loop
# ---------------------------------------------------------------

def class_weights(y, num_classes=5):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_pga(X_bands, X_extra, y, tr, va, te,
              prior: np.ndarray, lambda_c: float,
              epochs=100, bs=512, lr=1e-3, wd=1e-4,
              init_with_prior: bool = True):
    set_seed(SEED)
    n_bands = X_bands.shape[1]
    n_extra = X_extra.shape[1]
    model = PGAClassifier(
        n_bands=n_bands, n_extra=n_extra,
        init_attn_prior=prior if init_with_prior else None).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    cls_w = class_weights(y[tr]).to(DEVICE)

    prior_t = torch.tensor(prior, dtype=torch.float32, device=DEVICE)

    best_val = -1.0
    best_state = None
    history = []
    patience = 20

    tr_idx = tr.copy()
    rng = np.random.RandomState(SEED)

    for ep in range(epochs):
        model.train()
        rng.shuffle(tr_idx)
        losses = []
        reg_losses = []
        for i in range(0, len(tr_idx), bs):
            b = tr_idx[i:i + bs]
            xb = torch.from_numpy(X_bands[b]).to(DEVICE)
            xe = torch.from_numpy(X_extra[b]).to(DEVICE)
            yb = torch.from_numpy(y[b]).long().to(DEVICE)
            logits, attn = model(xb, xe)
            ce = F.cross_entropy(logits, yb, weight=cls_w)
            # KL between softmax(band_logits)=attn and prior
            log_attn = torch.log(attn + 1e-8)
            reg = F.kl_div(log_attn, prior_t, reduction='sum')
            loss = ce + lambda_c * reg
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(ce.item())
            reg_losses.append(reg.item())

        # validation
        model.eval()
        with torch.no_grad():
            va_x = torch.from_numpy(X_bands[va]).to(DEVICE)
            va_e = torch.from_numpy(X_extra[va]).to(DEVICE)
            pr, _ = model(va_x, va_e)
            val_f1 = f1_score(y[va], pr.argmax(1).cpu().numpy(),
                              average='macro', zero_division=0)
            te_x = torch.from_numpy(X_bands[te]).to(DEVICE)
            te_e = torch.from_numpy(X_extra[te]).to(DEVICE)
            pr_te, attn_final = model(te_x, te_e)
            te_pred = pr_te.argmax(1).cpu().numpy()
            test_f1 = f1_score(y[te], te_pred, average='macro',
                               zero_division=0)
            test_acc = accuracy_score(y[te], te_pred)

        history.append({'epoch': ep,
                        'ce': float(np.mean(losses)),
                        'reg': float(np.mean(reg_losses)),
                        'val_f1': float(val_f1),
                        'test_f1': float(test_f1),
                        'test_acc': float(test_acc)})

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {'model': {k: v.detach().clone()
                                    for k, v in model.state_dict().items()},
                          'val_f1': val_f1,
                          'test_f1': test_f1,
                          'test_acc': test_acc,
                          'test_pred': te_pred.tolist(),
                          'attn_final': attn_final.detach().cpu().numpy().tolist()}
            patience = 20
        else:
            patience -= 1
        if patience == 0:
            break

    return best_state, history


# ---------------------------------------------------------------
# Step 4: driver
# ---------------------------------------------------------------

def main():
    t_all = time.time()
    print(f'DEVICE={DEVICE}', flush=True)
    df = load_all()
    mask_te = df['region_lvl1'].values == HOLDOUT_REGION
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]

    # Held-out internal val from train
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(tr))
    n_val = int(0.1 * len(tr))
    va = tr[perm[:n_val]]
    tr_main = tr[perm[n_val:]]
    print(f'tr_main={len(tr_main)} va={len(va)} te={len(te)}', flush=True)

    # Build permutation prior using training data only (no holdout)
    prior, raw_importance = build_train_cv_band_prior(df, tr)

    # Features
    cols_bands = HZ_COLS
    cols_extra = ['month'] + MAX_COLS
    X_bands_raw = df[cols_bands].to_numpy(dtype=np.float32, copy=False)
    X_extra_raw = df[cols_extra].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    sb = StandardScaler().fit(X_bands_raw[tr_main])
    se = StandardScaler().fit(X_extra_raw[tr_main])
    X_bands = sb.transform(X_bands_raw).astype(np.float32)
    X_extra = se.transform(X_extra_raw).astype(np.float32)

    results = []
    for lam in [0.0, 0.1, 1.0, 10.0, 100.0]:
        for init_flag in [False, True]:
            tag = f'pga_lam{lam:g}_init{"P" if init_flag else "U"}'
            print(f'\n=== {tag} ===', flush=True)
            t0 = time.time()
            best, hist = train_pga(
                X_bands, X_extra, y, tr_main, va, te,
                prior, lambda_c=lam,
                init_with_prior=init_flag, epochs=80, bs=528)
            dt = time.time() - t0
            print(f'  {tag}: test_f1={best["test_f1"]:.4f} '
                  f'test_acc={best["test_acc"]:.4f} ({dt:.0f}s)', flush=True)
            results.append({
                'tag': tag, 'lambda_c': float(lam),
                'init_with_prior': bool(init_flag),
                'val_f1': float(best['val_f1']),
                'test_f1': float(best['test_f1']),
                'test_acc': float(best['test_acc']),
                'attn_final': best['attn_final'],
                'epochs_run': len(hist),
            })

    suffix = f'_seed{SEED}' if SEED != 42 else ''
    out_path = os.path.join(RESULTS_DIR, f'pga_results{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'seed': SEED,
                   'prior': prior.tolist(),
                   'raw_permutation_importance': raw_importance.tolist(),
                   'runs': results},
                  f, indent=2)
    print(f'\nSaved: {out_path}')
    print('\n=== PGA SUMMARY ===')
    print(f'{"config":>25s}  {"test_f1":>8s}  {"test_acc":>8s}')
    for r in results:
        print(f'{r["tag"]:>25s}  {r["test_f1"]:8.4f}  {r["test_acc"]:8.4f}')
    print(f'\nTotal {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
