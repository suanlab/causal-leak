"""Phase C (Strengthening 2) — Performance improvement via augmentation
and robust losses.

Evaluates five training recipes for a 1D-CNN on the raw 513-band spectrum
(with 20 MAX + month tabular features), under the canonical LORO
region 43 holdout:

1. Baseline: class-balanced cross-entropy (matches Phase 2B CNN ERM)
2. +SpecAugment: random frequency-band masking
3. +Mixup: convex interpolation in feature space
4. +Focal loss (gamma=2): emphasises hard examples
5. All combined (ResearchNet): SpecAugment + Mixup + Focal

Then runs an MLP variant (on the 57 causal features) with the same
augmentation recipes for comparison. Finally builds an ensemble of the
best variants.
"""
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from data import HZ_COLS, MAX_COLS, load_all  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
HOLDOUT_REGION = 43
SEED = int(os.environ.get('CAUSAL_SEED', '42'))   # variable for multi-seed runs
SPLIT_SEED = 42                                    # fixed val/train split seed
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed=None):
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ---------- augmentation ----------

def spec_augment(x_bands: torch.Tensor, n_masks: int = 2,
                 max_width: int = 40, prob: float = 0.5) -> torch.Tensor:
    """Frequency-band masking (SpecAugment variant). x_bands: [B, n_bands]."""
    if torch.rand(1).item() > prob:
        return x_bands
    out = x_bands.clone()
    B, F = out.shape
    for _ in range(n_masks):
        w = int(torch.randint(1, max_width + 1, (1,)).item())
        if w >= F:
            continue
        start = int(torch.randint(0, F - w, (1,)).item())
        out[:, start:start + w] = 0.0
    return out


def mixup_batch(x1, x2, y, alpha=0.2):
    """Mixup across batch: returns (x_mix_bands, x_mix_extra, y_a, y_b, lam)."""
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(x1.size(0), device=x1.device)
    return (lam * x1 + (1 - lam) * x1[idx],
            lam * x2 + (1 - lam) * x2[idx],
            y, y[idx], lam)


def focal_loss(logits, targets, alpha=None, gamma=2.0):
    ce = F.cross_entropy(logits, targets, weight=alpha, reduction='none')
    pt = torch.exp(-ce)
    return ((1 - pt) ** gamma * ce).mean()


# ---------- models ----------

class SpectrumCNN(nn.Module):
    def __init__(self, tab_dim, n_classes=5):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64), nn.ReLU(), nn.MaxPool1d(2),
            nn.Conv1d(64, 128, kernel_size=5, padding=2),
            nn.BatchNorm1d(128), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Linear(128 + tab_dim, 128),
            nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_classes),
        )

    def forward(self, x_bands, x_tab):
        if x_bands.dim() == 2:
            x_bands = x_bands.unsqueeze(1)
        h = self.conv(x_bands).squeeze(-1)
        return self.head(torch.cat([h, x_tab], dim=1))


class MLP(nn.Module):
    def __init__(self, n_in, hidden=(256, 128), n_classes=5, dropout=0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ---------- training ----------

def class_weights(y, num_classes=5):
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


def train_model(model, X_bands, X_tab, y, tr_idx, va_idx, te_idx,
                recipe, epochs=80, bs=528, lr=1e-3, wd=1e-4,
                use_cnn=True):
    set_seed(SEED)
    model = model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=wd)
    cls_w = class_weights(y[tr_idx]).to(DEVICE)

    best_val = -1.0
    best_state = None
    patience = 15
    rng = np.random.RandomState(SEED)

    for ep in range(epochs):
        model.train()
        idx = tr_idx.copy()
        rng.shuffle(idx)
        for i in range(0, len(idx), bs):
            b = idx[i:i + bs]
            xb = torch.from_numpy(X_bands[b]).to(DEVICE)
            xt = torch.from_numpy(X_tab[b]).to(DEVICE)
            yb = torch.from_numpy(y[b]).long().to(DEVICE)

            if recipe.get('specaugment') and use_cnn:
                xb = spec_augment(xb, n_masks=2, max_width=40, prob=0.5)

            if recipe.get('mixup'):
                xb, xt, y_a, y_b, lam = mixup_batch(xb, xt, yb, alpha=0.2)
                logits = model(xb, xt) if use_cnn else model(
                    torch.cat([xb, xt], dim=1))
                if recipe.get('focal'):
                    loss = (lam * focal_loss(logits, y_a, cls_w) +
                            (1 - lam) * focal_loss(logits, y_b, cls_w))
                else:
                    loss = (lam * F.cross_entropy(logits, y_a, weight=cls_w) +
                            (1 - lam) * F.cross_entropy(logits, y_b,
                                                        weight=cls_w))
            else:
                logits = model(xb, xt) if use_cnn else model(
                    torch.cat([xb, xt], dim=1))
                if recipe.get('focal'):
                    loss = focal_loss(logits, yb, cls_w)
                else:
                    loss = F.cross_entropy(logits, yb, weight=cls_w)

            opt.zero_grad()
            loss.backward()
            opt.step()

        # eval
        model.eval()
        with torch.no_grad():
            def pred(indices):
                preds = []
                for s in range(0, len(indices), 2048):
                    bb = indices[s:s + 2048]
                    xb = torch.from_numpy(X_bands[bb]).to(DEVICE)
                    xt = torch.from_numpy(X_tab[bb]).to(DEVICE)
                    logits = model(xb, xt) if use_cnn else model(
                        torch.cat([xb, xt], dim=1))
                    preds.append(logits.argmax(1).cpu().numpy())
                return np.concatenate(preds)

            va_p = pred(va_idx)
            te_p = pred(te_idx)
            val_f1 = f1_score(y[va_idx], va_p, average='macro',
                              zero_division=0)
            test_f1 = f1_score(y[te_idx], te_p, average='macro',
                               zero_division=0)
            test_acc = accuracy_score(y[te_idx], te_p)

        if val_f1 > best_val:
            best_val = val_f1
            best_state = {
                'val_f1': float(val_f1),
                'test_f1': float(test_f1),
                'test_acc': float(test_acc),
                'test_pred': te_p.tolist(),
                'epoch': ep,
            }
            patience = 15
        else:
            patience -= 1
        if patience == 0:
            break

    return best_state


def bands_in_range(lo, hi):
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


def main():
    t_all = time.time()
    print(f'DEVICE={DEVICE}', flush=True)
    df = load_all()
    mask_te = df['region_lvl1'].values == HOLDOUT_REGION
    tr = np.where(~mask_te)[0]
    te = np.where(mask_te)[0]
    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(tr))
    n_val = int(0.1 * len(tr))
    va = tr[perm[:n_val]]
    tr_main = tr[perm[n_val:]]
    print(f'tr_main={len(tr_main)} va={len(va)} te={len(te)}', flush=True)

    cols_bands = HZ_COLS
    cols_extra = ['month'] + MAX_COLS
    X_bands_raw = df[cols_bands].to_numpy(dtype=np.float32, copy=False)
    X_extra_raw = df[cols_extra].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int64)

    sb = StandardScaler().fit(X_bands_raw[tr_main])
    se = StandardScaler().fit(X_extra_raw[tr_main])
    X_bands = sb.transform(X_bands_raw).astype(np.float32)
    X_extra = se.transform(X_extra_raw).astype(np.float32)

    recipes = [
        ('cnn_baseline', {'specaugment': False, 'mixup': False,
                          'focal': False}),
        ('cnn_specaugment', {'specaugment': True, 'mixup': False,
                             'focal': False}),
        ('cnn_mixup', {'specaugment': False, 'mixup': True,
                       'focal': False}),
        ('cnn_focal', {'specaugment': False, 'mixup': False,
                       'focal': True}),
        ('cnn_all', {'specaugment': True, 'mixup': True,
                     'focal': True}),
    ]

    results = []
    for name, recipe in recipes:
        print(f'\n=== {name} ===', flush=True)
        model = SpectrumCNN(tab_dim=X_extra.shape[1])
        t0 = time.time()
        best = train_model(model, X_bands, X_extra, y,
                           tr_main, va, te, recipe, use_cnn=True)
        dt = time.time() - t0
        best['recipe'] = recipe
        best['name'] = name
        results.append(best)
        print(f'  {name}: test_f1={best["test_f1"]:.4f} '
              f'test_acc={best["test_acc"]:.4f} ({dt:.0f}s)', flush=True)

    # Also run MLP with best recipe on causal bands
    print('\n=== MLP on causal 65-422Hz features ===', flush=True)
    causal_cols = ['month'] + bands_in_range(65, 422) + MAX_COLS
    X_causal_raw = df[causal_cols].to_numpy(dtype=np.float32, copy=False)
    scaler_c = StandardScaler().fit(X_causal_raw[tr_main])
    X_causal = scaler_c.transform(X_causal_raw).astype(np.float32)
    zero_extra = np.zeros((len(X_causal), 0), dtype=np.float32)
    mlp_recipe = {'specaugment': False, 'mixup': True, 'focal': True}
    mlp = MLP(n_in=X_causal.shape[1])
    t0 = time.time()
    best_mlp = train_model(mlp, X_causal, zero_extra, y,
                           tr_main, va, te, mlp_recipe, use_cnn=False)
    best_mlp['recipe'] = mlp_recipe
    best_mlp['name'] = 'mlp_causal_mixup_focal'
    results.append(best_mlp)
    dt = time.time() - t0
    print(f'  mlp_causal_mixup_focal: test_f1={best_mlp["test_f1"]:.4f} '
          f'test_acc={best_mlp["test_acc"]:.4f} ({dt:.0f}s)', flush=True)

    suffix = f'_seed{SEED}' if SEED != 42 else ''
    out_path = os.path.join(RESULTS_DIR,
                            f'augmentation_results{suffix}.json')
    with open(out_path, 'w') as f:
        # Strip bulky test_pred before saving
        to_save = [dict((k, v) for k, v in r.items() if k != 'test_pred')
                   for r in results]
        to_save = {'seed': SEED,
                   'runs': to_save,
                   'test_preds': {r['name']: r['test_pred'] for r in results}}
        json.dump(to_save, f, indent=2)

    print('\n=== AUGMENTATION SUMMARY ===')
    print(f'{"name":>28s}  {"test_f1":>8s}  {"test_acc":>8s}')
    for r in results:
        print(f'{r["name"]:>28s}  {r["test_f1"]:8.4f}  {r["test_acc"]:8.4f}')
    print(f'\nTotal {time.time() - t_all:.0f}s', flush=True)


if __name__ == '__main__':
    main()
