"""Phase 1 / E4 — Domain-alignment baselines on the LORO Chungbuk fold.

Three multi-source domain-generalisation methods, all sharing the
57 causal-feature MLP backbone with the existing IRM/GroupDRO setup:

  - DeepCORAL: align second-order moments (covariance) across training
               environments (Sun & Saenko 2016).
  - DANN     : adversarial domain classifier with gradient reversal,
               targeting environment id (Ganin et al. 2016).
  - MMD-DG   : minimise pairwise Maximum Mean Discrepancy (Gaussian
               kernel, multi-bandwidth) between training environments.

All methods share the same MLP head (57 -> 256 -> 128 -> 5) and the
same env-balanced sampler, so the only difference between the four
methods (ERM/IRM/GroupDRO baseline + these three) is the auxiliary
loss term. This makes the comparison clean.

Compute: ~30 min total on 1 GPU (3 methods x ~10 min each).
"""
import json
import os
import sys
import time
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

sys.path.insert(0, os.path.dirname(__file__))
from irm_core import (
    DEVICE, EnvBalancedSampler, GroupDROWeights, RunConfig,
    SEED, SPLIT_SEED, _per_env_ce, class_weights, evaluate,
    make_splits, set_seed,
)

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')


# -------------------------------------------------------------
# MLP shared across methods (same as run_irm_mlp.py for fair comparison)
# -------------------------------------------------------------
class MLPFeat(nn.Module):
    """Returns (features, logits)."""

    def __init__(self, n_in: int, n_classes: int = 5,
                 hidden=(256, 128), dropout: float = 0.2):
        super().__init__()
        layers = []
        prev = n_in
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(),
                       nn.Dropout(dropout)]
            prev = h
        self.feat = nn.Sequential(*layers)
        self.cls = nn.Linear(prev, n_classes)

    def forward(self, x, return_features=False):
        h = self.feat(x)
        logits = self.cls(h)
        if return_features:
            return logits, h
        return logits


# -------------------------------------------------------------
# Loss terms
# -------------------------------------------------------------
def coral_loss_pair(h_a, h_b):
    """Frobenius distance between covariance matrices of two feature sets."""
    if h_a.size(0) < 2 or h_b.size(0) < 2:
        return torch.zeros((), device=h_a.device)
    h_a = h_a - h_a.mean(0, keepdim=True)
    h_b = h_b - h_b.mean(0, keepdim=True)
    cov_a = h_a.t() @ h_a / max(1, h_a.size(0) - 1)
    cov_b = h_b.t() @ h_b / max(1, h_b.size(0) - 1)
    return ((cov_a - cov_b) ** 2).sum() / (4 * cov_a.size(0) ** 2)


def coral_loss_multi(features_per_env):
    pairs = list(features_per_env.items())
    if len(pairs) < 2:
        return torch.zeros((), device=pairs[0][1].device)
    losses = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            losses.append(coral_loss_pair(pairs[i][1], pairs[j][1]))
    if not losses:
        return torch.zeros((), device=pairs[0][1].device)
    return torch.stack(losses).mean()


def gaussian_kernel(x, y, sigmas=(1.0, 2.0, 4.0, 8.0)):
    xx = (x * x).sum(1, keepdim=True)
    yy = (y * y).sum(1, keepdim=True)
    xy = x @ y.t()
    d2 = xx + yy.t() - 2 * xy
    k = torch.zeros_like(xy)
    for s in sigmas:
        k = k + torch.exp(-d2 / (2.0 * s * s))
    return k / len(sigmas)


def mmd_loss_pair(h_a, h_b):
    if h_a.size(0) < 2 or h_b.size(0) < 2:
        return torch.zeros((), device=h_a.device)
    k_aa = gaussian_kernel(h_a, h_a)
    k_bb = gaussian_kernel(h_b, h_b)
    k_ab = gaussian_kernel(h_a, h_b)
    return k_aa.mean() + k_bb.mean() - 2 * k_ab.mean()


def mmd_loss_multi(features_per_env):
    pairs = list(features_per_env.items())
    if len(pairs) < 2:
        return torch.zeros((), device=pairs[0][1].device)
    losses = []
    for i in range(len(pairs)):
        for j in range(i + 1, len(pairs)):
            losses.append(mmd_loss_pair(pairs[i][1], pairs[j][1]))
    return torch.stack(losses).mean()


# DANN gradient reversal layer
class GradReverse(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad):
        return -ctx.alpha * grad, None


def grad_reverse(x, alpha=1.0):
    return GradReverse.apply(x, alpha)


# -------------------------------------------------------------
# Training methods
# -------------------------------------------------------------
@dataclass
class DACfg:
    method: str   # 'coral' | 'dann' | 'mmd'
    # Budget aligned with run_irm_mlp.py (epochs=120, patience=25) so the
    # only difference between {ERM, IRM, GroupDRO, CORAL, DANN, MMD} is the
    # auxiliary loss term, per the methodology section's fairness claim.
    epochs: int = 120
    batch_size: int = 528
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 25
    align_lambda: float = 1.0   # weight on alignment loss
    dann_alpha_max: float = 1.0


def features_per_env(h, env, n_envs):
    out = {}
    for e in range(n_envs):
        mask = (env == e)
        if int(mask.sum()) >= 2:
            out[e] = h[mask]
    return out


def train_da(splits, cfg: DACfg, log_prefix=''):
    set_seed(SEED)
    n_in = splits.X_train.shape[1]
    n_envs = len(splits.env_ids)
    model = MLPFeat(n_in=n_in).to(DEVICE)

    # Domain head for DANN
    if cfg.method == 'dann':
        # 128 = last hidden of MLPFeat
        domain_head = nn.Sequential(
            nn.Linear(128, 128), nn.ReLU(), nn.Dropout(0.2),
            nn.Linear(128, n_envs),
        ).to(DEVICE)
        params = list(model.parameters()) + list(domain_head.parameters())
    else:
        domain_head = None
        params = list(model.parameters())

    opt = torch.optim.Adam(params, lr=cfg.lr, weight_decay=cfg.weight_decay)
    cls_w = class_weights(splits.y_train).to(DEVICE)

    sampler = EnvBalancedSampler(splits.env_train, cfg.batch_size, n_envs,
                                 seed=SEED)

    best_val_f1 = -1.0
    best_state = None
    patience = cfg.early_stop_patience
    history = []
    t0 = time.time()

    for epoch in range(cfg.epochs):
        model.train()
        if domain_head is not None:
            domain_head.train()

        # DANN alpha schedule (Ganin's 2/(1+exp(-10p)) - 1)
        p = epoch / max(1, cfg.epochs)
        dann_alpha = (2.0 / (1.0 + np.exp(-10.0 * p)) - 1.0) * cfg.dann_alpha_max

        ep_ce = 0.0
        ep_align = 0.0
        n_steps = 0

        for indices in sampler:
            Xb = torch.from_numpy(splits.X_train[indices]).to(DEVICE)
            yb = torch.from_numpy(splits.y_train[indices]).to(DEVICE)
            eb = torch.from_numpy(splits.env_train[indices]).to(DEVICE)

            logits, h = model(Xb, return_features=True)
            ce = F.cross_entropy(logits, yb, weight=cls_w)

            if cfg.method == 'coral':
                f_per_env = features_per_env(h, eb, n_envs)
                align = coral_loss_multi(f_per_env)
                total = ce + cfg.align_lambda * align
            elif cfg.method == 'mmd':
                f_per_env = features_per_env(h, eb, n_envs)
                align = mmd_loss_multi(f_per_env)
                total = ce + cfg.align_lambda * align
            elif cfg.method == 'dann':
                hr = grad_reverse(h, dann_alpha)
                d_logits = domain_head(hr)
                align = F.cross_entropy(d_logits, eb)
                total = ce + cfg.align_lambda * align
            else:
                raise ValueError(cfg.method)

            opt.zero_grad()
            total.backward()
            opt.step()

            ep_ce += float(ce.item())
            ep_align += float(align.item())
            n_steps += 1

        val = evaluate(model, splits.X_val, splits.y_val)
        test = evaluate(model, splits.X_test, splits.y_test)
        history.append({
            'epoch': epoch, 'ce': ep_ce / max(1, n_steps),
            'align': ep_align / max(1, n_steps),
            'dann_alpha': dann_alpha if cfg.method == 'dann' else 0.0,
            'val_f1': val['f1_macro'], 'test_f1': test['f1_macro'],
        })

        if val['f1_macro'] > best_val_f1:
            best_val_f1 = val['f1_macro']
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience = cfg.early_stop_patience
        else:
            patience -= 1

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(
                f'{log_prefix}[{cfg.method}] ep={epoch:3d}  '
                f'ce={history[-1]["ce"]:.4f}  '
                f'align={history[-1]["align"]:.4f}  '
                f'val_f1={val["f1_macro"]:.4f}  '
                f'test_f1={test["f1_macro"]:.4f}',
                flush=True,
            )

        if patience <= 0:
            print(f'{log_prefix}early stopping at epoch {epoch}', flush=True)
            break

    model.load_state_dict(best_state)
    final_test = evaluate(model, splits.X_test, splits.y_test)
    final_val = evaluate(model, splits.X_val, splits.y_val)
    return {
        'method': cfg.method,
        'align_lambda': cfg.align_lambda,
        'best_val_f1': best_val_f1,
        'val': final_val,
        'test': final_test,
        'history': history,
        'time_s': time.time() - t0,
    }


def main():
    print(f'DEVICE = {DEVICE}', flush=True)
    splits = make_splits(mode='causal')
    print(f'X_train={splits.X_train.shape}  '
          f'X_val={splits.X_val.shape}  X_test={splits.X_test.shape}  '
          f'n_envs={len(splits.env_ids)}', flush=True)

    runs = []
    for method, lam in [('coral', 1.0), ('mmd', 1.0), ('dann', 0.5)]:
        print(f'\n=== {method} (lambda={lam}) ===', flush=True)
        cfg = DACfg(method=method, align_lambda=lam)
        # epochs / patience inherited from DACfg defaults (120 / 25),
        # matched to run_irm_mlp.py for budget fairness.
        out = train_da(splits, cfg, log_prefix='  ')
        print(f'  DONE {method}: test_f1={out["test"]["f1_macro"]:.4f}  '
              f'test_acc={out["test"]["acc"]:.4f}  '
              f'({out["time_s"]:.0f}s)', flush=True)
        runs.append(out)

    suffix = f'_seed{SEED}' if SEED != 42 else ''
    out_path = os.path.join(RESULTS_DIR,
                            f'domain_alignment_results{suffix}.json')
    with open(out_path, 'w') as f:
        json.dump({'seed': SEED, 'runs': runs}, f, indent=2)
    print(f'\nSaved {out_path}', flush=True)

    print('\n=== DOMAIN-ALIGNMENT SUMMARY ===')
    print(f'{"method":>10s}  {"val_f1":>8s}  {"test_f1":>8s}  {"test_acc":>8s}')
    for r in runs:
        print(f'{r["method"]:>10s}  {r["val"]["f1_macro"]:8.4f}  '
              f'{r["test"]["f1_macro"]:8.4f}  {r["test"]["acc"]:8.4f}')


if __name__ == '__main__':
    main()
