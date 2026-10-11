"""Phase 2B — Shared IRM / ERM / GroupDRO training infrastructure.

Data setup (consistent with Phase 2C for direct comparison):
  - Train envs: 11 regions (광역시도), excluding region 43.
  - Test:       region 43 (only region with substantial counts of all 5 classes).
  - Val:        random 10 % of training rows across all envs (early stopping).
  - Features:
      'causal' : bands selected without seeing the holdout region
                 (band_selection.causal_feature_cols; MAX0-19 + month)
      'raw'    : all 513 HZ bands (reshaped to [N,1,513] for CNN) + tabular
                 extras (20 MAX + month) fed to the classifier head.
  - Standardisation: global StandardScaler fit on training rows only.

Training methods:
  - erm      : pooled risk minimisation (baseline).
  - irm      : IRMv1 penalty (Arjovsky et al. 2019) with warmup schedule.
  - groupdro : Sagawa et al. 2020 exponential reweighting over envs.
"""
from __future__ import annotations

import math
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score)
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(__file__))
from band_selection import causal_feature_cols
from data import HZ_COLS, LABEL_NAMES, MAX_COLS, load_all  # noqa: E402

HOLDOUT_REGION = 43
SEED = int(os.environ.get('CAUSAL_SEED', '42'))   # model-init seed (variable for multi-seed runs)
SPLIT_SEED = 42                                    # val-split seed (fixed for reproducibility)
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def set_seed(seed: int | None = None):
    if seed is None:
        seed = SEED
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def bands_in_range(lo: int, hi: int) -> List[str]:
    return [f'{f}HZ' for f in range(0, 5130, 10) if lo <= f < hi]


@dataclass
class Splits:
    X_train: np.ndarray
    y_train: np.ndarray
    env_train: np.ndarray        # 0..n_envs-1 index into env_ids
    X_val: np.ndarray
    y_val: np.ndarray
    env_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    env_ids: List[int]           # region codes for each env index
    feature_cols: List[str]
    raw_HZ: np.ndarray | None = None  # optional: [N, 513] unscaled HZ for CNN
    raw_HZ_train: np.ndarray | None = None
    raw_HZ_val: np.ndarray | None = None
    raw_HZ_test: np.ndarray | None = None


def make_splits(mode: str = 'causal', val_frac: float = 0.10) -> Splits:
    """Build train/val/test splits for Phase 2B.

    mode='causal' -> leakage-free band subset for HOLDOUT_REGION (MLP).
    mode='raw'    -> 534 features plus a separate `raw_HZ_*` ndarray that is
                     the standardised 513-band spectrum reshaped ready for a
                     Conv1d input.
    """
    assert mode in ('causal', 'raw')
    df = load_all()
    set_seed(SPLIT_SEED)

    if mode == 'causal':
        cols = causal_feature_cols(HOLDOUT_REGION)
    else:
        cols = ['month'] + HZ_COLS + MAX_COLS

    mask_test = (df['region_lvl1'].values == HOLDOUT_REGION)
    train_pool = np.where(~mask_test)[0]
    test_idx = np.where(mask_test)[0]

    rng = np.random.RandomState(SPLIT_SEED)
    perm = rng.permutation(len(train_pool))
    n_val = int(len(train_pool) * val_frac)
    val_pool = train_pool[perm[:n_val]]
    tr_pool = train_pool[perm[n_val:]]

    X_all = df[cols].to_numpy(dtype=np.float32, copy=False)
    y_all = df['label'].to_numpy(dtype=np.int64)

    scaler = StandardScaler()
    scaler.fit(X_all[tr_pool])
    X_scaled = scaler.transform(X_all).astype(np.float32)

    env_ids = sorted(df.loc[tr_pool, 'region_lvl1'].unique().tolist())
    env_map = {e: i for i, e in enumerate(env_ids)}
    env_train = np.array([env_map[e] for e in df.loc[tr_pool, 'region_lvl1']],
                         dtype=np.int64)
    env_val = np.array([env_map[e] for e in df.loc[val_pool, 'region_lvl1']],
                       dtype=np.int64)
    env_test = np.full(len(test_idx), -1, dtype=np.int64)

    splits = Splits(
        X_train=X_scaled[tr_pool],
        y_train=y_all[tr_pool],
        env_train=env_train,
        X_val=X_scaled[val_pool],
        y_val=y_all[val_pool],
        env_val=env_val,
        X_test=X_scaled[test_idx],
        y_test=y_all[test_idx],
        env_ids=env_ids,
        feature_cols=cols,
    )

    if mode == 'raw':
        raw = df[HZ_COLS].to_numpy(dtype=np.float32, copy=False)
        raw_scaler = StandardScaler()
        raw_scaler.fit(raw[tr_pool])
        raw = raw_scaler.transform(raw).astype(np.float32)
        splits.raw_HZ_train = raw[tr_pool]
        splits.raw_HZ_val = raw[val_pool]
        splits.raw_HZ_test = raw[test_idx]

    return splits


# ---------- class weights ----------
def class_weights(y: np.ndarray, num_classes: int = 5) -> torch.Tensor:
    """Inverse-frequency class weights, normalised to mean=1."""
    counts = np.bincount(y, minlength=num_classes).astype(np.float32)
    counts[counts == 0] = 1
    w = counts.sum() / (num_classes * counts)
    return torch.tensor(w, dtype=torch.float32)


# ---------- training methods ----------
def erm_step(model, batch, optimizer, cls_w):
    X, y, _ = batch
    logits = model(X)
    loss = F.cross_entropy(logits, y, weight=cls_w)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.item()), 'penalty': 0.0}


def _per_env_ce(logits, y, env, n_envs, cls_w):
    env_losses = {}
    for e in range(n_envs):
        mask = (env == e)
        n = int(mask.sum())
        if n < 2:
            continue
        env_losses[e] = F.cross_entropy(
            logits[mask], y[mask], weight=cls_w)
    return env_losses


def irm_penalty(logits_e, y_e, cls_w):
    """IRMv1 penalty: squared gradient of env loss w.r.t. a scalar 'w=1'."""
    scale = torch.tensor(1.0, device=logits_e.device, requires_grad=True)
    loss = F.cross_entropy(logits_e * scale, y_e, weight=cls_w)
    grad = torch.autograd.grad(loss, [scale], create_graph=True)[0]
    return grad ** 2


def irm_step(model, batch, optimizer, cls_w, lam: float, n_envs: int):
    X, y, env = batch
    logits = model(X)
    env_losses = {}
    penalty = 0.0
    for e in range(n_envs):
        mask = (env == e)
        n = int(mask.sum())
        if n < 2:
            continue
        loss_e = F.cross_entropy(logits[mask], y[mask], weight=cls_w)
        env_losses[e] = loss_e
        pen_e = irm_penalty(logits[mask], y[mask], cls_w)
        penalty = penalty + pen_e
    if len(env_losses) == 0:
        return {'loss': 0.0, 'penalty': 0.0}
    mean_loss = sum(env_losses.values()) / len(env_losses)
    mean_pen = penalty / len(env_losses)
    full = mean_loss + lam * mean_pen
    if lam > 1.0:
        full = full / lam
    optimizer.zero_grad()
    full.backward()
    optimizer.step()
    return {'loss': float(mean_loss.item()),
            'penalty': float(mean_pen.item())}


class GroupDROWeights:
    def __init__(self, n_envs: int, eta: float = 0.01, device=DEVICE):
        self.q = torch.ones(n_envs, device=device) / n_envs
        self.eta = eta

    def update(self, env_losses: Dict[int, torch.Tensor]):
        with torch.no_grad():
            for e, l in env_losses.items():
                self.q[e] = self.q[e] * torch.exp(self.eta * l.detach())
            self.q = self.q / self.q.sum()


def groupdro_step(model, batch, optimizer, cls_w, dro: GroupDROWeights,
                  n_envs: int):
    X, y, env = batch
    logits = model(X)
    env_losses = _per_env_ce(logits, y, env, n_envs, cls_w)
    if not env_losses:
        return {'loss': 0.0, 'penalty': 0.0}
    dro.update(env_losses)
    weighted = sum(dro.q[e] * env_losses[e] for e in env_losses)
    optimizer.zero_grad()
    weighted.backward()
    optimizer.step()
    return {'loss': float(weighted.item()), 'penalty': 0.0}


# ---------- sampling ----------
class EnvBalancedSampler:
    """Yields batches drawn equally from each environment."""

    def __init__(self, env_train: np.ndarray, batch_size: int,
                 n_envs: int, seed: int = SEED):
        self.rng = np.random.RandomState(seed)
        self.n_envs = n_envs
        self.batch_size = batch_size
        self.per_env = max(1, batch_size // n_envs)
        self.env_indices = [np.where(env_train == e)[0] for e in range(n_envs)]

    def __iter__(self):
        # one epoch ≈ enough batches to cover the largest env once
        max_env = max(len(ix) for ix in self.env_indices)
        n_batches = max(1, max_env // self.per_env)
        for _ in range(n_batches):
            chunks = []
            for ix in self.env_indices:
                if len(ix) == 0:
                    continue
                pick = self.rng.choice(ix, size=self.per_env, replace=True)
                chunks.append(pick)
            yield np.concatenate(chunks)


# ---------- evaluation ----------
@torch.no_grad()
def evaluate(model, X: np.ndarray, y: np.ndarray, batch_size=2048,
             raw_HZ: np.ndarray | None = None):
    model.eval()
    preds = []
    n = len(X)
    for i in range(0, n, batch_size):
        xb = torch.from_numpy(X[i:i + batch_size]).to(DEVICE)
        if raw_HZ is not None:
            rb = torch.from_numpy(raw_HZ[i:i + batch_size]).to(DEVICE)
            logits = model(xb, rb)
        else:
            logits = model(xb)
        preds.append(logits.argmax(dim=1).cpu().numpy())
    pred = np.concatenate(preds)
    model.train()
    return {
        'acc': float(accuracy_score(y, pred)),
        'f1_macro': float(f1_score(y, pred, average='macro',
                                   zero_division=0)),
        'f1_per_class': [
            float(f1_score(y, pred, labels=[i], average='macro',
                           zero_division=0))
            for i in range(5)
        ],
        'cm': confusion_matrix(y, pred, labels=list(range(5))).tolist(),
    }


# ---------- run loop ----------
@dataclass
class RunConfig:
    method: str                   # 'erm' | 'irm' | 'groupdro'
    epochs: int = 120
    batch_size: int = 528
    lr: float = 1e-3
    weight_decay: float = 1e-4
    irm_lambda_max: float = 1e2
    irm_warmup_epochs: int = 30
    dro_eta: float = 1e-2
    early_stop_patience: int = 20
    use_raw_HZ: bool = False


def train_and_eval(model_ctor, splits: Splits, cfg: RunConfig,
                   log_prefix: str = ''):
    """Train one model with the configured method and evaluate on val+test.

    model_ctor: callable returning a fresh nn.Module.
    Returns dict with history, best_val, test metrics, and timing.
    """
    set_seed(SEED)
    model = model_ctor().to(DEVICE)
    optimizer = torch.optim.Adam(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    cls_w = class_weights(splits.y_train).to(DEVICE)
    n_envs = len(splits.env_ids)

    sampler = EnvBalancedSampler(
        splits.env_train, cfg.batch_size, n_envs, seed=SEED)

    best_val_f1 = -1.0
    best_state = None
    patience_left = cfg.early_stop_patience
    history = []
    dro = GroupDROWeights(n_envs, cfg.dro_eta) if cfg.method == 'groupdro' else None

    t0 = time.time()
    for epoch in range(cfg.epochs):
        model.train()
        # IRM λ schedule: 1 during warmup, ramp to λ_max
        if cfg.method == 'irm':
            if epoch < cfg.irm_warmup_epochs:
                lam = 1.0
            else:
                frac = (epoch - cfg.irm_warmup_epochs) / max(
                    1, cfg.epochs - cfg.irm_warmup_epochs)
                lam = 1.0 + (cfg.irm_lambda_max - 1.0) * frac
        else:
            lam = 0.0

        ep_loss = 0.0
        ep_pen = 0.0
        n_steps = 0
        for indices in sampler:
            Xb = torch.from_numpy(splits.X_train[indices]).to(DEVICE)
            yb = torch.from_numpy(splits.y_train[indices]).to(DEVICE)
            eb = torch.from_numpy(splits.env_train[indices]).to(DEVICE)
            if cfg.use_raw_HZ:
                rb = torch.from_numpy(
                    splits.raw_HZ_train[indices]).to(DEVICE).unsqueeze(1)
                logits = model(Xb, rb)
                # Use logits directly in the loop by wrapping closures below
                batch = ((Xb, rb), yb, eb)
            else:
                batch = (Xb, yb, eb)

            if cfg.method == 'erm':
                out = _step_erm(model, batch, optimizer, cls_w, cfg.use_raw_HZ)
            elif cfg.method == 'irm':
                out = _step_irm(model, batch, optimizer, cls_w, lam, n_envs,
                                cfg.use_raw_HZ)
            elif cfg.method == 'groupdro':
                out = _step_dro(model, batch, optimizer, cls_w, dro, n_envs,
                                cfg.use_raw_HZ)
            else:
                raise ValueError(cfg.method)
            ep_loss += out['loss']
            ep_pen += out['penalty']
            n_steps += 1

        val_metrics = evaluate(
            model, splits.X_val, splits.y_val,
            raw_HZ=splits.raw_HZ_val if cfg.use_raw_HZ else None)
        test_metrics = evaluate(
            model, splits.X_test, splits.y_test,
            raw_HZ=splits.raw_HZ_test if cfg.use_raw_HZ else None)

        history.append({
            'epoch': epoch,
            'loss': ep_loss / max(1, n_steps),
            'penalty': ep_pen / max(1, n_steps),
            'lam': lam,
            'val_acc': val_metrics['acc'],
            'val_f1': val_metrics['f1_macro'],
            'test_acc': test_metrics['acc'],
            'test_f1': test_metrics['f1_macro'],
        })

        if val_metrics['f1_macro'] > best_val_f1:
            best_val_f1 = val_metrics['f1_macro']
            best_state = {k: v.detach().clone()
                          for k, v in model.state_dict().items()}
            patience_left = cfg.early_stop_patience
        else:
            patience_left -= 1

        if epoch % 5 == 0 or epoch == cfg.epochs - 1:
            print(
                f'{log_prefix}[{cfg.method}] ep={epoch:3d}  '
                f'loss={history[-1]["loss"]:.4f}  '
                f'pen={history[-1]["penalty"]:.4e}  lam={lam:.1f}  '
                f'val_f1={val_metrics["f1_macro"]:.4f}  '
                f'test_f1={test_metrics["f1_macro"]:.4f}',
                flush=True,
            )
        if patience_left <= 0:
            print(f'{log_prefix}early stopping at epoch {epoch}', flush=True)
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    final_test = evaluate(
        model, splits.X_test, splits.y_test,
        raw_HZ=splits.raw_HZ_test if cfg.use_raw_HZ else None)
    final_val = evaluate(
        model, splits.X_val, splits.y_val,
        raw_HZ=splits.raw_HZ_val if cfg.use_raw_HZ else None)
    return {
        'method': cfg.method,
        'best_val_f1': best_val_f1,
        'val': final_val,
        'test': final_test,
        'history': history,
        'time_s': time.time() - t0,
    }


# ---------- per-step helpers for raw-HZ support ----------
def _step_erm(model, batch, optimizer, cls_w, use_raw):
    (X, y, env) = batch if not use_raw else ((batch[0][0], batch[0][1]),
                                             batch[1], batch[2])
    if use_raw:
        logits = model(X[0], X[1])
    else:
        logits = model(X)
    loss = F.cross_entropy(logits, y, weight=cls_w)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return {'loss': float(loss.item()), 'penalty': 0.0}


def _step_irm(model, batch, optimizer, cls_w, lam, n_envs, use_raw):
    if use_raw:
        Xtab, Xraw = batch[0]
        logits = model(Xtab, Xraw)
    else:
        logits = model(batch[0])
    y, env = batch[1], batch[2]
    env_losses = {}
    penalty = 0.0
    for e in range(n_envs):
        mask = (env == e)
        if int(mask.sum()) < 2:
            continue
        loss_e = F.cross_entropy(logits[mask], y[mask], weight=cls_w)
        env_losses[e] = loss_e
        penalty = penalty + irm_penalty(logits[mask], y[mask], cls_w)
    if not env_losses:
        return {'loss': 0.0, 'penalty': 0.0}
    mean_loss = sum(env_losses.values()) / len(env_losses)
    mean_pen = penalty / len(env_losses)
    full = mean_loss + lam * mean_pen
    if lam > 1.0:
        full = full / lam
    optimizer.zero_grad()
    full.backward()
    optimizer.step()
    return {'loss': float(mean_loss.item()),
            'penalty': float(mean_pen.item())}


def _step_dro(model, batch, optimizer, cls_w, dro, n_envs, use_raw):
    if use_raw:
        Xtab, Xraw = batch[0]
        logits = model(Xtab, Xraw)
    else:
        logits = model(batch[0])
    y, env = batch[1], batch[2]
    env_losses = _per_env_ce(logits, y, env, n_envs, cls_w)
    if not env_losses:
        return {'loss': 0.0, 'penalty': 0.0}
    dro.update(env_losses)
    weighted = sum(dro.q[e] * env_losses[e] for e in env_losses)
    optimizer.zero_grad()
    weighted.backward()
    optimizer.step()
    return {'loss': float(weighted.item()), 'penalty': 0.0}
