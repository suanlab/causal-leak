"""Figure 9 — Extra baselines and domain-alignment methods on the
Chungbuk LORO fold (vector PDF).

Bars show OOD macro-F1 across:
  - tree models (XGBoost flat, hierarchical, LightGBM, CatBoost, RF)
  - tabular DL (TabNet)
  - invariance / domain-alignment (MLP ERM, IRM, GroupDRO, DeepCORAL,
    DANN, MMD-DG)
  - ensemble (tree-only soft average — deployment candidate)

Reads results/{extra_baselines,domain_alignment,ensemble,
hierarchical,cross_region,irm_mlp}_results.json and writes
results/figs/fig9_baselines.pdf.
"""
import json
import os
import sys

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def safe_load(path):
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def main():
    rows = []
    # tree models
    cr = safe_load(os.path.join(RESULTS_DIR, 'cross_region_results.json'))
    if cr is not None:
        # find the canonical Chungbuk fold (region 43) for signal_all and
        # causal regimes
        for r in cr.get('folds', []):
            if r.get('region') == 43:
                rows.append(('XGB causal',
                             r['causal']['f1_macro'],
                             'tree'))
                rows.append(('XGB signal\\_all',
                             r['signal_all']['f1_macro'],
                             'tree'))
                break
    h = safe_load(os.path.join(RESULTS_DIR, 'hierarchical_results.json'))
    if h is not None:
        rows.append(('XGB hierarchical', h.get('f1_macro', np.nan), 'tree'))
    eb = safe_load(os.path.join(RESULTS_DIR, 'extra_baselines_results.json'))
    if eb is not None:
        for n, r in eb.get('runs', {}).items():
            if 'f1_macro' in r:
                cat = 'tabnet' if n == 'tabnet' else 'tree'
                rows.append((n.replace('_', ' '), r['f1_macro'], cat))
    en = safe_load(os.path.join(RESULTS_DIR, 'ensemble_results.json'))
    if en is not None:
        rows.append(('Tree-only ensemble',
                     en.get('tree_soft_average_f1', np.nan), 'ensemble'))
    irm_m = safe_load(os.path.join(RESULTS_DIR, 'irm_mlp_results.json'))
    runs = irm_m.get('runs', irm_m) if isinstance(irm_m, dict) else irm_m
    if runs is not None:
        for r in runs:
            method = r.get('method', '?')
            rows.append((f'MLP {method}', r['test']['f1_macro'], 'invar'))
    da = safe_load(os.path.join(RESULTS_DIR, 'domain_alignment_results.json'))
    if da is not None:
        for r in da.get('runs', []):
            rows.append((r['method'].upper(),
                         r['test']['f1_macro'], 'invar'))

    if not rows:
        print('No data yet — skip figure.')
        return

    cat_color = {'tree': '#2c7fb8', 'tabnet': '#7fcdbb',
                 'invar': '#df65b0', 'ensemble': '#e7298a'}
    rows.sort(key=lambda r: -r[1])
    names = [r[0] for r in rows]
    vals = [r[1] for r in rows]
    cols = [cat_color.get(r[2], '#999999') for r in rows]

    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    bars = ax.barh(range(len(names)), vals, color=cols)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlim(0, max(0.5, max(vals) * 1.1))
    ax.set_xlabel('OOD macro-F1 on Chungbuk LORO fold')
    ax.axvline(0.4326, color='red', linestyle='--', linewidth=0.8,
               alpha=0.6, label='Deployment candidate (0.4326)')
    ax.legend(loc='lower right', fontsize=8)
    ax.grid(axis='x', alpha=0.3)
    for b, v in zip(bars, vals):
        ax.text(v + 0.005, b.get_y() + b.get_height() / 2,
                f'{v:.3f}', va='center', fontsize=7)
    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig9_baselines.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
