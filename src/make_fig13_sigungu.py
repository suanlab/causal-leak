"""Figure 13 — 시군구 (county-level) Leave-One-County-Out OOD per-fold
delta macro-F1 between causal (57-feat) and signal_all (534-feat)
XGBoost.

Bars sorted by causal advantage; positive (causal wins) shaded blue,
negative (signal_all wins) shaded red. Width proportional to fold size.

Reads results/sigungu_ood_results.json, writes
results/figs/fig13_sigungu.pdf (vector).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def main():
    path = os.path.join(RESULTS_DIR, 'sigungu_ood_results.json')
    if not os.path.exists(path):
        print(f'no {path} — run run_sigungu_ood.py first')
        return
    with open(path) as f:
        data = json.load(f)

    folds = data['folds']
    # Sort by delta_f1
    folds = sorted(folds, key=lambda r: r['delta_f1_macro'])

    names = [r['sigungu'] for r in folds]
    deltas = [r['delta_f1_macro'] for r in folds]
    n_test = [r['n_test'] for r in folds]
    n_cls = [r['n_classes_in_test'] for r in folds]

    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    colors = ['#C0392B' if d < 0 else '#2E86C1' for d in deltas]
    bars = ax.barh(range(len(names)), deltas, color=colors, alpha=0.85)
    ax.set_yticks(range(len(names)))
    ax.set_yticklabels([f'{n}  (#cls={c}, n={t})'
                        for n, c, t in zip(names, n_cls, n_test)],
                       fontsize=11)
    ax.axvline(0, color='black', linewidth=0.5)
    ax.set_xlabel(r'$\Delta$ macro-F1 (causal $-$ signal\_all)')
    ax.set_title('LOCO 16-fold: causal-feature advantage per sigungu',
                 fontsize=12)
    ax.grid(axis='x', alpha=0.3)
    summary = data['summary']
    weighted = summary['weighted_mean_delta']
    unweighted = summary['unweighted_mean_delta']
    ax.axvline(weighted, color='#27AE60', linestyle='--', linewidth=0.9,
               label=f'weighted mean Δ={weighted:+.4f}')
    ax.axvline(unweighted, color='#7F8C8D', linestyle=':', linewidth=0.9,
               label=f'unweighted mean Δ={unweighted:+.4f}')
    ax.set_xlim(-0.25, 0.15)
    ax.legend(loc='upper left', fontsize=12)

    # Annotate bars with delta values
    for b, d in zip(bars, deltas):
        x = d + (0.005 if d >= 0 else -0.005)
        ax.text(x, b.get_y() + b.get_height() / 2, f'{d:+.3f}',
                va='center', ha='left' if d >= 0 else 'right',
                fontsize=11)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig13_sigungu.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
