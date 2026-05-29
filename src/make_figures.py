"""Generate paper-ready figures from Phase 1 + Phase 2 result files.

Produces PNGs in results/figs/:
  - fig1_matrix.png           : 2x3 shortcut collapse bar chart (Phase 1)
  - fig2_loro_per_region.png  : LORO per-region bars (Phase 2A)
  - fig3_band_importance.png  : Frequency-band permutation importance (2C)
  - fig4_confusion_<cell>.png : Confusion heatmaps for key cells
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import LABEL_NAMES  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIG_DIR, exist_ok=True)

plt.rcParams.update({
    'figure.dpi': 110,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
    'axes.grid': True,
    'grid.alpha': 0.3,
})


def fig1_matrix():
    path = os.path.join(RESULTS_DIR, 'matrix_results.json')
    if not os.path.exists(path):
        print('skip fig1: matrix_results.json missing')
        return
    with open(path) as f:
        results = json.load(f)

    order = [
        ('full', 'paper_split'),
        ('full', 'random'),
        ('full', 'group_site'),
        ('signal', 'paper_split'),
        ('signal', 'random'),
        ('signal', 'group_site'),
    ]
    lookup = {(r['feat_mode'], r['split_mode']): r for r in results}
    labels = [f'{f}\n{s}' for f, s in order]
    accs = [lookup[k]['acc_mean'] for k in order]
    accs_std = [lookup[k]['acc_std'] for k in order]
    f1s = [lookup[k]['f1_mean'] for k in order]
    f1s_std = [lookup[k]['f1_std'] for k in order]

    x = np.arange(len(order))
    w = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    bars_acc = ax.bar(x - w / 2, accs, w, yerr=accs_std, label='Accuracy',
                      color='#2E86AB', capsize=3)
    bars_f1 = ax.bar(x + w / 2, f1s, w, yerr=f1s_std, label='Macro-F1',
                     color='#E63946', capsize=3)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Score')
    ax.set_ylim(0, 1.05)
    ax.set_title('Phase 1: Shortcut Collapse under Group-Site OOD Evaluation')
    ax.axhline(0.20, color='gray', linestyle='--', alpha=0.4,
               label='Random baseline (0.20)')
    ax.legend(loc='lower left')
    for bars, vals in [(bars_acc, accs), (bars_f1, f1s)]:
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02,
                    f'{v:.3f}', ha='center', fontsize=12)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig1_matrix.pdf')
    plt.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def fig2_loro():
    path = os.path.join(RESULTS_DIR, 'loro_results.json')
    if not os.path.exists(path):
        print('skip fig2: loro_results.json missing')
        return
    with open(path) as f:
        results = json.load(f)

    regions = sorted(set(r['region'] for r in results[0]['per_region']))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    colors = {'full': '#2E86AB', 'signal': '#E63946'}
    for ax_idx, metric in enumerate(['acc', 'f1_macro']):
        ax = axes[ax_idx]
        for r in results:
            per_r = {pr['region']: pr for pr in r['per_region']}
            vals = [per_r[reg][metric] for reg in regions]
            sizes = [per_r[reg]['n_test'] for reg in regions]
            ax.plot(regions, vals, 'o-',
                    label=f'{r["feat_mode"]} (macro={r["macro_" + metric.replace("f1_macro", "f1")]:.3f})',
                    color=colors[r['feat_mode']], markersize=8)
        ax.set_ylabel(metric)
        ax.set_ylim(0, 1.05)
        ax.legend()
        ax.set_title(f'Leave-One-Region-Out {metric}')
    axes[1].set_xlabel('Region code (admin lvl 1)')
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig2_loro_per_region.pdf')
    plt.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def fig3_bands():
    path = os.path.join(RESULTS_DIR, 'freq_analysis.json')
    if not os.path.exists(path):
        print('skip fig3: freq_analysis.json missing')
        return
    with open(path) as f:
        data = json.load(f)

    perm = data['band_perm_importance']
    names = [r['group'] for r in perm]
    f1_drops = [r['f1_macro_drop'] for r in perm]

    fig, axes = plt.subplots(2, 1, figsize=(9, 7.5))
    axes[0].bar(names, f1_drops, color='#E63946')
    axes[0].set_ylabel('Macro-F1 drop when permuted')
    axes[0].set_title('Phase 2C — Band permutation importance '
                      f'(holdout region {data["holdout_region"]})')
    axes[0].tick_params(axis='x', rotation=30, labelsize=13)

    gains = data['band_gain']
    axes[1].bar(list(gains.keys()), list(gains.values()), color='#2E86AB')
    axes[1].set_ylabel('XGBoost gain importance (summed)')
    axes[1].set_title('Phase 2C — Band gain importance')
    axes[1].tick_params(axis='x', rotation=30, labelsize=13)

    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig3_band_importance.pdf')
    plt.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def fig4_confusion():
    path = os.path.join(RESULTS_DIR, 'matrix_results.json')
    if not os.path.exists(path):
        print('skip fig4: matrix_results.json missing')
        return
    with open(path) as f:
        results = json.load(f)

    key_cells = [('full', 'random'), ('full', 'group_site'),
                 ('signal', 'group_site')]
    lookup = {(r['feat_mode'], r['split_mode']): r for r in results}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
    for ax, key in zip(axes, key_cells):
        r = lookup[key]
        cm = np.sum([np.array(f['cm']) for f in r['folds']], axis=0)
        cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
        im = ax.imshow(cm_norm, cmap='Reds', vmin=0, vmax=1)
        ax.set_xticks(range(5))
        ax.set_yticks(range(5))
        ax.set_xticklabels(LABEL_NAMES, rotation=35, ha='right')
        ax.set_yticklabels(LABEL_NAMES)
        ax.set_xlabel('Predicted')
        ax.set_ylabel('True')
        ax.set_title(f'{key[0]} / {key[1]}\nacc={r["acc_mean"]:.3f}')
        for i in range(5):
            for j in range(5):
                ax.text(j, i, f'{cm_norm[i, j]:.2f}', ha='center',
                        va='center',
                        color='white' if cm_norm[i, j] > 0.5 else 'black',
                        fontsize=12)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    path = os.path.join(FIG_DIR, 'fig4_confusion.pdf')
    plt.savefig(path)
    plt.close(fig)
    print(f'wrote {path}')


def main():
    fig1_matrix()
    fig2_loro()
    fig3_bands()
    fig4_confusion()
    print('Done.')


if __name__ == '__main__':
    main()
