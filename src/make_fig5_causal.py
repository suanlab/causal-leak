"""Phase 2C follow-up figure — causal band subset comparison.

Produces results/figs/fig5_causal_bands.png showing macro-F1 and per-class F1
for the five feature-set variants on LORO region 43.
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
    'font.size': 14,
    'axes.titlesize': 15,
    'axes.labelsize': 14,
    'xtick.labelsize': 13,
    'ytick.labelsize': 13,
    'legend.fontsize': 13,
})


def main():
    with open(os.path.join(RESULTS_DIR, 'causal_bands_results.json')) as f:
        data = json.load(f)

    order = [
        'signal_all', 'causal_65_422Hz', 'low_only_0_65Hz',
        'mid_confounder_422_1470Hz', 'anti_causal_788_5120Hz',
    ]
    pretty = {
        'signal_all':                'all 513 HZ\n(534 feat)',
        'causal_65_422Hz':           'causal 65–422 Hz\n(57 feat)',
        'low_only_0_65Hz':           'low 0–65 Hz\n(28 feat)',
        'mid_confounder_422_1470Hz': 'mid 422–1470 Hz\n(125 feat)',
        'anti_causal_788_5120Hz':    'anti 788–5120 Hz\n(455 feat)',
    }
    lookup = {d['config']: d for d in data}
    labels = [pretty[k] for k in order]
    f1s = [lookup[k]['f1_macro'] for k in order]
    accs = [lookup[k]['acc'] for k in order]

    fig, axes = plt.subplots(2, 1, figsize=(9, 8.5))

    x = np.arange(len(order))
    w = 0.38
    axes[0].bar(x - w / 2, accs, w, color='#2E86AB', label='Accuracy')
    axes[0].bar(x + w / 2, f1s, w, color='#E63946', label='Macro-F1')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=13)
    axes[0].set_ylim(0, 0.55)
    axes[0].set_ylabel('Score')
    axes[0].set_title('Phase 2C — Frequency-band subset under LORO '
                      f'region {43} ({lookup["signal_all"]["config"]}: '
                      'baseline)')
    axes[0].legend(loc='upper right')
    for xi, (a, f) in enumerate(zip(accs, f1s)):
        axes[0].text(xi - w / 2, a + 0.01, f'{a:.3f}', ha='center',
                     fontsize=12)
        axes[0].text(xi + w / 2, f + 0.01, f'{f:.3f}', ha='center',
                     fontsize=12)
    axes[0].axhline(f1s[0], color='#E63946', linestyle='--', alpha=0.4)

    mat = np.array([[lookup[k]['f1_per_class'][i] for k in order]
                    for i in range(5)])
    im = axes[1].imshow(mat, cmap='RdYlGn', vmin=0, vmax=1, aspect='auto')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels([pretty[k].split('\n')[0] for k in order],
                            rotation=20, ha='right', fontsize=13)
    axes[1].set_yticks(range(5))
    axes[1].set_yticklabels(LABEL_NAMES)
    axes[1].set_title('Per-class F1')
    for i in range(5):
        for j in range(len(order)):
            axes[1].text(j, i, f'{mat[i, j]:.3f}', ha='center', va='center',
                         fontsize=12,
                         color='white' if mat[i, j] < 0.5 else 'black')
    plt.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig5_causal_bands.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
