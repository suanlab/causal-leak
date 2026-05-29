"""Figure 12 — Multi-seed test-F1 distribution per training method.

Panel (a): boxplot of MLP {ERM, IRM, GroupDRO} test_f1 over 5 seeds.
Panel (b): boxplot of CNN {ERM, IRM, GroupDRO} test_f1 over 5 seeds.
Panel (c): boxplot of CNN augmentation recipes (baseline / SpecAugment /
           Mixup / Focal / all) over 5 seeds.
Panel (d): boxplot of PGA configurations (5 lambda × {U,P} init) over
           5 seeds.

Reads results/multiseed_summary.json and writes
results/figs/fig12_multiseed.pdf (vector).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIGS_DIR = os.path.join(RESULTS_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)


def boxbar(ax, configs_dict, title, ymin=0.20, ymax=0.45, color='#3498DB'):
    names = list(configs_dict.keys())
    data = [configs_dict[n]['test_f1_seeds']
            if 'test_f1_seeds' in configs_dict[n]
            else [configs_dict[n]['test_f1_mean']]
            for n in names]
    bp = ax.boxplot(data, patch_artist=True, widths=0.55)
    for patch in bp['boxes']:
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    for med in bp['medians']:
        med.set_color('black')
    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels([n.replace('_', '\n') for n in names],
                       fontsize=7, rotation=0)
    ax.set_ylim(ymin, ymax)
    ax.set_title(title, fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    # Add mean ± std labels just below the whisker bottom of each box so
    # they do not collide with the box body or the XGB reference line.
    for i, n in enumerate(names):
        m = configs_dict[n]['test_f1_mean']
        s = configs_dict[n]['test_f1_std']
        data_low = min(data[i]) if data[i] else m
        # Place text below the box, anchored 1-1.5% of the y-range below
        # the bottom whisker; clamp to ymin so the label is always visible.
        y_text = max(ymin + 0.005, data_low - (ymax - ymin) * 0.07)
        ax.text(i + 1, y_text, f'{m:.3f}\n±{s:.3f}',
                ha='center', va='top', fontsize=6.5,
                bbox=dict(boxstyle='round,pad=0.15',
                          facecolor='white', edgecolor='none', alpha=0.85))


def main():
    path = os.path.join(RESULTS_DIR, 'multiseed_summary.json')
    if not os.path.exists(path):
        print(f'no {path} — run aggregate_multiseed.py first')
        return
    with open(path) as f:
        data = json.load(f)

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.0))

    # MLP
    mlp = {k: v for k, v in data.get('mlp', {}).items()
           if k in ['erm', 'irm', 'groupdro']}
    boxbar(axes[0, 0], mlp,
           '(a) MLP, 57 causal features (n=5)',
           ymin=0.25, ymax=0.40, color='#2E86C1')

    # CNN
    cnn = {k: v for k, v in data.get('cnn', {}).items()
           if k in ['erm', 'irm', 'groupdro']}
    boxbar(axes[0, 1], cnn,
           '(b) CNN, 513 raw bands (n=5)',
           ymin=0.25, ymax=0.40, color='#28B463')

    # Augmentation
    aug = data.get('aug', {})
    boxbar(axes[1, 0], aug,
           '(c) CNN augmentation recipes (n=5)',
           ymin=0.20, ymax=0.45, color='#E67E22')

    # PGA
    pga = data.get('pga', {})
    boxbar(axes[1, 1], pga,
           '(d) Permutation-Guided Attention (n=5)',
           ymin=0.20, ymax=0.40, color='#8E44AD')

    # Reference line for causal XGBoost. Only annotate when 0.4273
    # actually falls inside the panel's y-range; otherwise the label is
    # clipped or overlaps neighbouring panels.
    for ax in axes.flat:
        ymin, ymax = ax.get_ylim()
        if ymin <= 0.4273 <= ymax:
            ax.axhline(0.4273, color='red', linestyle='--', linewidth=0.9,
                       alpha=0.7)
            ax.text(0.98, 0.4273 - (ymax - ymin) * 0.02,
                    'XGB causal 0.4273', fontsize=7,
                    color='red', va='top', ha='right',
                    transform=ax.get_yaxis_transform())
        else:
            # Panel does not span 0.4273; draw a labelled arrow at the
            # axis top so the reader still knows where the anchor sits.
            ax.annotate('XGB causal 0.4273 (above)',
                        xy=(0.5, ymax), xycoords=('axes fraction', 'data'),
                        xytext=(0.5, ymax - (ymax - ymin) * 0.05),
                        textcoords=('axes fraction', 'data'),
                        ha='center', va='top', fontsize=7, color='red',
                        arrowprops=dict(arrowstyle='->', color='red',
                                        lw=0.7))
        ax.set_ylabel('OOD macro-F1', fontsize=8)

    fig.tight_layout()
    out = os.path.join(FIGS_DIR, 'fig12_multiseed.pdf')
    fig.savefig(out, format='pdf', bbox_inches='tight', pad_inches=0.05)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
