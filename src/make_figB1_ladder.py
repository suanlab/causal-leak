"""Paper B, Figure 1 — evaluation ladder (vector PDF).

Grouped bars of accuracy and macro-F1 across row-CV, group-CV and LORO, for
the identifier-rich and signal-only feature sets. The point is the collapse,
and that identifiers buy 5.3 points at row level and nothing afterwards.

Reads results/matrix_results.json and results/loro_results.json.
Writes results/figs/figB1_ladder.pdf.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)

TIERS = ['Row-CV', 'Group-CV\n(by installation)', 'LORO\n(by region)']


def main():
    mat = json.load(open(os.path.join(R, 'matrix_results.json')))
    loro = json.load(open(os.path.join(R, 'loro_results.json')))
    m = {(r['feat_mode'], r['split_mode']): r for r in mat}
    lo = {s['feat_mode']: s for s in loro}

    series = {}
    for feat, label in [('full', 'with identifiers (544)'),
                        ('signal', 'signal only (534)')]:
        series[label] = {
            'acc': [m[(feat, 'random')]['acc_mean'],
                    m[(feat, 'group_site')]['acc_mean'],
                    lo[feat]['weighted_acc']],
            'f1': [m[(feat, 'random')]['f1_mean'],
                   m[(feat, 'group_site')]['f1_mean'],
                   lo[feat]['macro_f1']],
        }

    fig, axes = plt.subplots(1, 2, figsize=(9.0, 3.4), sharey=True)
    x = np.arange(len(TIERS))
    w = 0.36
    colours = ['#b2182b', '#4393c3']

    for ax, metric, title in zip(axes, ['acc', 'f1'],
                                 ['Accuracy', 'Macro-F1']):
        for j, (label, d) in enumerate(series.items()):
            ax.bar(x + (j - 0.5) * w, d[metric], w, label=label,
                   color=colours[j], edgecolor='black', linewidth=0.5)
            for xi, v in zip(x + (j - 0.5) * w, d[metric]):
                ax.text(xi, v + 0.015, f'{v:.3f}', ha='center', fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels(TIERS, fontsize=8)
        ax.set_title(title, fontsize=10)
        ax.set_ylim(0, 1.08)
        ax.tick_params(axis='y', labelsize=8)
        ax.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
        ax.set_axisbelow(True)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    axes[0].set_ylabel('score', fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIGS, 'figB1_ladder.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')
    for label, d in series.items():
        print(f"  {label}: acc {['%.3f' % v for v in d['acc']]}")


if __name__ == '__main__':
    main()
