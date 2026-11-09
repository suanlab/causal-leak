"""Paper B, Figure 2 — the dispatch trade-off.

(a) Over the eight typing regions: share of in/out leak recordings missed
    against share of noise/other recordings that would send a crew, as the
    threshold on s = p(in) + p(out) sweeps, for each candidate.
(b) Over the four regions with normal recordings: missed leaks against false
    dispatches on normal recordings.

Curves are swept on the held-out regions (diagnostic); the dotted diagonal
in (a) is what a score unrelated to the class would give. Markers are the
protocol-valid operating points: argmax (circle) and thresholds calibrated on
the training pool for 5 % and 10 % miss (triangle, square).

Reads results/dispatch_operating_points_results.json.
Writes results/figs/figB2_dispatch.pdf (figB2_dispatch_all.pdf with FIG_DISPATCH=all).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)

MODELS = [('lightgbm', 'LightGBM', '#b2182b'),
          ('tree_only_soft_avg', 'Tree-only soft average', '#2166ac'),
          ('xgb_flat', 'GBT flat', '#4393c3'),
          ('xgb_hier', 'GBT hierarchical', '#92c5de')]
OPS = [('argmax', 'o'), ('miss05', '^'), ('miss10', 's')]


# FIG_DISPATCH=all draws the all-signal run (figB2_dispatch_all.pdf).
_ALL = os.environ.get('FIG_DISPATCH', 'nested') == 'all'
IN_NAME = ('dispatch_operating_points_all_results.json' if _ALL
           else 'dispatch_operating_points_results.json')
OUT_SUFFIX = '_all' if _ALL else ''


def main():
    d = json.load(open(os.path.join(R, IN_NAME)))
    cur, pooled = d['curves_test_fitted'], d['pooled']
    fig, axes = plt.subplots(1, 2, figsize=(10.0, 3.9))

    ax = axes[0]
    for m, label, col in MODELS:
        c = cur['all_regions'][m]
        x = np.array(c['missed']) / c['n_leak']
        y = np.array(c['nuisance_dispatch']) / c['n_nuisance']
        ax.plot(x, y, color=col, lw=1.2, label=label)
        for op, mk in OPS:
            p = pooled[op][m]['all_regions']
            ax.plot(p['missed'] / p['n_leak'],
                    p['nuisance_dispatch'] / p['n_nuisance'], mk, color=col,
                    mec='black', mew=0.5, ms=6)
    ax.plot([0, 1], [1, 0], ls=':', lw=1, color='grey')
    ax.text(0.62, 0.42, 'no discrimination', fontsize=7, color='grey',
            rotation=-38)
    ax.set_xlabel('in/out leaks not dispatched (share)', fontsize=9)
    ax.set_ylabel('noise/other recordings dispatched (share)', fontsize=9)
    ax.set_title('(a) eight typing regions', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax = axes[1]
    for m, label, col in MODELS:
        c = cur['far_regions'][m]
        x = np.array(c['missed']) / c['n_leak']
        y = np.array(c['false_dispatch'])
        ax.plot(x, y, color=col, lw=1.2)
        for op, mk in OPS:
            p = pooled[op][m]['far_regions']
            ax.plot(p['missed'] / p['n_leak'], p['false_dispatch'], mk,
                    color=col, mec='black', mew=0.5, ms=6)
    n_norm = cur['far_regions'][MODELS[0][0]]['n_normal']
    ax.set_xlabel('in/out leaks not dispatched (share)', fontsize=9)
    ax.set_ylabel(f'normal recordings dispatched (of {n_norm:,})', fontsize=9)
    ax.set_title('(b) four regions with normal recordings', fontsize=10)
    ax.set_xlim(0, 1)
    ax.set_yscale('symlog', linthresh=1)
    ax.set_ylim(0, 300)

    for a in axes:
        a.tick_params(labelsize=8)
        a.grid(linestyle=':', linewidth=0.5, alpha=0.6)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    h = [plt.Line2D([], [], marker=mk, ls='', color='grey', mec='black',
                    mew=0.5) for _, mk in OPS]
    leg1 = axes[0].legend(fontsize=7, frameon=False, loc='upper right')
    axes[0].add_artist(leg1)
    axes[1].legend(h, ['argmax', 'calibrated, 5 % miss', 'calibrated, 10 % miss'],
                   fontsize=7, frameon=False, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIGS, 'figB2_dispatch' + OUT_SUFFIX + '.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
