"""Operating characteristics of the audit's tests on planted shortcuts (vector PDF).

(a) Step A1: rejection rate of the A1 decision (permutation p <= 0.05 and lift
    >= 0.25) against the share lambda of recordings that take their unit's
    label, for 50, 500 and 2,700 units of 18 recordings; the permutation test
    alone is dashed for 2,700 units (lambda = 0 is the null; the dotted line
    is the nominal 5 %).
(b) Test T3: rejection rate of the cluster-bootstrap boundary test on the
    planted selection-defined release (CWRU), against the fraction epsilon of
    the boundary-bin windows put back into the release.

Reads results/audit_semisynthetic_results.json.
Writes results/figs/figB5_operating.pdf.
"""
import json
import os

import matplotlib.pyplot as plt

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)


def main():
    d = json.load(open(os.path.join(R, 'audit_semisynthetic_results.json')))
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.0))
    ax = axes[0]
    styles = {'50': ('#4393c3', 'o'), '500': ('#2166ac', 's'), '2700': ('#b2182b', '^')}
    for G, rows in d['a1_simulation'].items():
        lam = [r['lambda'] for r in rows]
        rej = [r['reject_rate'] for r in rows]
        c, mk = styles[G]
        ax.plot(range(len(lam)), rej, marker=mk, ms=4, lw=1, color=c,
                label=f'{int(G):,} units')
        if G == '2700':
            ax.plot(range(len(lam)), [r['reject_rate_permutation'] for r in rows],
                    ls='--', lw=0.9, color=c, label='permutation test alone')
    ax.set_xticks(range(len(lam)))
    ax.set_xticklabels([f'{v:g}' for v in lam])
    ax.axhline(0.05, ls=':', lw=0.8, color='black')
    ax.set_xlabel('share of recordings carrying the unit label, $\\lambda$', fontsize=8)
    ax.set_ylabel('rejection rate', fontsize=8)
    ax.set_title('(a) A1 decision', fontsize=9)
    ax.legend(fontsize=7, frameon=False, loc='upper left')

    ax = axes[1]
    rows = d['t3_power']
    eps = [r['epsilon'] for r in rows]
    ax.plot(range(len(eps)), [r['reject_rate'] for r in rows],
            marker='o', ms=4, lw=1, color='#b2182b')
    ax.set_xticks(range(len(eps)))
    ax.set_xticklabels([f'{v:g}' for v in eps])
    ax.axhline(0.05, ls=':', lw=0.8, color='black')
    ax.set_xlabel('share of boundary-bin windows kept, $\\varepsilon$', fontsize=8)
    ax.set_title('(b) T3 depletion test', fontsize=9)
    for a in axes:
        a.set_ylim(-0.03, 1.05)
        a.tick_params(labelsize=7.5)
        a.grid(linestyle=':', linewidth=0.5, alpha=0.6)
        a.set_axisbelow(True)
        for sp in ('top', 'right'):
            a.spines[sp].set_visible(False)
    fig.tight_layout()
    out = os.path.join(FIGS, 'figB5_operating.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
