"""Operating characteristics of the audit's tests on planted shortcuts (vector PDF).

(a) Step A1: rejection rate of the A1 decision (permutation p <= 0.05 and lift
    >= 0.25) against the share lambda of recordings that take their unit's
    label, for 50, 500 and 2,700 units of 18 recordings; the permutation test
    alone is dashed for 2,700 units (lambda = 0 is the null; the dotted line
    is the nominal 5 %).
(b) T3: pass rate of the support diagnostic at the boundary bin against the
    share epsilon of boundary-bin recordings kept, in simulated releases of
    40 to 2,710 clusters (epsilon = 1 is no selection, so its pass rate is the
    false-pass rate) and on the planted CWRU release (40 files, dashed).

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
    sim = d['t3_cluster_sim']
    shades = ['#9ecae1', '#6baed6', '#3182bd', '#08519c', '#08306b']
    eps = [r['epsilon_kept'] for r in next(iter(sim.values()))]
    xs = range(len(eps))
    for (G, rows), c in zip(sim.items(), shades):
        ax.plot(xs, [r['reject_rate'] for r in rows], marker='o', ms=3.5, lw=1,
                color=c, label=f'{int(G):,} clusters')
    cw = {r['epsilon']: r['reject_rate'] for r in d['t3_power']}
    cx = [i for i, e in enumerate(eps) if e in cw]
    ax.plot(cx, [cw[eps[i]] for i in cx], marker='s', ms=3.5, lw=1, ls='--',
            color='#b2182b', label='CWRU (40 files)')
    ax.set_xticks(list(xs))
    ax.set_xticklabels([f'{e:g}' for e in eps])
    ax.axhline(0.05, ls=':', lw=0.8, color='black')
    ax.set_xlabel('share of boundary-bin recordings kept, $\\varepsilon$', fontsize=8)
    ax.set_title('(b) T3 support diagnostic', fontsize=9)
    ax.legend(fontsize=6.5, frameon=False, loc='lower left')
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
