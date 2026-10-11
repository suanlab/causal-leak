"""Paper B, Figure 2 — the operational comparison (vector PDF).

Left: normal sensors on which each model raised at least one false alarm, out of
the 1,239 normal sensors in the four folds where a false-alarm rate is
estimable. Recordings from one sensor are not independent, so the sensor is the
unit the comparison is tested on. Right: false-alarm recordings by fold.

This is the figure the deployment recommendation rests on: the ensemble raises
a third as many false alarms as the highest-scoring single model.

Reads results/operational_folds_results.json.
Writes results/figs/figB2_falsealarms.pdf.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)

MODELS = [
    ('xgb_hier', 'GBT hierarchical', '#92c5de'),
    ('tree_only_soft_avg', 'Tree-only soft average', '#2166ac'),
    ('xgb_flat', 'GBT flat', '#f4a582'),
    ('lightgbm', 'LightGBM', '#b2182b'),
]


def wilson(c, n, z=1.96):
    if n == 0:
        return 0.0, 0.0
    p = c / n
    den = 1 + z * z / n
    ctr = (p + z * z / (2 * n)) / den
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, ctr - half), ctr + half


def main():
    d = json.load(open(os.path.join(R, 'operational_folds_results.json')))
    est = [f for f in d['folds'] if f['far_estimable']]
    n_tot = sum(f['n_normal'] for f in est)

    n_sensors = sum(len(f['normal_sensors']) for f in est)
    counts, sensors = [], []
    for key, _, _ in MODELS:
        counts.append(sum(len(f['false_alarm_sensors'][key]) for f in est))
        sensors.append(len({(f['region'], s) for f in est
                            for s in f['false_alarm_sensors'][key]}))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.4, 3.5),
                                   gridspec_kw={'width_ratios': [1, 1.25]})

    # ---- (a) aggregate rate with Wilson CI ----
    ypos = np.arange(len(MODELS))
    ax1.barh(ypos, sensors, color=[c for _, _, c in MODELS],
             edgecolor='black', linewidth=0.5)
    for i, (sn, c) in enumerate(zip(sensors, counts)):
        ax1.text(sn + 0.3, i, f'{sn} sensors ({c} recordings)', va='center',
                 fontsize=7.5)
    ax1.set_yticks(ypos)
    ax1.set_yticklabels([lab for _, lab, _ in MODELS], fontsize=8)
    ax1.invert_yaxis()
    ax1.set_xlabel(f'normal sensors with a false alarm (of {n_sensors:,})',
                   fontsize=8.5)
    ax1.set_xlim(0, max(sensors) * 1.9)
    ax1.tick_params(axis='x', labelsize=8)
    ax1.set_title('(a) sensors affected, 4 folds', fontsize=9)
    ax1.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
    ax1.set_axisbelow(True)

    # ---- (b) per-fold counts ----
    regions = [f['region'] for f in est]
    x = np.arange(len(regions))
    w = 0.2
    for j, (key, lab, col) in enumerate(MODELS):
        vals = [len(f['false_alarm_sensors'][key]) for f in est]
        ax2.bar(x + (j - 1.5) * w, vals, w, label=lab, color=col,
                edgecolor='black', linewidth=0.5)
    ax2.set_xticks(x)
    ax2.set_xticklabels([f'region {r}\n({f["n_normal"]:,} normal)'
                         for r, f in zip(regions, est)], fontsize=7.5)
    ax2.set_ylabel('false alarms (count)', fontsize=8.5)
    ax2.set_title('(b) by fold', fontsize=9)
    ax2.tick_params(axis='y', labelsize=8)
    ax2.legend(fontsize=7, frameon=False, ncol=2)
    ax2.grid(axis='y', linestyle=':', linewidth=0.5, alpha=0.6)
    ax2.set_axisbelow(True)

    for ax in (ax1, ax2):
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)

    fig.tight_layout()
    out = os.path.join(FIGS, 'figB2_falsealarms.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')
    for (k, lab, _), c, sn in zip(MODELS, counts, sensors):
        print(f'  {lab:<24}{c:>4} recordings on {sn:>3} sensors')


if __name__ == '__main__':
    main()
