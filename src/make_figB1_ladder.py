"""Paper B, Figure 1 — evaluation ladder (vector PDF).

Grouped bars of accuracy, five-class macro-F1 and leak-class macro-F1 across
row-CV, group-CV and LORO (region-weighted), for the identifier-rich and
signal-only feature sets. The point is the collapse, that identifiers buy 5.3
points at row level and nothing afterwards, and that leak typing under LORO
(regions weighted by leak recordings) sits a few points above the class-share
reference (dashed).

Reads results/matrix_results.json, results/loro_results.json and
results/leak_f1_results.json.
Writes results/figs/figB1_ladder.pdf (or figB1_ladder_alarm.pdf with FIG_TERM=alarm).
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
# The four non-normal classes are the recordings on which the device alarmed.
# FIG_TERM=alarm names them 'alarm classes' (data-mining manuscript, where
# 'leak' would collide with data leakage) and writes <name>_alarm.pdf.
TERM = os.environ.get('FIG_TERM', 'leak')
SUFFIX = '' if TERM == 'leak' else f'_{TERM}'
os.makedirs(FIGS, exist_ok=True)

TIERS = ['Row-CV', 'Group-CV\n(by installation)', 'LORO\n(by region)']
# The alarm variant adds the group-CV-by-sensor rung (run_sensor_holdout.py)
# between row-CV and group-CV by site.
WITH_SENSOR = TERM != 'leak'
if WITH_SENSOR:
    TIERS = ['Row-CV', 'Group-CV\n(by sensor)', 'Group-CV\n(by site)', 'LORO\n(by region)']


def main():
    mat = json.load(open(os.path.join(R, 'matrix_results.json')))
    loro = json.load(open(os.path.join(R, 'loro_results.json')))
    m = {(r['feat_mode'], r['split_mode']): r for r in mat}
    lo = {s['feat_mode']: s for s in loro}
    lf = json.load(open(os.path.join(R, 'leak_f1_results.json')))['ladder']

    sh = (json.load(open(os.path.join(R, 'sensor_holdout_results.json')))['ladder_group_sensor']
          if WITH_SENSOR else None)
    series = {}
    for feat, label in [('full', 'with identifiers (544)'),
                        ('signal', 'signal only (534)')]:
        series[label] = {
            'acc': [m[(feat, 'random')]['acc_mean'],
                    m[(feat, 'group_site')]['acc_mean'],
                    lo[feat]['weighted_acc']],
            'f1': [m[(feat, 'random')]['f1_mean'],
                   m[(feat, 'group_site')]['f1_mean'],
                   lo[feat]['weighted_f1']],
            'leak': [lf[f'{feat}_random']['f1_macro_leak'],
                     lf[f'{feat}_group_site']['f1_macro_leak'],
                     lf[f'{feat}_loro']['leak_rows_weighted']['k2']['leak_weighted']],
        }
        if WITH_SENSOR:
            g = sh[feat]
            f5 = float(np.mean([x['f1_macro'] for x in g['folds']]))
            for key, v in (('acc', g['acc_mean']), ('f1', f5), ('leak', g['alarm_f1_mean'])):
                series[label][key].insert(1, v)
    # LORO leak-F1 weights regions by leak recordings over the regions with at
    # least two leak classes; its class-share reference is weighted the same way.
    ref = lf['signal_loro']['leak_rows_weighted']['k2']['leak_weighted_chance']

    fig, axes = plt.subplots(1, 3, figsize=(13.5 if WITH_SENSOR else 11.5, 3.4),
                             sharey=True)
    x = np.arange(len(TIERS))
    w = 0.36
    colours = ['#b2182b', '#4393c3']

    for ax, metric, title in zip(axes, ['acc', 'f1', 'leak'],
                                 ['Accuracy', 'Macro-F1, five classes',
                                  f'Macro-F1, {TERM} classes']):
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

    last = len(TIERS) - 1
    axes[2].plot([last - 0.45, last + 0.45], [ref, ref], ls='--', lw=1, color='black')
    axes[2].annotate(f'class-share reference {ref:.3f}', xy=(last + 0.3, ref),
                     xytext=(last + 0.45, ref + 0.22), fontsize=6.5, ha='right',
                     arrowprops=dict(arrowstyle='-', lw=0.6))
    axes[0].set_ylabel('score', fontsize=9)
    axes[0].legend(fontsize=7.5, frameon=False, loc='upper right')
    fig.tight_layout()
    out = os.path.join(FIGS, 'figB1_ladder' + SUFFIX + '.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}')
    for label, d in series.items():
        print(f"  {label}: acc {['%.3f' % v for v in d['acc']]}")


if __name__ == '__main__':
    main()
