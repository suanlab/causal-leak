"""Paper B, Figure 3 — model families on the canonical held-out region.

Horizontal bars of leak-class macro-F1 (mean F1 over in/out/noise/other),
grouped by family, with seed error bars where a model was run over five seeds,
and the class-share reference (0.25 for four leak classes) marked. Diamonds
mark the networks re-run with early stopping on whole held-out training
regions (CAUSAL_VAL_MODE=region) instead of a random 10 % of recordings. The
five-class score is padded by the normal class, which the device's own
trigger count separates (run_device_rule.py), so it is not plotted.

Reads results/leak_f1_results.json (summarize_leak_f1.py).
Writes results/figs/figB3_models.pdf (or figB3_models_alarm.pdf with FIG_TERM=alarm).
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

FAMILY_COLOUR = {'Trees': '#2166ac', 'Tabular DL': '#92c5de',
                 'Invariance': '#f4a582', 'Alignment': '#b2182b'}
NAMES = [
    ('lightgbm', 'LightGBM', 'Trees'),
    ('tree_only_soft_avg', 'Tree-only soft average', 'Trees'),
    ('xgb_flat', 'GBT flat', 'Trees'),
    ('xgb_hier', 'GBT hierarchical', 'Trees'),
    ('catboost', 'CatBoost', 'Trees'),
    ('random_forest', 'Random forest', 'Trees'),
    ('tabnet_causal', 'TabNet', 'Tabular DL'),
    ('cnn_erm', 'CNN ERM', 'Invariance'),
    ('cnn_groupdro', 'CNN GroupDRO', 'Invariance'),
    ('cnn_irm', 'CNN IRM', 'Invariance'),
    ('mlp_groupdro', 'MLP GroupDRO', 'Invariance'),
    ('mlp_irm', 'MLP IRM', 'Invariance'),
    ('mlp_erm', 'MLP ERM', 'Invariance'),
    ('coral', 'DeepCORAL', 'Alignment'),
    ('mmd', 'MMD-DG', 'Alignment'),
    ('dann', 'DANN', 'Alignment'),
]


def main():
    c43 = json.load(open(os.path.join(R, 'leak_f1_results.json')))[
        'canonical_region43']
    ref = c43['chance']
    cm = c43['models']
    rows = [(label, cm[k]['f1_macro_leak'], cm[k].get('f1_macro_leak_std'), fam,
             cm.get(f'{k}_regionval'))
            for k, label, fam in NAMES if k in cm]
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    errs = [r[2] if r[2] is not None else 0 for r in rows]
    cols = [FAMILY_COLOUR[r[3]] for r in rows]

    ax.barh(y, vals, xerr=errs, capsize=2.5, color=cols, edgecolor='black',
            linewidth=0.5, error_kw=dict(lw=0.8))
    for i, (v, e, r) in enumerate(zip(vals, errs, rows)):
        right = v + e
        if r[4]:
            right = max(right, r[4]['f1_macro_leak'] + r[4].get('f1_macro_leak_std', 0))
        ax.text(right + 0.006, i, f'{v:.3f}', va='center', fontsize=7)

    for i, r in enumerate(rows):
        rv = r[4]
        if rv:
            ax.errorbar(rv['f1_macro_leak'], i, xerr=rv.get('f1_macro_leak_std', 0),
                        fmt='D', ms=4.5, color='white', mec='black', mew=0.8,
                        ecolor='black', elinewidth=0.6, capsize=2, zorder=3)
    ax.axvline(ref, ls='--', lw=1, color='black')
    ax.text(ref, len(rows) - 0.4, f' class-share reference {ref:.2f}',
            fontsize=7, va='bottom')

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel(f'{TERM}-class macro-F1 on the held-out region', fontsize=9)
    ax.set_xlim(0, max(np.array(vals) + np.array(errs)) * 1.16)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black',
                             linewidth=0.5) for c in FAMILY_COLOUR.values()]
    handles.append(plt.Line2D([], [], marker='D', ls='', color='white',
                              mec='black', ms=4.5))
    ax.legend(handles, list(FAMILY_COLOUR.keys()) + ['region-held-out\nearly stopping'],
              fontsize=7, frameon=False, loc='lower right', bbox_to_anchor=(1.0, 0.0), title='family',
              title_fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(FIGS, 'figB3_models' + SUFFIX + '.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}  ({len(rows)} models)')
    print(f'  best: {rows[-1][0]} = {rows[-1][1]:.4f}')


if __name__ == '__main__':
    main()
