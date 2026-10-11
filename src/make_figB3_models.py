"""Paper B, Figure 3 — model families on the canonical held-out region.

Horizontal bars of macro-F1 grouped by family, with seed error bars where a
model was run over five seeds. Shows that no invariance regulariser or
alignment method reaches the tree ensembles on the same inputs.

Reads results/{extra_baselines_multiseed,ensemble,hierarchical,
multiseed_summary,domain_alignment,extra_baselines}_results.json.
Writes results/figs/figB3_models.pdf.
"""
import json
import os

import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(__file__)
R = os.path.join(HERE, '..', 'results')
FIGS = os.path.join(HERE, '..', 'results', 'figs')
os.makedirs(FIGS, exist_ok=True)
L = lambda n: json.load(open(os.path.join(R, n)))

FAMILY_COLOUR = {'Trees': '#2166ac', 'Tabular DL': '#92c5de',
                 'Invariance': '#f4a582', 'Alignment': '#b2182b'}


def main():
    eb = L('extra_baselines_results.json')['runs']
    ms = L('multiseed_summary.json')
    ens = L('ensemble_results.json')['ensembles']
    hier = L('hierarchical_results.json')
    ebm = L('extra_baselines_multiseed.json')['summary']
    da = L('domain_alignment_results.json')['runs']

    def best_da(method):
        vals = [r['test']['f1_macro'] for r in da if r.get('method') == method]
        return max(vals) if vals else None

    # The four gradient-boosted rows come from one protocol (full training
    # pool, region 43 held out) so they are comparable with each other;
    # run_extra_baselines holds out 10% for validation and is not.
    op = L('operational_folds_results.json')
    f43 = next(f for f in op['folds'] if f['region'] == 43)['models']
    rows = [
        ('LightGBM', f43['lightgbm']['f1_macro'], None, 'Trees'),
        ('Tree-only soft average', f43['tree_only_soft_avg']['f1_macro'], None, 'Trees'),
        ('GBT flat', f43['xgb_flat']['f1_macro'], None, 'Trees'),
        ('GBT hierarchical', f43['xgb_hier']['f1_macro'], None, 'Trees'),
        ('CatBoost', ebm['catboost']['f1_macro_mean'],
         ebm['catboost']['f1_macro_std'], 'Trees'),
        ('Random forest', ebm['random_forest']['f1_macro_mean'],
         ebm['random_forest']['f1_macro_std'], 'Trees'),
        ('TabNet', ebm['tabnet_causal']['f1_macro_mean'],
         ebm['tabnet_causal']['f1_macro_std'], 'Tabular DL'),
        ('CNN ERM', ms['cnn']['erm']['test_f1_mean'],
         ms['cnn']['erm']['test_f1_std'], 'Invariance'),
        ('CNN GroupDRO', ms['cnn']['groupdro']['test_f1_mean'],
         ms['cnn']['groupdro']['test_f1_std'], 'Invariance'),
        ('CNN IRM', ms['cnn']['irm']['test_f1_mean'],
         ms['cnn']['irm']['test_f1_std'], 'Invariance'),
        ('MLP GroupDRO', ms['mlp']['groupdro']['test_f1_mean'],
         ms['mlp']['groupdro']['test_f1_std'], 'Invariance'),
        ('MLP IRM', ms['mlp']['irm']['test_f1_mean'],
         ms['mlp']['irm']['test_f1_std'], 'Invariance'),
        ('MLP ERM', ms['mlp']['erm']['test_f1_mean'],
         ms['mlp']['erm']['test_f1_std'], 'Invariance'),
        ('DeepCORAL', best_da('coral'), None, 'Alignment'),
        ('MMD-DG', best_da('mmd'), None, 'Alignment'),
        ('DANN', best_da('dann'), None, 'Alignment'),
    ]
    rows = [r for r in rows if r[1] is not None]
    rows.sort(key=lambda r: r[1])

    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    y = np.arange(len(rows))
    vals = [r[1] for r in rows]
    errs = [r[2] if r[2] is not None else 0 for r in rows]
    cols = [FAMILY_COLOUR[r[3]] for r in rows]

    ax.barh(y, vals, xerr=errs, capsize=2.5, color=cols, edgecolor='black',
            linewidth=0.5, error_kw=dict(lw=0.8))
    for i, (v, e) in enumerate(zip(vals, errs)):
        ax.text(v + e + 0.006, i, f'{v:.4f}', va='center', fontsize=7)

    # mark the recommendation
    idx = next(i for i, r in enumerate(rows) if r[0] == 'Tree-only soft average')
    ax.text(0.004, idx, '▸', va='center', ha='left', fontsize=11,
            color='black')

    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows], fontsize=8)
    ax.set_xlabel('macro-F1 on the held-out region', fontsize=9)
    ax.set_xlim(0, max(np.array(vals) + np.array(errs)) * 1.16)
    ax.tick_params(axis='x', labelsize=8)
    ax.grid(axis='x', linestyle=':', linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)
    for sp in ('top', 'right'):
        ax.spines[sp].set_visible(False)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor='black',
                             linewidth=0.5) for c in FAMILY_COLOUR.values()]
    ax.legend(handles, FAMILY_COLOUR.keys(), fontsize=7.5, frameon=False,
              loc='lower right', title='family', title_fontsize=7.5)

    fig.tight_layout()
    out = os.path.join(FIGS, 'figB3_models.pdf')
    fig.savefig(out, bbox_inches='tight')
    print(f'wrote {out}  ({len(rows)} models)')
    print(f'  best: {rows[-1][0]} = {rows[-1][1]:.4f}')


if __name__ == '__main__':
    main()
