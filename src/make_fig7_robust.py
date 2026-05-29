"""Figure 7 — Robust causal band analysis (Strengthening 1).

Left: bootstrap 95% CIs per band group per region.
Right: gain vs permutation-mean vs SHAP share on the canonical
Chungbuk fold.
"""
import json
import os
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
FIG_DIR = os.path.join(RESULTS_DIR, 'figs')


def main():
    rb = json.load(open(os.path.join(RESULTS_DIR, 'robust_bands_results.json')))
    groups = ['10-121', '121-422', '422-1470', '1470-5120']
    regions = sorted(rb['per_region'].keys(), key=lambda x: int(x))
    colors = {'10-121': '#999999',
              '121-422': '#2ecc71',
              '422-1470': '#e74c3c',
              '1470-5120': '#95a5a6'}

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: bootstrap CIs per region per band
    x_positions = np.arange(len(regions))
    width = 0.2
    for i, g in enumerate(groups):
        means = [rb['per_region'][r]['groups'][g]['mean'] for r in regions]
        cil = [rb['per_region'][r]['groups'][g]['ci95_low'] for r in regions]
        cih = [rb['per_region'][r]['groups'][g]['ci95_high'] for r in regions]
        yerr_lo = np.abs([m - l for m, l in zip(means, cil)])
        yerr_hi = np.abs([h - m for h, m in zip(means, cih)])
        yerr = np.vstack([yerr_lo, yerr_hi])
        axes[0].errorbar(x_positions + (i - 1.5) * width, means,
                         yerr=yerr, fmt='o', capsize=3, markersize=6,
                         label=f'{g} Hz', color=colors[g])
    axes[0].axhline(0, color='black', linewidth=0.8)
    axes[0].set_xticks(x_positions)
    axes[0].set_xticklabels([f'Region {r}' for r in regions])
    axes[0].set_ylabel(r'Permutation $\Delta_{\mathrm{F1}}$'
                       ' (macro-F1 drop on permutation)')
    axes[0].set_title('Bootstrap 95% CIs per band, per LORO fold')
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(alpha=0.3)

    # Right: gain vs permutation mean vs SHAP share on region 43
    # Use canonical region 43 freq_analysis for gain, robust_bands for consensus
    # approximate gain shares from freq_analysis.json
    freq = json.load(open(os.path.join(RESULTS_DIR, 'freq_analysis.json')))
    gain_per_group = freq['band_gain']
    # aggregate to our 4 groups
    lower = {'10-19Hz': '10-121', '19-35Hz': '10-121',
             '35-65Hz': '10-121', '65-121Hz': '10-121',
             '121-226Hz': '121-422', '226-422Hz': '121-422',
             '422-788Hz': '422-1470', '788-1470Hz': '422-1470',
             '1470-2744Hz': '1470-5120', '2744-5120Hz': '1470-5120'}
    bucket = {g: 0.0 for g in groups}
    for k, v in gain_per_group.items():
        bucket[lower[k]] += v
    total = sum(bucket.values())
    gain_share = [bucket[g] / total for g in groups]

    # permutation mean on region 43 (the canonical)
    perm_r43 = [rb['per_region']['43']['groups'][g]['mean'] for g in groups]

    # SHAP share from robust_bands_results
    shap_share_dict = rb.get('shap', {}).get('per_group_share', {})
    shap_share = [shap_share_dict.get(g, 0) for g in groups]

    x_g = np.arange(len(groups))
    w = 0.28
    ax = axes[1]
    # We plot 2 shares (gain, SHAP) and separately permutation Δf1 on twin axis
    b1 = ax.bar(x_g - w, gain_share, w, color='#3498db', label='XGBoost gain share')
    b2 = ax.bar(x_g,     shap_share, w, color='#9b59b6', label='SHAP share')
    ax.set_ylabel('Share of importance (gain / SHAP)', color='#333333')
    ax.set_xticks(x_g)
    ax.set_xticklabels([f'{g} Hz' for g in groups])
    ax.set_title('Chungbuk fold: gain vs.\\ SHAP vs.\\ permutation Δf1')
    ax.set_ylim(0, 0.75)
    ax.legend(loc='upper left')
    for bar, v in zip(b1, gain_share):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v*100:.0f}%',
                ha='center', fontsize=8)
    for bar, v in zip(b2, shap_share):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f'{v*100:.0f}%',
                ha='center', fontsize=8)

    ax.set_ylim(0.0, 0.8)
    ax2 = ax.twinx()
    ax2.bar(x_g + w, perm_r43, w, color='#e67e22', alpha=0.9,
            label='Permutation Δf1')
    ax2.axhline(0, color='black', linewidth=0.5)
    ax2.set_ylabel('Permutation Δf1 on Chungbuk', color='#e67e22')
    ax2.tick_params(axis='y', labelcolor='#e67e22')
    ax2.legend(loc='upper right')

    plt.tight_layout()
    out = os.path.join(FIG_DIR, 'fig7_robust_bands.pdf')
    plt.savefig(out, dpi=300, bbox_inches='tight')
    plt.close(fig)
    print(f'wrote {out}')


if __name__ == '__main__':
    main()
