"""A1 / EAAI revision — Synthetic negative-control study for the
four-probe shortcut audit.

The empirical audit on AI Hub flags an identifier shortcut. But
\\emph{does the audit protocol distinguish shortcuts from non-shortcuts}?
This script applies the same four probes to two synthetic
modifications of the dataset where ground truth is known:

  (i)  SHORTCUT-PRESERVED: original AI Hub split (positive control).
       The audit should fail (i.e.\\ the lookup should reproduce
       the published accuracy).

  (ii) IDENTIFIER-SCRAMBLED: each row's `sid` is replaced by a
       \\emph{fresh independent} sensor identifier drawn uniformly
       from a synthetic pool of $S_\\mathrm{new} = 5000$ codes,
       making sensor-label association independent of the physical
       signal. The audit should now pass (P1 purity drops to
       chance, P2 lookup drops to majority baseline).

  (iii) LABEL-PERMUTED: labels are independently permuted within each
        training sensor (so per-sensor purity is broken but sensor IDs
        still cluster within-class). This is an intermediate case.

For each setting, the four P1/P2/P3/P4 metrics are computed.
The audit is "well-calibrated" if the metrics correctly distinguish
the three regimes.

Output: results/audit_negative_control_results.json. Compute: <60s on CPU.
"""
import json
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all, LABEL_MAP  # noqa: E402

RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', 'results')
SEED = 42


def compute_p1(tr):
    """Per-sensor label purity = fraction of rows in each sensor
    that match the modal label at that sensor."""
    p = tr.groupby('sid')['leaktype'].apply(
        lambda s: s.value_counts(normalize=True).iloc[0])
    return {
        'mean': float(p.mean()),
        'median': float(p.median()),
        'p10': float(p.quantile(0.1)),
        'n_sensors': int(len(p)),
        'frac_purity_1.0': float((p == 1.0).mean()),
    }


def compute_p2_p4(tr, va):
    """P2: sid->modal-label lookup applied to val.
    P4: P2 + meta-column override (lrate==90 implies abnormal).
    Returns covered-row accuracy and coverage for each."""
    lookup = tr.groupby('sid')['leaktype'].agg(
        lambda s: s.value_counts().index[0]).to_dict()
    pred_p2 = va['sid'].map(lookup)
    covered = pred_p2.notna()
    accuracy_p2 = float((pred_p2[covered] ==
                         va.loc[covered, 'leaktype']).mean()) if covered.any() \
                                                              else 0.0
    coverage_p2 = float(covered.mean())

    # P4: override with lrate==90 rule
    pred_p4 = pred_p2.copy()
    # If lrate != 90, predict normal (covers 90% of normal samples in AI Hub)
    p4_mask = va['lrate'] != 90
    pred_p4 = pred_p4.where(~p4_mask, other='normal')
    # Where lookup is missing (uncovered) AND lrate=90, fall back to majority
    miss = pred_p4.isna()
    if miss.any():
        majority = tr['leaktype'].value_counts().index[0]
        pred_p4 = pred_p4.where(~miss, other=majority)
    accuracy_p4 = float((pred_p4 == va['leaktype']).mean())
    return {
        'p2_coverage': coverage_p2,
        'p2_covered_accuracy': accuracy_p2,
        'p4_full_accuracy': accuracy_p4,
    }


def compute_p3(tr, va):
    overlap = set(va['sid']) & set(tr['sid'])
    return {
        'val_sensors_in_train': len(overlap),
        'val_sensors_total': int(va['sid'].nunique()),
        'overlap_fraction': float(len(overlap) / max(1, va['sid'].nunique())),
    }


def four_probe_audit(tr, va):
    return {
        'P1_per_sensor_purity': compute_p1(tr),
        'P2_P4_lookup': compute_p2_p4(tr, va),
        'P3_sensor_overlap': compute_p3(tr, va),
    }


def main():
    rng = np.random.RandomState(SEED)
    df = load_all()
    tr_orig = df[df['split'] == 'train'].copy()
    va_orig = df[df['split'] == 'val'].copy()

    out = {}

    # ---- (i) original AI Hub (positive control) ----
    print('=== (i) SHORTCUT-PRESERVED (original AI Hub) ===', flush=True)
    out['shortcut_preserved'] = four_probe_audit(tr_orig, va_orig)
    for k, v in out['shortcut_preserved'].items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # ---- (ii) identifier-scrambled (negative control) ----
    print('\n=== (ii) IDENTIFIER-SCRAMBLED (each row gets independent sid) ===',
          flush=True)
    n_new = 5000
    tr_scram = tr_orig.copy()
    va_scram = va_orig.copy()
    # Assign each train row a fresh sid uniformly from a new pool
    tr_scram['sid'] = [f'SC{c:05d}'
                       for c in rng.randint(0, n_new, size=len(tr_scram))]
    va_scram['sid'] = [f'SC{c:05d}'
                       for c in rng.randint(0, n_new, size=len(va_scram))]
    out['identifier_scrambled'] = four_probe_audit(tr_scram, va_scram)
    for k, v in out['identifier_scrambled'].items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # ---- (iii) global label permutation (intermediate negative control) ----
    print('\n=== (iii) LABEL-PERMUTED globally (sensor-class binding broken) ===',
          flush=True)
    tr_perm = tr_orig.copy()
    perm_idx = rng.permutation(len(tr_perm))
    tr_perm['leaktype'] = tr_perm['leaktype'].values[perm_idx]
    # Validation labels stay unchanged (so lookup tests faithfulness)
    out['label_permuted_globally'] = four_probe_audit(tr_perm, va_orig)
    for k, v in out['label_permuted_globally'].items():
        print(f'  {k}:')
        for kk, vv in v.items():
            print(f'    {kk}: {vv}')

    # Summary table
    print('\n=== SUMMARY ===')
    print(f'{"setting":>35s}  {"P1 mean":>8s}  {"P1 frac=1":>10s}  '
          f'{"P2 acc":>8s}  {"P4 acc":>8s}  {"P3 overlap":>10s}')
    for k in ['shortcut_preserved', 'identifier_scrambled',
             'label_permuted_globally']:
        r = out[k]
        print(f'{k:>35s}  '
              f'{r["P1_per_sensor_purity"]["mean"]:8.4f}  '
              f'{r["P1_per_sensor_purity"]["frac_purity_1.0"]:10.4f}  '
              f'{r["P2_P4_lookup"]["p2_covered_accuracy"]:8.4f}  '
              f'{r["P2_P4_lookup"]["p4_full_accuracy"]:8.4f}  '
              f'{r["P3_sensor_overlap"]["overlap_fraction"]:10.4f}')

    out_path = os.path.join(RESULTS_DIR, 'audit_negative_control_results.json')
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'\nSaved {out_path}')


if __name__ == '__main__':
    t0 = time.time()
    main()
    print(f'Total: {time.time() - t0:.1f}s')
