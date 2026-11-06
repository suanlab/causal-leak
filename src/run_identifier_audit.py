"""Identifier audit: how far does the sensor roster alone go?

Uses the benchmark's own train/validation split (the split under which 99.79 %
was published) and fits no model:

  purity     share of sensors carrying a single label over their lifetime
  overlap    validation sensors that also appear in training
  lookup     sid -> majority training label; coverage and accuracy on covered
             validation rows
  + lrate    for validation rows the lookup does not cover, predict 'normal'
             when lrate < 90 (lrate is 90 for every non-normal recording).
             Conservative: every other uncovered row counts as an error.
             Filled: those rows get the most common training label among
             lrate == 90 rows instead.

Output: results/identifier_audit_results.json
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from data import load_all  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
NORMAL = 4


def main():
    df = load_all()
    tr, va = df[df['split'] == 'train'], df[df['split'] == 'val']

    n_lab = df.groupby('sid')['label'].nunique()
    tr_lab = tr.groupby('sid')['label'].nunique()
    tr_sids, va_sids = set(tr['sid']), set(va['sid'])

    majority = tr.groupby('sid')['label'].agg(lambda s: s.value_counts().idxmax())
    covered = va['sid'].isin(majority.index).to_numpy()
    pred = np.full(len(va), -1)
    pred[covered] = va.loc[covered, 'sid'].map(majority).to_numpy()
    y = va['label'].to_numpy()
    acc_covered = float((pred[covered] == y[covered]).mean())

    fallback = int(tr.loc[tr['lrate'] == 90, 'label'].value_counts().idxmax())
    lr = va['lrate'].to_numpy()
    pred_c = pred.copy()
    pred_c[~covered] = np.where(lr[~covered] < 90, NORMAL, -1)
    pred_f = pred.copy()
    pred_f[~covered] = np.where(lr[~covered] < 90, NORMAL, fallback)

    out = {
        'n_sensors': int(df['sid'].nunique()),
        'sensors_with_two_labels': int((n_lab > 1).sum()),
        'train_sensors': len(tr_sids),
        'train_sensors_single_label': int((tr_lab == 1).sum()),
        'val_sensors': len(va_sids),
        'val_sensors_in_train': len(va_sids & tr_sids),
        'val_rows': int(len(va)),
        'lookup_coverage': float(covered.mean()),
        'lookup_accuracy_on_covered': acc_covered,
        'uncovered_rows': int((~covered).sum()),
        'uncovered_lrate_below_90': int((lr[~covered] < 90).sum()),
        'lookup_plus_lrate_accuracy_conservative': float((pred_c == y).mean()),
        'lookup_plus_lrate_accuracy_filled': float((pred_f == y).mean()),
        'fallback_label': fallback,
    }
    with open(os.path.join(RES, 'identifier_audit_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2)
    for k, v in out.items():
        print(f'{k:<28} {v}')


if __name__ == '__main__':
    main()
