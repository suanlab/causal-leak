"""Lineage of the analysed release, from the AI Hub archives to the loaded table.

The analyses read two derived files, train_with_site_name_4.csv and
val_with_site_name_4.csv, produced in 2023 by notebooks kept in kcc-water/.
This script fixes their relation to the files AI Hub distributes, without
redistributing any data:

1. SHA-256 and size of the two distributed archives (Training, Validation),
   of the ten class CSVs extracted from them, and of the two derived files;
   each extracted CSV must be byte-identical to a member of its archive.
2. Row counts per split and class in the distributed CSVs and the derived
   files.
3. Row preservation: every derived row matches exactly one distributed row of
   the same split and class on (site, sid, ldate), and agrees with it on
   every distributed column (lrate, llevel, leaktype, the 513 spectrum bins,
   the 20 MAX columns); no distributed row is missing.
4. The added columns: the derived files add an address for each site
   (site_name and its four administrative levels, by name and by code). The
   four codes are checked against the digits of the site code itself
   (S-AABBBCCCDD), so the region of every recording follows from the site code
   distributed by AI Hub.

Output: results/provenance_results.json (hashes, counts, check outcomes).
"""
import glob
import hashlib
import json
import os
import sys
import zipfile

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from data import DATA_DIR, HZ_COLS, MAX_COLS  # noqa: E402

RES = os.path.join(os.path.dirname(__file__), '..', 'results')
CLASSES = ('in', 'out', 'noise', 'other', 'normal')
KEY = ['site', 'sid', 'ldate']
CODES = {'site_광역시도n': (2, 4), 'site_시군구n': (4, 7),
         'site_읍면동n': (7, 10), 'site_통리n': (10, 12)}


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def describe(path):
    return {'file': os.path.relpath(path, DATA_DIR), 'bytes': os.path.getsize(path),
            'sha256': sha256(path)}


def main():
    out = {'data_dir': 'kcc-water/Dataset (CAUSAL_LEAK_DATA_DIR)', 'archives': [],
           'distributed_csv': [], 'derived_csv': [], 'splits': {}}
    for split, folder in (('train', 'Training'), ('val', 'Validation')):
        member_hashes = set()
        for z in sorted(glob.glob(os.path.join(DATA_DIR, folder, '*.zip'))):
            out['archives'].append(describe(z))
            with zipfile.ZipFile(z) as zf:
                for info in zf.infolist():
                    if not info.is_dir():
                        member_hashes.add(hashlib.sha256(zf.read(info)).hexdigest())
        raw = []
        for c in CLASSES:
            p = os.path.join(DATA_DIR, folder, f'{c}-{folder.lower()}.csv')
            d_ = describe(p)
            d_['identical_to_archive_member'] = d_['sha256'] in member_hashes
            out['distributed_csv'].append(d_)
            raw.append(pd.read_csv(p, low_memory=False))
        raw = pd.concat(raw, ignore_index=True)
        p = os.path.join(DATA_DIR, f'{split}_with_site_name_4.csv')
        out['derived_csv'].append(describe(p))
        der = pd.read_csv(p, low_memory=False)

        cols = ['lrate', 'llevel', 'leaktype'] + HZ_COLS + MAX_COLS
        rk = raw.set_index(KEY)
        dk = der.set_index(KEY)
        rec = {
            'distributed_rows': int(len(raw)), 'derived_rows': int(len(der)),
            'distributed_by_class': raw['leaktype'].value_counts().to_dict(),
            'derived_by_class': der['leaktype'].value_counts().to_dict(),
            'distributed_key_unique': bool(rk.index.is_unique),
            'derived_key_unique': bool(dk.index.is_unique),
        }
        common = dk.index.intersection(rk.index)
        rec['derived_rows_matched'] = int(len(common))
        rec['distributed_rows_missing_from_derived'] = int(len(rk.index.difference(dk.index)))
        a = dk.loc[common, cols]
        b = rk.loc[common, cols]
        num = [c for c in cols if c != 'leaktype']
        diff = ~np.isclose(a[num].to_numpy(float), b[num].to_numpy(float), equal_nan=True)
        rec['numeric_cells_compared'] = int(diff.size)
        rec['numeric_cells_differing'] = int(diff.sum())
        rec['label_mismatches'] = int((a['leaktype'].to_numpy() != b['leaktype'].to_numpy()).sum())
        added = [c for c in der.columns if c not in raw.columns]
        rec['added_columns'] = added
        site = der['site'].astype(str)
        rec['region_codes_from_site_code'] = {
            col: bool((site.str[lo:hi].astype(int) == der[col]).all())
            for col, (lo, hi) in CODES.items()}
        rec['site_name_missing'] = int(der['site_name'].isna().sum())
        out['splits'][split] = rec
        print(f"{split}: distributed {rec['distributed_rows']}, derived {rec['derived_rows']}, "
              f"matched {rec['derived_rows_matched']}, missing {rec['distributed_rows_missing_from_derived']}, "
              f"differing cells {rec['numeric_cells_differing']}/{rec['numeric_cells_compared']}, "
              f"label mismatches {rec['label_mismatches']}, codes {rec['region_codes_from_site_code']}",
              flush=True)
    s = out['splits']
    out['extracted_csv_match_archives'] = bool(all(
        d['identical_to_archive_member'] for d in out['distributed_csv']))
    out['all_checks_pass'] = out['extracted_csv_match_archives'] and bool(all(
        r['distributed_rows'] == r['derived_rows'] == r['derived_rows_matched']
        and r['distributed_rows_missing_from_derived'] == 0
        and r['numeric_cells_differing'] == 0 and r['label_mismatches'] == 0
        and r['distributed_key_unique'] and r['derived_key_unique']
        and all(r['region_codes_from_site_code'].values()) for r in s.values()))
    out['total_rows'] = int(sum(r['derived_rows'] for r in s.values()))
    with open(os.path.join(RES, 'provenance_results.json'), 'w') as fh:
        json.dump(out, fh, indent=2, ensure_ascii=False)
    print(f"total rows {out['total_rows']}; all checks pass: {out['all_checks_pass']}")


if __name__ == '__main__':
    main()
