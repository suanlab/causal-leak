"""Data loading + feature set selection for the shortcut-learning 2x2 experiment."""
import os
import numpy as np
import pandas as pd

# Data root. Override via $CAUSAL_LEAK_DATA_DIR to point at any AI Hub
# vibration-dataset layout that contains:
#   <data_dir>/train_with_site_name_4.csv
#   <data_dir>/val_with_site_name_4.csv
DATA_DIR = os.environ.get(
    'CAUSAL_LEAK_DATA_DIR',
    os.path.join(os.path.dirname(__file__), '..', '..', 'kcc-water', 'Dataset'),
)

LABEL_MAP = {'in': 0, 'out': 1, 'noise': 2, 'other': 3, 'normal': 4}
LABEL_NAMES = ['in', 'out', 'noise', 'other', 'normal']

HZ_COLS = [f'{i}HZ' for i in range(0, 5130, 10)]   # 513 bands
MAX_COLS = [f'MAX{i}' for i in range(20)]          # 20

REGION_RENAME = {
    'site_광역시도n': 'region_lvl1',
    'site_시군구n':   'region_lvl2',
    'site_읍면동n':   'region_lvl3',
    'site_통리n':     'region_lvl4',
}
REGION_COLS = list(REGION_RENAME.values())


def load_all() -> pd.DataFrame:
    """Load train + val merged into one dataframe with derived columns."""
    tr = pd.read_csv(os.path.join(DATA_DIR, 'train_with_site_name_4.csv'))
    va = pd.read_csv(os.path.join(DATA_DIR, 'val_with_site_name_4.csv'))
    tr = pd.concat([tr, pd.Series('train', index=tr.index, name='split')],
                   axis=1).copy()
    va = pd.concat([va, pd.Series('val', index=va.index, name='split')],
                   axis=1).copy()
    df = pd.concat([tr, va], ignore_index=True).rename(
        columns=REGION_RENAME).copy()

    derived = pd.DataFrame({
        'site_int': df['site'].astype(str).str[2:].astype(np.int64),
        'sid_int': df['sid'].astype(str).str[2:].astype(np.int64),
        'ldate_int': df['ldate'].astype(np.int64),
        'year': (df['ldate'] // 10000).astype(np.int32),
        'month': ((df['ldate'] // 100) % 100).astype(np.int32),
        'label': df['leaktype'].map(LABEL_MAP).astype(np.int32),
    })
    df = pd.concat([df, derived], axis=1)

    for c in REGION_COLS:
        if c in df.columns:
            df[c] = df[c].fillna(-1).astype(np.int32)
    return df


def get_xy(df: pd.DataFrame, mode: str):
    """Return (X, y) for the chosen feature set.

    full   -> identifier-rich set that mirrors the shortcut-friendly setup
              in the original KCC notebooks (site, sid, lrate, llevel, regional
              codes, ldate-derived, HZ, MAX).
    signal -> physical-signal-only set: HZ spectrum + MAX aggregates + month.
    """
    if mode == 'full':
        ident = ['site_int', 'sid_int', 'lrate', 'llevel',
                 'year', 'month', 'ldate_int'] + REGION_COLS
        cols = ident + HZ_COLS + MAX_COLS
    elif mode == 'signal':
        cols = ['month'] + HZ_COLS + MAX_COLS
    else:
        raise ValueError(f'unknown mode: {mode}')
    X = df[cols].to_numpy(dtype=np.float32, copy=False)
    y = df['label'].to_numpy(dtype=np.int32)
    return X, y, cols
