"""
Shared configuration and helpers for the five project notebooks
(01_spatial_canvas → 02_temporal_extraction → 03_spatial_downscale → 04_mtl_engine → 05_explainability_nds).

Every notebook starts with `from project_config import *`, so paths, constants and shared
functions are defined once and stay identical across the pipeline.
"""

import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
# Relative to the folder the notebooks run from.
RAW_SPATIAL = 'Data/raw_spatial/'        # OpenCellID export (keCellTowers.csv)
RAW_OOKLA = 'Data/raw_ookla/'            # Ookla Open Data mobile tiles for Kenya
PROCESSED = 'Data/processed/'            # intermediate datasets passed between notebooks
TABLES = 'Outputs/tables/'               # Chapter 4 tables (T4.xx_*.csv)
FIGURES = 'Outputs/figures/'             # Chapter 4 figures (F4.xx_*.png)
MODELS = 'Outputs/models/'               # fitted models, scalers and forecasts
for _d in (PROCESSED, TABLES, FIGURES, MODELS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- study design
EARTH_R = 6371008.8                      # mean Earth radius (m), for haversine distances
CLUSTER_EPS_M = 100                      # maximum within-site distance (m), complete-linkage clustering of cells (Notebook 01)
MAX_TILE_DIST_KM = 10                    # Ookla tiles further than this from any site are dropped
# Mobile Network Code -> operator; unlisted codes become 'Other' (Notebook 01).
OPERATOR_MAP = {1: 'Safaricom', 2: 'Safaricom', 3: 'Airtel', 5: 'Airtel', 7: 'Telkom Kenya', 10: 'Jamii (Faiba)'}
TECH_W = {'GSM': 0.1, 'UMTS': 1.0, 'LTE': 5.0}   # relative per-cell capacity weights

STRESS_QUANTILE = 0.80                   # stressed = top 20% of load/capacity in a quarter
LOW_TESTS = 20                           # < 20 Ookla tests in a quarter = low demand confidence
SPARSE_SITES = 5                         # < 5 observed sites = sparse infrastructure records

# Quarters are indexed 0-16 (q_idx). Targets up to 12 are used for development, 13-16 for testing.
TRAIN_MAX_Q = 12                         # targets up to q_idx 12 (Q3 2024/25) = development period
L = 4                                    # lookback window (quarters)
K_NEIGH = 5                              # spatial neighbours per sub-county
SEEDS = [0, 1, 2, 3, 4]
FOLDS = [(8, (9, 10)), (10, (11, 12))]   # rolling origin: (train up to, validation quarters)
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)   # XGBoost settings not tuned (04 and 05)


def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


def save_table(df, code, name, index=False):
    """Save a Chapter 4 table as Outputs/tables/T4.xx_name.csv and return it for display."""
    path = f"{TABLES}T{code}_{name}.csv"
    df.to_csv(path, index=index, encoding='utf-8-sig')   # BOM so Excel on Windows shows ρ, κ, – correctly
    print(f"[saved] {path}")
    return df


def save_figure(code, name, fig=None):
    """Save the current (or given) matplotlib figure as Outputs/figures/F4.xx_name.png."""
    path = f"{FIGURES}F{code}_{name}.png"
    (fig or plt.gcf()).savefig(path, dpi=200, bbox_inches='tight')
    print(f"[saved] {path}")


# ---------------------------------------------------------------- spatial helpers
_TF = None


def in_kenya(lat, lon):
    """National-border test using the IANA time-zone polygons bundled with `timezonefinder`
    (they follow OpenStreetMap national borders; Kenya is the only country in Africa/Nairobi)."""
    global _TF
    if _TF is None:
        from timezonefinder import TimezoneFinder
        _TF = TimezoneFinder()
    return np.array([_TF.timezone_at(lng=x, lat=y) == 'Africa/Nairobi' for y, x in zip(lat, lon)])


# A quadkey names a Bing Maps tile: one digit 0-3 per zoom level (+1 = east half, +2 = south half).
def quadkey_to_latlon(qk):
    """Centre of a web-Mercator tile given its quadkey (Ookla tiles are zoom 16, ~600 m)."""
    x = y = 0
    z = len(qk)                          # zoom level = number of digits
    for i, c in enumerate(qk):
        m = 1 << (z - i - 1)
        d = int(c)
        if d & 1:
            x |= m
        if d & 2:
            y |= m
    n = 2 ** z
    lon = (x + 0.5) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / n))))
    return lat, lon


# x: one value per sub-county; W: N x N spatial weights (W[i, j] > 0 if j neighbours i).
def morans_i(x, W, perms=999, seed=42):
    """Global Moran's I with a two-sided permutation p-value."""
    z = x - x.mean()
    I = (len(x) / W.sum()) * (z @ W @ z) / (z @ z)
    # Permutation test: shuffling values across sub-counties removes any spatial pattern.
    rng = np.random.default_rng(seed)
    sims = np.array([(len(x) / W.sum()) * (zp @ W @ zp) / (zp @ zp)
                     for zp in (rng.permutation(z) for _ in range(perms))])
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(I - sims.mean())) + 1) / (perms + 1)
    return I, p


def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# ---------------------------------------------------------------- neural architectures (04 and 05)
try:
    import torch
    import torch.nn as nn

    class SeqNet(nn.Module):
        """GRU/LSTM encoder; static covariates concatenated after the encoder; optional
        classification head (stress) and regression head (log load gap)."""
        def __init__(self, f_dyn, f_stat, hidden=32, cell='gru', mode='mtl'):
            super().__init__()
            self.rnn = (nn.GRU if cell == 'gru' else nn.LSTM)(f_dyn, hidden, batch_first=True)
            d = hidden + f_stat              # encoder output + static covariates
            head = lambda: nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
            self.cls = head() if mode in ('mtl', 'cls') else None
            self.reg = head() if mode in ('mtl', 'reg') else None

        def encode(self, xd, xs):
            out, _ = self.rnn(xd)
            return torch.cat([out[:, -1, :], xs], dim=1)

        def forward(self, xd, xs):
            h = self.encode(xd, xs)
            return (self.cls(h).squeeze(1) if self.cls else None,
                    self.reg(h).squeeze(1) if self.reg else None)

    class TGCN(SeqNet):
        """Temporal Graph Convolutional Network (Zhao et al., 2020): a graph convolution over the
        neighbour graph at every quarter, followed by a GRU. One quarter = one graph of all sub-counties."""
        def __init__(self, f_dyn, f_stat, A_hat, hidden=32, gc_dim=16):
            super().__init__(gc_dim + f_dyn, f_stat, hidden, 'gru', 'mtl')
            self.register_buffer('A', A_hat)
            self.gc = nn.Linear(f_dyn, gc_dim)

        def encode(self, xd, xs):
            # Graph convolution Z = ReLU(A_hat X W), applied at every quarter.
            z = torch.relu(torch.einsum('ij,jlf->ilf', self.A, self.gc(xd)))
            out, _ = self.rnn(torch.cat([z, xd], dim=-1))
            return torch.cat([out[:, -1, :], xs], dim=1)

    class StressProb(nn.Module):
        """Exposes only the classification head as a probability (used by SHAP)."""
        def __init__(self, m):
            super().__init__()
            self.m = m                       # the trained network

        def forward(self, xd, xs):
            return torch.sigmoid(self.m(xd, xs)[0]).unsqueeze(1)
except ImportError:      # notebooks 01-03 do not need PyTorch
    pass
