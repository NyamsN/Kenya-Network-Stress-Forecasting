"""
Shared configuration and helpers for the five project notebooks
(01_spatial_canvas → 02_temporal_extraction → 03_spatial_downscale → 04_mtl_engine → 05_explainability_nds).

Every notebook starts with `from project_config import *`, so paths, constants and shared
functions are defined once and stay identical across the pipeline.
"""
# HOW THIS FILE IS USED
# - 'from project_config import *' copies every public name defined here (the libraries
#   imported below, the paths, constants, functions and network classes) into the notebook.
#   Names that start with an underscore (_d, _TF) are private and are not copied.
# - Python looks for modules in the folder the notebook runs from, so this file must sit in
#   the same folder as the five notebooks.
# - Changing a setting here changes it in every notebook, which keeps the pipeline consistent.

# Libraries used by all notebooks. Because they are imported here, the notebooks can use os,
# random, np (NumPy), pd (pandas) and plt (Matplotlib) without importing them again.
import os
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------------------------------------------------------- paths
# All paths are RELATIVE: they are read from the folder the notebook runs in (in Jupyter, the
# notebook's own folder). The project therefore runs on any computer without editing paths, as
# long as the notebooks, this file and the Data/ and Outputs/ folders sit together in one
# project folder. Each path ends in '/', so a file name can be added with '+',
# e.g. PROCESSED + 'sites_all_technologies.csv'.
RAW_SPATIAL = 'Data/raw_spatial/'        # OpenCellID export (keCellTowers.csv)
RAW_OOKLA = 'Data/raw_ookla/'            # Ookla Open Data mobile tiles for Kenya
PROCESSED = 'Data/processed/'            # intermediate datasets passed between notebooks
TABLES = 'Outputs/tables/'               # Chapter 4 tables (T4.xx_*.csv)
FIGURES = 'Outputs/figures/'             # Chapter 4 figures (F4.xx_*.png)
MODELS = 'Outputs/models/'               # fitted models, scalers and forecasts
# Create the output folders if they do not exist yet; exist_ok=True means "no error if the
# folder is already there". The raw-data folders are not created, because they must already
# contain the input files.
for _d in (PROCESSED, TABLES, FIGURES, MODELS):
    os.makedirs(_d, exist_ok=True)

# ---------------------------------------------------------------- study design
# Spatial settings (Notebooks 01 and 03).
# EARTH_R converts between metres and the angles (radians) used by the haversine distance:
# angle = distance / EARTH_R, and distance = angle x EARTH_R.
EARTH_R = 6371008.8                      # mean Earth radius (m), for haversine distances
# Two cells join the same approximate site only if they are at most 100 m apart (Table 4.2).
CLUSTER_EPS_M = 100                      # maximum within-site distance (m), complete-linkage clustering of cells (Notebook 01)
# Notebook 03 assigns each Ookla tile to its nearest site; tiles farther than this are dropped.
MAX_TILE_DIST_KM = 10                    # Ookla tiles further than this from any site are dropped
# Mobile Network Code (the 'net' column of OpenCellID) -> operator. One operator can hold
# several codes. Codes not listed become 'Other' in Notebook 01.
OPERATOR_MAP = {1: 'Safaricom', 2: 'Safaricom', 3: 'Airtel', 5: 'Airtel', 7: 'Telkom Kenya', 10: 'Jamii (Faiba)'}
# Notebook 03: capacity = 0.1 x GSM cells + 1 x UMTS cells + 5 x LTE cells, so one LTE cell
# counts as much as five UMTS cells and a GSM cell as one tenth of a UMTS cell.
TECH_W = {'GSM': 0.1, 'UMTS': 1.0, 'LTE': 5.0}   # relative per-cell capacity weights

# Stress target and data-quality flags (Notebook 03).
# In every quarter the 80th percentile of the load/capacity ratio is the threshold, and the
# sub-counties above it are labelled stressed. LOW_TESTS sets the low_confidence flag;
# SPARSE_SITES sets the sparse_infrastructure flag (also reported in Notebook 01, Step 6).
STRESS_QUANTILE = 0.80                   # stressed = top 20% of load/capacity in a quarter
LOW_TESTS = 20                           # < 20 Ookla tests in a quarter = low demand confidence
SPARSE_SITES = 5                         # < 5 observed sites = sparse infrastructure records

# Model training and evaluation (Notebooks 04 and 05). q_idx numbers the quarters 0, 1, 2, ...
# in time order (set in Notebook 02). Targets after TRAIN_MAX_Q form the test period.
TRAIN_MAX_Q = 12                         # targets up to q_idx 12 (Q3 2024/25) = development period
# The sequence models read the last L quarters of features to forecast the next quarter.
L = 4                                    # lookback window (quarters)
# The 5 nearest sub-counties (by centroid distance) are each sub-county's neighbours. They
# define the spatial weights used for Moran's I, the spatial-lag features and the T-GCN graph.
K_NEIGH = 5                              # spatial neighbours per sub-county
# Every neural model is trained once with each seed; Notebook 04 reports the mean and the
# spread across seeds (Table 4.17), so results do not hinge on one lucky initialisation.
SEEDS = [0, 1, 2, 3, 4]
# Each fold is (last training quarter, validation quarters): train up to 8 and validate on
# 9-10, then train up to 10 and validate on 11-12. Validation always lies after training in
# time, so tuning never uses information from the future.
FOLDS = [(8, (9, 10)), (10, (11, 12))]   # rolling origin: (train up to, validation quarters)
# XGBoost settings kept fixed while max_depth and n_estimators are tuned in Notebook 04.
# learning_rate: how strongly each new tree corrects the previous ones; subsample and
# colsample_bytree: the share of rows and of features drawn at random for each tree.
# Writing **XGB_FIXED inside a function call passes these entries as keyword arguments.
XGB_FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)   # XGBoost settings not tuned (04 and 05)


# Fix the random-number generators of Python, NumPy and (if installed) PyTorch so a re-run
# gives the same results. Called as set_seeds(42) at the start of Notebook 01.
def set_seeds(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    # PyTorch is optional here: without it, 'import torch' raises ImportError and the function
    # simply skips this part, so Notebooks 01-03 also work on a computer without PyTorch.
    try:
        import torch
        torch.manual_seed(seed)
    except ImportError:
        pass


# Inputs: df = the table; code = its Chapter 4 number as text (e.g. '4.01'); name = a short
# file name; index = whether to write the row labels as the first column. It returns the same
# df, so a notebook can save and show a table in one line: display(save_table(...)).
# Used by every notebook.
def save_table(df, code, name, index=False):
    """Save a Chapter 4 table as Outputs/tables/T4.xx_name.csv and return it for display."""
    path = f"{TABLES}T{code}_{name}.csv"
    df.to_csv(path, index=index, encoding='utf-8-sig')   # BOM so Excel on Windows shows ρ, κ, – correctly
    print(f"[saved] {path}")
    return df


# Saves the figure passed as 'fig' or, if none is given, the current figure (plt.gcf() = 'get
# current figure'). dpi=200 gives a sharp image for the report; bbox_inches='tight' trims
# empty margins. Call it BEFORE plt.show(), which clears the figure in Jupyter.
def save_figure(code, name, fig=None):
    """Save the current (or given) matplotlib figure as Outputs/figures/F4.xx_name.png."""
    path = f"{FIGURES}F{code}_{name}.png"
    # 'fig or plt.gcf()' uses fig if one was given, otherwise the current figure.
    (fig or plt.gcf()).savefig(path, dpi=200, bbox_inches='tight')
    print(f"[saved] {path}")


# ---------------------------------------------------------------- spatial helpers
# _TF will hold the time-zone finder used by in_kenya(). It starts empty (None) and is built on
# the first call only, because loading the time-zone polygons takes a moment; later calls
# reuse it.
_TF = None


# Inputs: sequences of latitudes and longitudes. Output: a NumPy array of True/False, one
# per point (True = inside Kenya), used as a boolean mask in Notebooks 01 (cells) and 02
# (Ookla tiles).
def in_kenya(lat, lon):
    """National-border test using the IANA time-zone polygons bundled with `timezonefinder`
    (they follow OpenStreetMap national borders; Kenya is the only country in Africa/Nairobi)."""
    # 'global' lets the function store the finder in the module-level variable _TF.
    global _TF
    if _TF is None:
        # Imported here rather than at the top, so the package is only needed when used.
        from timezonefinder import TimezoneFinder
        _TF = TimezoneFinder()
    # List comprehension: for every (latitude y, longitude x) pair from zip(), look up the
    # time zone and test whether it is 'Africa/Nairobi'. Any other zone or the sea gives False.
    return np.array([_TF.timezone_at(lng=x, lat=y) == 'Africa/Nairobi' for y, x in zip(lat, lon)])


# A quadkey is the name of a square map tile (the Bing Maps tile system used by Ookla): a
# string of digits 0-3, one per zoom level. Each digit picks one quarter of the tile named by
# the digits before it (+1 = east half, +2 = south half), so the string length is the zoom.
# Returns the tile centre (lat, lon). Used in Notebook 02 to place each Ookla tile on the map.
def quadkey_to_latlon(qk):
    """Centre of a web-Mercator tile given its quadkey (Ookla tiles are zoom 16, ~600 m)."""
    # x = tile column (counted west to east), y = tile row (north to south), built bit by bit.
    x = y = 0
    z = len(qk)                          # zoom level = number of digits
    for i, c in enumerate(qk):
        # m is the bit for this zoom level, 2 ** (z - i - 1); '<<' shifts the 1 to the left.
        m = 1 << (z - i - 1)
        d = int(c)
        # '&' tests a bit and '|=' sets one: digit bit 1 goes to the column, bit 2 to the row.
        if d & 1:
            x |= m
        if d & 2:
            y |= m
    # n tiles per side at this zoom; the +0.5 moves from the tile's corner to its centre.
    n = 2 ** z
    # Column -> longitude is a straight line from -180 to 180 degrees; row -> latitude uses the
    # inverse web-Mercator formula, because Mercator rows are not evenly spaced in latitude.
    lon = (x + 0.5) / n * 360 - 180
    lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * (y + 0.5) / n))))
    return lat, lon


# Global Moran's I measures spatial autocorrelation: whether neighbouring sub-counties have
# similar values (I above 0), opposite values (I below 0) or unrelated values (I near 0).
# Inputs: x = one value per sub-county (NumPy array); W = N x N spatial weights matrix
# (W[i, j] > 0 when j is a neighbour of i). Returns (I, p-value). Used in Notebook 04
# (Tables 4.11 and 4.19) and Notebook 05 (spatial pattern of the NDS).
def morans_i(x, W, perms=999, seed=42):
    """Global Moran's I with a two-sided permutation p-value."""
    # Deviations from the mean.
    z = x - x.mean()
    # I = (n / sum of all weights) x (z'Wz) / (z'z); '@' is matrix multiplication.
    I = (len(x) / W.sum()) * (z @ W @ z) / (z @ z)
    # Permutation test: shuffle the values across sub-counties 999 times and recompute I.
    # Shuffling destroys any spatial pattern, so these values show what I looks like by chance.
    # (rng.permutation(z) for _ in range(perms)) is a generator; '_' is an unused loop counter.
    rng = np.random.default_rng(seed)
    sims = np.array([(len(x) / W.sum()) * (zp @ W @ zp) / (zp @ zp)
                     for zp in (rng.permutation(z) for _ in range(perms))])
    # Two-sided p-value: the share of shuffled values at least as far from their mean as the
    # observed I. The +1s count the observed value itself, so p is never exactly 0 (the
    # smallest possible p is 1/1000 with 999 permutations).
    p = (np.sum(np.abs(sims - sims.mean()) >= abs(I - sims.mean())) + 1) / (perms + 1)
    return I, p


# Min-max scaling: rescales values to the range 0 to 1 (smallest -> 0, largest -> 1). The tiny
# 1e-12 avoids dividing by zero when all values are equal. Used in Notebook 05 to put the NDS
# components on a common scale.
def minmax(x):
    x = np.asarray(x, float)
    return (x - x.min()) / (x.max() - x.min() + 1e-12)


# ---------------------------------------------------------------- neural architectures (04 and 05)
# The network classes live here, not in Notebook 04, so Notebook 05 can rebuild exactly the
# same networks and load the weights trained in Notebook 04. They need PyTorch, so they sit
# inside try/except ImportError: if PyTorch is not installed (for example on an Intel Mac, see
# the README), 'import torch' fails, the classes are skipped and Notebooks 01-03 still run.
# Notebook 04 prints the source of SeqNet and TGCN, so the comments inside them appear there.
try:
    import torch
    import torch.nn as nn

    # SeqNet: GRU-MTL, LSTM-MTL and the single-task GRUs of Notebook 04 (GRU-MTL reused in 05).
    class SeqNet(nn.Module):
        """GRU/LSTM encoder; static covariates concatenated after the encoder; optional
        classification head (stress) and regression head (log load gap)."""
        # f_dyn / f_stat: number of time-varying / static features; hidden: size of the GRU or
        # LSTM memory; cell: 'gru' or 'lstm'; mode: 'mtl' = both heads, 'cls' = stress head
        # only, 'reg' = gap head only.
        def __init__(self, f_dyn, f_stat, hidden=32, cell='gru', mode='mtl'):
            super().__init__()
            # batch_first=True: the input is shaped (sub-county windows, quarters, features).
            self.rnn = (nn.GRU if cell == 'gru' else nn.LSTM)(f_dyn, hidden, batch_first=True)
            d = hidden + f_stat              # encoder output + static covariates
            # A head: d -> 32 units -> ReLU -> dropout (10% of values zeroed while training,
            # against overfitting) -> 1 output. The lambda builds a new head with its own
            # weights on each call. A head not needed in this mode is None.
            head = lambda: nn.Sequential(nn.Linear(d, 32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
            self.cls = head() if mode in ('mtl', 'cls') else None
            self.reg = head() if mode in ('mtl', 'reg') else None

        def encode(self, xd, xs):
            # xd: (batch, L quarters, f_dyn); xs: (batch, f_stat). 'out' holds the hidden state
            # after every quarter; out[:, -1, :] is the state after the last quarter, which
            # summarises the window. The static covariates are appended to it (dim=1).
            out, _ = self.rnn(xd)
            return torch.cat([out[:, -1, :], xs], dim=1)

        def forward(self, xd, xs):
            # Returns (stress logit, gap prediction); a switched-off head gives None. The
            # logit becomes a probability through a sigmoid. squeeze(1): (batch, 1) -> (batch,)
            h = self.encode(xd, xs)
            return (self.cls(h).squeeze(1) if self.cls else None,
                    self.reg(h).squeeze(1) if self.reg else None)

    # TGCN inherits the heads and forward() from SeqNet and replaces only encode().
    # A_hat: the normalised N x N adjacency matrix of the neighbour graph (from Notebook 04).
    class TGCN(SeqNet):
        """Temporal Graph Convolutional Network (Zhao et al., 2020): a graph convolution over the
        neighbour graph at every quarter, followed by a GRU. One quarter = one graph of all sub-counties."""
        def __init__(self, f_dyn, f_stat, A_hat, hidden=32, gc_dim=16):
            # GRU input = graph features (gc_dim) + raw features (f_dyn); both heads.
            super().__init__(gc_dim + f_dyn, f_stat, hidden, 'gru', 'mtl')
            # Stored with the model (saved and moved with it) but not trained.
            self.register_buffer('A', A_hat)
            # The weight matrix W of the graph convolution: f_dyn -> gc_dim features.
            self.gc = nn.Linear(f_dyn, gc_dim)

        def encode(self, xd, xs):
            # Z = ReLU(A_hat X W) at every quarter: self.gc applies W, and einsum 'ij,jlf->ilf'
            # sums over neighbours j of each sub-county i, weighted by A[i, j]. This mixes the
            # rows of the batch, so a T-GCN batch holds all sub-counties of one quarter.
            z = torch.relu(torch.einsum('ij,jlf->ilf', self.A, self.gc(xd)))
            out, _ = self.rnn(torch.cat([z, xd], dim=-1))
            return torch.cat([out[:, -1, :], xs], dim=1)

    # Wrapper for SHAP (Notebook 05): runs a trained network and returns only the stress
    # probability, as one column, the output shape SHAP's GradientExplainer works with.
    class StressProb(nn.Module):
        """Exposes only the classification head as a probability (used by SHAP)."""
        def __init__(self, m):
            super().__init__()
            self.m = m                       # the trained network

        def forward(self, xd, xs):
            # [0] picks the stress logit from the (stress, gap) pair; sigmoid turns it into a
            # probability; unsqueeze(1) changes the shape from (batch,) to (batch, 1).
            return torch.sigmoid(self.m(xd, xs)[0]).unsqueeze(1)
except ImportError:      # notebooks 01-03 do not need PyTorch
    pass
