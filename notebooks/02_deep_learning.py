# %% [markdown]
# # 02 — Deep Learning Forecasting (LSTM, GRU, Transformer, LSTM Auto-Encoder)
#
# Long notebook. Per target (`BOD`, `COD`, `NH3N`, `NO3N`, `TSS`), we:
#
# 1. Frame the problem with a **sliding window** (lookback L weeks) and a
#    **prediction horizon** PH (weeks ahead).
# 2. Add **one-hot calendar features** (month, quarter, week-of-year bucket) as
#    exogenous inputs alongside the scaled target.
# 3. Fit four PyTorch architectures with a small hyper-parameter grid:
#    `LSTM`, `GRU`, `Transformer`, `LSTMAutoEncoder`.
# 4. Select the best config per (target, arch) by validation RMSE
#    (regression) or validation F1 (classification).
# 5. Refit on the full training fold, forecast the 20% test fold, and compare
#    against **naive persistence** + **seasonal naive (52w)** baselines.
# 6. Also train a **classification head** using the per-target 75th-percentile
#    threshold on the training data (same grid).
#
# Headline metric: `MASE` (regression) / `F1` (classification).

# %%
import sys
import math
import itertools
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    load_clean_resample,
    train_test_split_ts,
    add_calendar_onehot,
    make_windows,
    metrics,
    classification_metrics,
    naive_persistence,
    seasonal_naive,
    plot_pred,
    plot_confusion,
    save_metrics,
    TARGETS,
    FIG_DIR,
    METRICS_DIR,
    WEEKLY_PARQUET,
)

SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", DEVICE)

FIG_OUT = FIG_DIR / "02_deep_learning"
FIG_OUT.mkdir(parents=True, exist_ok=True)
METRICS_CSV = METRICS_DIR / "02_deep_learning.csv"
if METRICS_CSV.exists():
    METRICS_CSV.unlink()

# %% [markdown]
# ## Load weekly frame + calendar features + split

# %%
if WEEKLY_PARQUET.exists():
    df_raw = pd.read_parquet(WEEKLY_PARQUET)
else:
    df_raw = load_clean_resample()

df = add_calendar_onehot(df_raw)
calendar_cols = [c for c in df.columns if c not in TARGETS]
print("calendar feature columns:", len(calendar_cols))
df.head()

# %%
train_df, test_df = train_test_split_ts(df, test_frac=0.2)
print("train", train_df.shape, "test", test_df.shape)

# %% [markdown]
# ## Per-target scaling (fit on train only)

# %%
class MinMax:
    def __init__(self):
        self.lo = 0.0; self.hi = 1.0
    def fit(self, x: np.ndarray):
        self.lo = float(np.min(x)); self.hi = float(np.max(x))
        if self.hi - self.lo < 1e-12:
            self.hi = self.lo + 1.0
        return self
    def transform(self, x):
        return (np.asarray(x, dtype=np.float32) - self.lo) / (self.hi - self.lo)
    def inverse(self, x):
        return np.asarray(x, dtype=np.float32) * (self.hi - self.lo) + self.lo


def build_feature_matrix(target: str, train_df: pd.DataFrame, test_df: pd.DataFrame):
    """Return (X_train, X_test, scaler) as float32 matrices.

    Column 0 is the scaled target; remaining columns are calendar one-hots.
    """
    scaler = MinMax().fit(train_df[target].values)
    y_tr = scaler.transform(train_df[target].values)
    y_te = scaler.transform(test_df[target].values)
    X_tr = np.concatenate([y_tr.reshape(-1, 1), train_df[calendar_cols].values.astype(np.float32)], axis=1)
    X_te = np.concatenate([y_te.reshape(-1, 1), test_df[calendar_cols].values.astype(np.float32)], axis=1)
    return X_tr.astype(np.float32), X_te.astype(np.float32), scaler

# %% [markdown]
# ## Model zoo

# %%
class LSTMModel(nn.Module):
    def __init__(self, in_feat: int, hidden: int, num_layers: int, dropout: float, out_dim: int = 1):
        super().__init__()
        self.rnn = nn.LSTM(in_feat, hidden, num_layers=num_layers,
                           batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class GRUModel(nn.Module):
    def __init__(self, in_feat: int, hidden: int, num_layers: int, dropout: float, out_dim: int = 1):
        super().__init__()
        self.rnn = nn.GRU(in_feat, hidden, num_layers=num_layers,
                          batch_first=True,
                          dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)
    def forward(self, x):
        out, _ = self.rnn(x)
        return self.head(out[:, -1, :])


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, : x.size(1), :]


class TransformerModel(nn.Module):
    def __init__(self, in_feat: int, embed: int, num_layers: int, nhead: int,
                 dropout: float, out_dim: int = 1):
        super().__init__()
        self.proj = nn.Linear(in_feat, embed)
        self.posenc = PositionalEncoding(embed)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed, nhead=nhead, dim_feedforward=4 * embed,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        self.head = nn.Linear(embed, out_dim)
    def forward(self, x):
        z = self.proj(x)
        z = self.posenc(z)
        z = self.encoder(z)
        return self.head(z[:, -1, :])


class LSTMAutoEncoder(nn.Module):
    """Conv1D -> Dropout -> MaxPool -> LSTM -> RepeatVector -> LSTM -> Dense.

    Follows Fig. 4 of the reference paper but with a single output per window.
    """
    def __init__(self, in_feat: int, hidden: int, num_layers: int, dropout: float,
                 out_dim: int = 1):
        super().__init__()
        self.conv = nn.Conv1d(in_feat, hidden, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.enc = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.dec = nn.LSTM(hidden, hidden, num_layers=num_layers, batch_first=True,
                           dropout=dropout if num_layers > 1 else 0.0)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, x):
        z = x.transpose(1, 2)
        z = self.relu(self.conv(z))
        z = self.drop(z)
        if z.size(-1) >= 2:
            z = self.pool(z)
        z = z.transpose(1, 2)
        _, (h, _) = self.enc(z)
        last = h[-1]
        repeat_len = max(z.size(1), 1)
        dec_in = last.unsqueeze(1).repeat(1, repeat_len, 1)
        dec_out, _ = self.dec(dec_in)
        return self.head(dec_out[:, -1, :])


def build_model(arch: str, in_feat: int, units: int, num_layers: int, dropout: float,
                nhead: int = 2, out_dim: int = 1) -> nn.Module:
    if arch == "LSTM":
        return LSTMModel(in_feat, units, num_layers, dropout, out_dim)
    if arch == "GRU":
        return GRUModel(in_feat, units, num_layers, dropout, out_dim)
    if arch == "Transformer":
        embed = units
        if embed % nhead != 0:
            embed = (embed // nhead) * nhead if embed >= nhead else nhead
            embed = max(embed, nhead)
        return TransformerModel(in_feat, embed, num_layers, nhead, dropout, out_dim)
    if arch == "LSTM_AE":
        return LSTMAutoEncoder(in_feat, units, num_layers, dropout, out_dim)
    raise ValueError(arch)

# %% [markdown]
# ## Training + evaluation helpers

# %%
def fit_torch(model, train_loader, val_loader, lr, loss_fn, epochs=120, patience=15,
              use_logits: bool = False):
    model.to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    best_val = float("inf"); best_state = None; bad = 0
    for _ in range(epochs):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE); yb = yb.to(DEVICE)
            opt.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
        model.eval()
        vl = 0.0; n = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(DEVICE); yb = yb.to(DEVICE)
                pred = model(xb)
                vl += loss_fn(pred, yb).item() * xb.size(0)
                n += xb.size(0)
        vl = vl / max(n, 1)
        if vl < best_val - 1e-6:
            best_val = vl
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val


@torch.no_grad()
def recursive_forecast_regression(model, X_train_feat: np.ndarray, X_test_feat: np.ndarray,
                                  lookback: int, horizon: int, scaler: MinMax) -> np.ndarray:
    """Roll the scaled target forward one step at a time across the test horizon.

    Calendar features for each future step come from `X_test_feat` (they are known);
    the scaled target column is replaced with the model's own predictions as we go.
    """
    model.eval()
    hist = X_train_feat.copy()
    hist = np.concatenate([hist, X_test_feat.copy()], axis=0)
    n_train = len(X_train_feat)
    preds_scaled = np.zeros(len(X_test_feat), dtype=np.float32)
    for t in range(len(X_test_feat)):
        start = n_train + t - lookback
        if start < 0:
            pad = np.zeros((-start, hist.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, hist[: n_train + t]], axis=0)
        else:
            window = hist[start : n_train + t]
        x = torch.tensor(window.reshape(1, lookback, -1), device=DEVICE)
        y_pred_scaled = float(model(x).cpu().numpy().ravel()[0])
        preds_scaled[t] = y_pred_scaled
        hist[n_train + t, 0] = y_pred_scaled
    return scaler.inverse(preds_scaled)


@torch.no_grad()
def rolling_prob_classification(model, X_train_feat: np.ndarray, X_test_feat: np.ndarray,
                                lookback: int) -> np.ndarray:
    """One-step-ahead probability per test point (calendar known; target column is actual history)."""
    model.eval()
    hist = np.concatenate([X_train_feat, X_test_feat], axis=0)
    n_train = len(X_train_feat)
    probs = np.zeros(len(X_test_feat), dtype=np.float32)
    for t in range(len(X_test_feat)):
        start = n_train + t - lookback
        if start < 0:
            pad = np.zeros((-start, hist.shape[1]), dtype=np.float32)
            window = np.concatenate([pad, hist[: n_train + t]], axis=0)
        else:
            window = hist[start : n_train + t]
        x = torch.tensor(window.reshape(1, lookback, -1), device=DEVICE)
        logit = float(model(x).cpu().numpy().ravel()[0])
        probs[t] = 1.0 / (1.0 + math.exp(-logit))
    return probs


@dataclass
class Config:
    arch: str
    units: int
    num_layers: int
    dropout: float
    lr: float
    lookback: int
    horizon: int
    nhead: int = 2


def build_grid() -> List[Config]:
    grid = []
    archs = ["LSTM", "GRU", "Transformer", "LSTM_AE"]
    for arch, units, layers, lr, lb, ph in itertools.product(
        archs, [32, 64], [1, 2], [1e-3, 1e-4], [4, 8], [1, 4]
    ):
        dropouts = [0.1, 0.2] if layers == 2 else [0.0]
        for drp in dropouts:
            if arch == "Transformer":
                for nh in [2, 4]:
                    grid.append(Config(arch, units, layers, drp, lr, lb, ph, nhead=nh))
            else:
                grid.append(Config(arch, units, layers, drp, lr, lb, ph))
    return grid


GRID = build_grid()
print("grid size:", len(GRID))

# %% [markdown]
# ## Baselines logged first

# %%
baseline_rows = []
for target in TARGETS:
    y_train = train_df[target]
    y_test = test_df[target]
    np_pred = naive_persistence(y_train, y_test)
    sn_pred = seasonal_naive(y_train, y_test, period=52)
    m_np = plot_pred(y_train, y_test, np_pred, target, "naive_persistence", FIG_OUT)
    m_sn = plot_pred(y_train, y_test, sn_pred, target, "seasonal_naive_52", FIG_OUT)
    baseline_rows.append({"target": target, "model": "naive_persistence", "is_best": True, **m_np})
    baseline_rows.append({"target": target, "model": "seasonal_naive_52", "is_best": True, **m_sn})

save_metrics(baseline_rows, METRICS_CSV)

# %% [markdown]
# ## Regression sweep — per target per arch, pick best by val RMSE
#
# Note: this loop is the heavy cell. It runs thousands of tiny fits. If it is
# too slow on CPU, shrink `GRID` (e.g. drop `horizon=4` or `num_layers=2`).

# %%
def regression_sweep(target: str, X_tr: np.ndarray, X_te: np.ndarray, scaler: MinMax,
                     arch_filter: Iterable[str]) -> Dict[str, Dict]:
    """Returns {arch: best_config_dict_with_val_rmse_scaled}."""
    y_col = X_tr[:, 0]
    results: Dict[str, Dict] = {}
    best_per_arch: Dict[str, Tuple[float, Config, float]] = {}
    n = len(X_tr)
    n_val = max(int(n * 0.2), 10)
    inner_train_end = n - n_val

    for cfg in GRID:
        if cfg.arch not in arch_filter:
            continue
        if cfg.lookback >= inner_train_end or cfg.lookback + cfg.horizon >= n:
            continue
        Xtr, ytr = make_windows(X_tr[:inner_train_end], y_col[:inner_train_end],
                                cfg.lookback, cfg.horizon)
        Xva, yva = make_windows(
            X_tr[max(0, inner_train_end - cfg.lookback - cfg.horizon + 1):],
            y_col[max(0, inner_train_end - cfg.lookback - cfg.horizon + 1):],
            cfg.lookback, cfg.horizon,
        )
        if len(Xtr) < 4 or len(Xva) < 2:
            continue
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                             torch.from_numpy(ytr.reshape(-1, 1))),
                               batch_size=16, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva),
                                             torch.from_numpy(yva.reshape(-1, 1))),
                               batch_size=16, shuffle=False)
        torch.manual_seed(SEED)
        model = build_model(cfg.arch, X_tr.shape[1], cfg.units, cfg.num_layers,
                            cfg.dropout, nhead=cfg.nhead)
        model, val_mse = fit_torch(model, tr_loader, va_loader, cfg.lr, nn.MSELoss())
        val_rmse = float(np.sqrt(val_mse))

        if cfg.arch not in best_per_arch or val_rmse < best_per_arch[cfg.arch][0]:
            best_per_arch[cfg.arch] = (val_rmse, cfg, val_mse)

    for arch, (vrmse, cfg, _) in best_per_arch.items():
        results[arch] = {"val_rmse_scaled": vrmse, **asdict(cfg)}
    return results


all_rows: List[Dict] = list(baseline_rows)
arch_list = ["LSTM", "GRU", "Transformer", "LSTM_AE"]

for target in TARGETS:
    print(f"\n===== Regression: {target} =====")
    X_tr, X_te, scaler = build_feature_matrix(target, train_df, test_df)

    best_by_arch = regression_sweep(target, X_tr, X_te, scaler, arch_list)

    for arch in arch_list:
        if arch not in best_by_arch:
            print(f"  {arch}: no viable config")
            continue
        cfg_d = best_by_arch[arch]
        cfg = Config(
            arch=arch, units=cfg_d["units"], num_layers=cfg_d["num_layers"],
            dropout=cfg_d["dropout"], lr=cfg_d["lr"], lookback=cfg_d["lookback"],
            horizon=cfg_d["horizon"], nhead=cfg_d.get("nhead", 2),
        )
        print(f"  {arch} best val RMSE(scaled)={cfg_d['val_rmse_scaled']:.4f}  "
              f"L={cfg.lookback} H={cfg.horizon} u={cfg.units} layers={cfg.num_layers} "
              f"drp={cfg.dropout} lr={cfg.lr}")

        Xfull, yfull = make_windows(X_tr, X_tr[:, 0], cfg.lookback, cfg.horizon)
        n_val = max(int(len(Xfull) * 0.2), 4)
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xfull[:-n_val]),
                                             torch.from_numpy(yfull[:-n_val].reshape(-1, 1))),
                               batch_size=16, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xfull[-n_val:]),
                                             torch.from_numpy(yfull[-n_val:].reshape(-1, 1))),
                               batch_size=16, shuffle=False)
        torch.manual_seed(SEED)
        final_model = build_model(arch, X_tr.shape[1], cfg.units, cfg.num_layers,
                                  cfg.dropout, nhead=cfg.nhead)
        final_model, _ = fit_torch(final_model, tr_loader, va_loader, cfg.lr, nn.MSELoss())

        preds = recursive_forecast_regression(final_model, X_tr, X_te,
                                              cfg.lookback, cfg.horizon, scaler)
        y_pred = pd.Series(preds, index=test_df.index)
        m = plot_pred(train_df[target], test_df[target], y_pred, target, f"{arch}_best", FIG_OUT)
        all_rows.append({
            "target": target, "model": arch, "task": "regression",
            "is_best": True, **asdict(cfg),
            "val_rmse_scaled": cfg_d["val_rmse_scaled"], **m,
        })

save_metrics(all_rows[len(baseline_rows):], METRICS_CSV)
print("Saved regression metrics ->", METRICS_CSV)

# %% [markdown]
# ## Classification sweep — per-target 75th-percentile threshold

# %%
def classification_sweep(target: str, X_tr: np.ndarray, X_te: np.ndarray,
                         y_bin_tr: np.ndarray, y_bin_te: np.ndarray,
                         arch_filter: Iterable[str]) -> Dict[str, Dict]:
    """Returns {arch: best_config by val F1}."""
    n = len(X_tr)
    n_val = max(int(n * 0.2), 10)
    inner_train_end = n - n_val
    best_per_arch: Dict[str, Tuple[float, Config]] = {}

    for cfg in GRID:
        if cfg.arch not in arch_filter:
            continue
        if cfg.lookback >= inner_train_end or cfg.lookback + cfg.horizon >= n:
            continue
        Xtr, ytr = make_windows(X_tr[:inner_train_end], y_bin_tr[:inner_train_end],
                                cfg.lookback, cfg.horizon)
        Xva, yva = make_windows(
            X_tr[max(0, inner_train_end - cfg.lookback - cfg.horizon + 1):],
            y_bin_tr[max(0, inner_train_end - cfg.lookback - cfg.horizon + 1):],
            cfg.lookback, cfg.horizon,
        )
        if len(Xtr) < 4 or len(Xva) < 2:
            continue
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xtr),
                                             torch.from_numpy(ytr.reshape(-1, 1))),
                               batch_size=16, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xva),
                                             torch.from_numpy(yva.reshape(-1, 1))),
                               batch_size=16, shuffle=False)
        torch.manual_seed(SEED)
        model = build_model(cfg.arch, X_tr.shape[1], cfg.units, cfg.num_layers,
                            cfg.dropout, nhead=cfg.nhead)
        loss_fn = nn.BCEWithLogitsLoss()
        model, _ = fit_torch(model, tr_loader, va_loader, cfg.lr, loss_fn)

        model.eval()
        with torch.no_grad():
            vlogits = model(torch.from_numpy(Xva).to(DEVICE)).cpu().numpy().ravel()
        vprobs = 1.0 / (1.0 + np.exp(-vlogits))
        vcm = classification_metrics(yva.astype(int), vprobs, threshold=0.5)
        f1 = vcm["F1"]

        if cfg.arch not in best_per_arch or f1 > best_per_arch[cfg.arch][0]:
            best_per_arch[cfg.arch] = (f1, cfg)

    return {arch: {"val_F1": f1, **asdict(cfg)} for arch, (f1, cfg) in best_per_arch.items()}


cls_rows: List[Dict] = []
thresholds_per_target: Dict[str, float] = {}

for target in TARGETS:
    print(f"\n===== Classification: {target} =====")
    thr = float(np.quantile(train_df[target].values, 0.75))
    thresholds_per_target[target] = thr
    pos_rate_train = float(np.mean(train_df[target].values > thr))
    pos_rate_test = float(np.mean(test_df[target].values > thr))
    print(f"  threshold={thr:.3f}  pos_rate train={pos_rate_train:.2f}  test={pos_rate_test:.2f}")

    X_tr, X_te, scaler = build_feature_matrix(target, train_df, test_df)
    y_bin_tr = (train_df[target].values > thr).astype(np.float32)
    y_bin_te = (test_df[target].values > thr).astype(np.float32)

    best_by_arch = classification_sweep(target, X_tr, X_te, y_bin_tr, y_bin_te, arch_list)

    for arch in arch_list:
        if arch not in best_by_arch:
            print(f"  {arch}: no viable config")
            continue
        cfg_d = best_by_arch[arch]
        cfg = Config(
            arch=arch, units=cfg_d["units"], num_layers=cfg_d["num_layers"],
            dropout=cfg_d["dropout"], lr=cfg_d["lr"], lookback=cfg_d["lookback"],
            horizon=cfg_d["horizon"], nhead=cfg_d.get("nhead", 2),
        )
        print(f"  {arch} val F1={cfg_d['val_F1']:.3f}  L={cfg.lookback} H={cfg.horizon} "
              f"u={cfg.units} layers={cfg.num_layers} drp={cfg.dropout} lr={cfg.lr}")

        Xfull, yfull = make_windows(X_tr, y_bin_tr, cfg.lookback, cfg.horizon)
        n_val = max(int(len(Xfull) * 0.2), 4)
        tr_loader = DataLoader(TensorDataset(torch.from_numpy(Xfull[:-n_val]),
                                             torch.from_numpy(yfull[:-n_val].reshape(-1, 1))),
                               batch_size=16, shuffle=True)
        va_loader = DataLoader(TensorDataset(torch.from_numpy(Xfull[-n_val:]),
                                             torch.from_numpy(yfull[-n_val:].reshape(-1, 1))),
                               batch_size=16, shuffle=False)
        torch.manual_seed(SEED)
        final_model = build_model(arch, X_tr.shape[1], cfg.units, cfg.num_layers,
                                  cfg.dropout, nhead=cfg.nhead)
        final_model, _ = fit_torch(final_model, tr_loader, va_loader, cfg.lr,
                                   nn.BCEWithLogitsLoss())

        probs = rolling_prob_classification(final_model, X_tr, X_te, cfg.lookback)
        cm = classification_metrics(y_bin_te.astype(int), probs, threshold=0.5)
        print(f"    test F1={cm['F1']:.3f} AUC={cm['AUC']:.3f} acc={cm['accuracy']:.3f}")

        plot_confusion(cm, labels=("low", "high"),
                       title=f"{target} — {arch} (thr={thr:.2f})",
                       out_path=FIG_OUT / f"confusion_{target}_{arch}.png")

        cls_rows.append({
            "target": target, "model": arch, "task": "classification",
            "threshold": thr, "pos_rate_train": pos_rate_train, "pos_rate_test": pos_rate_test,
            **asdict(cfg), "val_F1": cfg_d["val_F1"], **cm,
        })

save_metrics(cls_rows, METRICS_CSV)
print("Saved classification metrics ->", METRICS_CSV)

# %% [markdown]
# ## Summary tables

# %%
df_all = pd.read_csv(METRICS_CSV)
print("Regression — best by MASE per target:")
reg = df_all[df_all.get("task", pd.Series([None] * len(df_all))).fillna("baseline") != "classification"]
reg_view = reg[["target", "model", "MASE", "RMSE", "MAE", "MAPE", "sMAPE", "R2", "beats_naive"]]
reg_view

# %%
print("Classification — summary:")
cls = df_all[df_all.get("task", pd.Series([None] * len(df_all))) == "classification"]
cls_view = cls[["target", "model", "threshold", "accuracy", "precision", "recall", "F1", "AUC"]]
cls_view
