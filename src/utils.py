"""Shared utilities for the WWTP forecasting notebooks.

Single source of truth for: data loading + cleaning + weekly resampling,
train/test splitting, error metrics (regression + classification), naive
baselines, calendar feature engineering, sliding-window builder, and plotting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

TARGETS: list[str] = ["BOD", "COD", "NH3N", "NO3N", "TSS"]

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
METRICS_DIR = OUTPUTS_DIR / "metrics"
WEEKLY_PARQUET = OUTPUTS_DIR / "weekly.parquet"


# ---------------------------------------------------------------------------
# Data loading + cleaning
# ---------------------------------------------------------------------------

def _ensure_dirs() -> None:
    for p in (OUTPUTS_DIR, FIG_DIR, METRICS_DIR):
        p.mkdir(parents=True, exist_ok=True)


def _clean_censored(value) -> float:
    """Convert censored strings like '< 25' or '< 1.0' to half the detection limit."""
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float)):
        return float(value)
    s = str(value).strip()
    if s == "" or s.lower() in {"na", "nan", "none"}:
        return np.nan
    if s.startswith("<"):
        try:
            return float(s.lstrip("<").strip()) / 2.0
        except ValueError:
            return np.nan
    if s.startswith(">"):
        try:
            return float(s.lstrip(">").strip())
        except ValueError:
            return np.nan
    try:
        return float(s)
    except ValueError:
        return np.nan


def _read_year_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    keep = ["Date"] + TARGETS
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"{path.name} missing columns: {missing}")
    df = df[keep].copy()
    for col in TARGETS:
        df[col] = df[col].map(_clean_censored)
    df["Date"] = pd.to_datetime(df["Date"], format="%d-%b-%y", errors="coerce")
    df = df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    return df


def load_clean_resample(freq: str = "W-MON") -> pd.DataFrame:
    """Load 2023/2024/2025 CSVs, clean censored values, resample weekly + interpolate."""
    frames = []
    for year in (2023, 2024, 2025):
        frames.append(_read_year_csv(DATA_DIR / f"{year}.csv"))
    raw = pd.concat(frames, ignore_index=True).sort_values("Date")
    raw = raw.drop_duplicates(subset=["Date"], keep="first").set_index("Date")
    weekly = raw[TARGETS].resample(freq).mean()
    weekly = weekly.interpolate(method="linear", limit_direction="both")
    weekly = weekly.ffill().bfill()
    return weekly


def train_test_split_ts(df: pd.DataFrame, test_frac: float = 0.2):
    """Chronological split: last `test_frac` rows are test."""
    n = len(df)
    n_test = max(1, int(round(n * test_frac)))
    n_train = n - n_test
    return df.iloc[:n_train].copy(), df.iloc[n_train:].copy()


# ---------------------------------------------------------------------------
# Calendar features + sliding windows
# ---------------------------------------------------------------------------

def add_calendar_onehot(df: pd.DataFrame) -> pd.DataFrame:
    """Append one-hot calendar features (month=12, quarter=4, woy-bucket=4) to df."""
    out = df.copy()
    idx = pd.DatetimeIndex(out.index)
    months = pd.get_dummies(idx.month, prefix="m").set_index(out.index)
    months = months.reindex(columns=[f"m_{i}" for i in range(1, 13)], fill_value=0)
    quarters = pd.get_dummies(idx.quarter, prefix="q").set_index(out.index)
    quarters = quarters.reindex(columns=[f"q_{i}" for i in range(1, 5)], fill_value=0)
    woy = np.clip((np.asarray(idx.isocalendar().week.values, dtype=int) - 1) // 13, 0, 3)
    woy_bins = pd.get_dummies(woy, prefix="wob").set_index(out.index)
    woy_bins = woy_bins.reindex(columns=[f"wob_{i}" for i in range(0, 4)], fill_value=0)
    return pd.concat([out, months, quarters, woy_bins], axis=1).astype(
        {c: float for c in list(months.columns) + list(quarters.columns) + list(woy_bins.columns)}
    )


def make_windows(
    X: np.ndarray,
    y: np.ndarray,
    lookback: int,
    horizon: int = 1,
    step: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build sliding windows for supervised sequence learning.

    X: (T, F) feature matrix, y: (T,) target vector.
    Returns Xw: (N, lookback, F), yw: (N,) where yw[i] = y[i + lookback + horizon - 1].
    """
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    T, F = X.shape if X.ndim == 2 else (len(X), 1)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xw, yw = [], []
    last_start = T - lookback - horizon + 1
    for i in range(0, last_start, step):
        Xw.append(X[i : i + lookback])
        yw.append(y[i + lookback + horizon - 1])
    if not Xw:
        return (np.zeros((0, lookback, X.shape[1]), dtype=np.float32),
                np.zeros((0,), dtype=np.float32))
    return np.asarray(Xw, dtype=np.float32), np.asarray(yw, dtype=np.float32)


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def naive_persistence(y_train: pd.Series, y_test: pd.Series) -> pd.Series:
    """y_hat_t = y_{t-1}. First test point uses last train value."""
    last = float(y_train.iloc[-1])
    preds = np.concatenate([[last], y_test.values[:-1]])
    return pd.Series(preds, index=y_test.index, name="naive_persistence")


def seasonal_naive(y_train: pd.Series, y_test: pd.Series, period: int = 52) -> pd.Series:
    """y_hat_t = y_{t-period}. Falls back to naive persistence when history too short."""
    full = pd.concat([y_train, y_test])
    preds = []
    for i, idx in enumerate(y_test.index):
        pos = len(y_train) + i - period
        if pos >= 0:
            preds.append(float(full.iloc[pos]))
        else:
            preds.append(float(y_train.iloc[-1]))
    return pd.Series(preds, index=y_test.index, name=f"seasonal_naive_{period}")


# ---------------------------------------------------------------------------
# Metrics (regression + classification)
# ---------------------------------------------------------------------------

def metrics(
    y_true: Sequence[float],
    y_pred: Sequence[float],
    y_train: Optional[Sequence[float]] = None,
) -> Dict[str, float]:
    """Regression metrics.

    Returns MAPE, sMAPE, RMSE, MAE, R2, MASE, beats_naive.
    MASE requires `y_train` (falls back to NaN if not provided or too short).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]; y_pred = y_pred[mask]
    out: Dict[str, float] = {
        "MAPE": np.nan, "sMAPE": np.nan, "RMSE": np.nan,
        "MAE": np.nan, "R2": np.nan, "MASE": np.nan, "beats_naive": np.nan,
    }
    if y_true.size == 0:
        return out
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = np.where(np.abs(y_true) < 1e-8, np.nan, y_true)
    mape = float(np.nanmean(np.abs(err / denom)) * 100.0)
    s_den = (np.abs(y_true) + np.abs(y_pred)) / 2.0
    s_den = np.where(s_den < 1e-8, np.nan, s_den)
    smape = float(np.nanmean(np.abs(err) / s_den) * 100.0)
    ss_res = float(np.sum(err ** 2))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    mase = np.nan
    beats = np.nan
    if y_train is not None:
        yt = np.asarray(y_train, dtype=float)
        yt = yt[np.isfinite(yt)]
        if yt.size > 1:
            naive_mae = float(np.mean(np.abs(np.diff(yt))))
            if naive_mae > 1e-12:
                mase = mae / naive_mae
                beats = float(mase < 1.0)

    out.update({
        "MAPE": mape, "sMAPE": smape, "RMSE": rmse,
        "MAE": mae, "R2": float(r2), "MASE": mase, "beats_naive": beats,
    })
    return out


def classification_metrics(
    y_true_bin: Sequence[int],
    y_prob: Sequence[float],
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Accuracy, precision, recall, F1, AUC + confusion matrix entries.

    Safe against degenerate cases (all one class, no positives predicted, etc.).
    """
    y_true = np.asarray(y_true_bin, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    n = tp + tn + fp + fn
    acc = (tp + tn) / n if n > 0 else np.nan
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
    try:
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y_true)) < 2:
            auc = np.nan
        else:
            auc = float(roc_auc_score(y_true, y_prob))
    except Exception:
        auc = np.nan
    return {
        "accuracy": float(acc), "precision": float(prec), "recall": float(rec),
        "F1": float(f1), "AUC": float(auc) if auc == auc else np.nan,
        "TP": tp, "TN": tn, "FP": fp, "FN": fn,
    }


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def format_metrics(m: Dict[str, float]) -> str:
    parts = []
    if "MASE" in m and m["MASE"] == m["MASE"]:
        parts.append(f"MASE={m['MASE']:.2f}")
    parts.append(f"RMSE={m['RMSE']:.3f}")
    parts.append(f"MAE={m['MAE']:.3f}")
    parts.append(f"MAPE={m['MAPE']:.1f}%")
    parts.append(f"R2={m['R2']:.2f}")
    return "  ".join(parts)


def plot_pred(
    y_train: pd.Series,
    y_test: pd.Series,
    y_pred: pd.Series,
    target: str,
    model_name: str,
    out_dir: Path | str,
    show: bool = True,
) -> Dict[str, float]:
    """Plot train + test + prediction + naive-persistence overlay. Returns metrics."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    m = metrics(y_test.values, y_pred.values, y_train=y_train.values)
    naive = naive_persistence(y_train, y_test)

    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.plot(y_train.index, y_train.values, color="#4c72b0", label="Train", linewidth=1.2)
    ax.plot(y_test.index, y_test.values, color="#000000", label="Test (true)", linewidth=1.5)
    ax.plot(naive.index, naive.values, color="#888888", linestyle=":", linewidth=1.0,
            label="Naive (persistence)")
    ax.plot(y_pred.index, y_pred.values, color="#dd8452", label=f"{model_name} pred",
            linewidth=1.5, linestyle="--")
    ax.axvspan(y_test.index.min(), y_test.index.max(), color="#eeeeee", alpha=0.5, zorder=0)
    ax.set_title(f"{target} — {model_name}\n{format_metrics(m)}")
    ax.set_xlabel("Date")
    ax.set_ylabel(target)
    ax.legend(loc="best", fontsize=8)
    fig.tight_layout()

    fname = f"{target}_{model_name}.png".replace(" ", "_").replace("/", "-")
    fig.savefig(out_dir / fname, dpi=120)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return m


def plot_confusion(cm: Dict[str, int], labels: Tuple[str, str], title: str, out_path: Path | str) -> None:
    """Plot a 2x2 confusion matrix heatmap from a {TP,TN,FP,FN} dict."""
    mat = np.array([[cm["TN"], cm["FP"]], [cm["FN"], cm["TP"]]], dtype=int)
    fig, ax = plt.subplots(figsize=(3.8, 3.4))
    im = ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, str(mat[i, j]), ha="center", va="center",
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels([f"pred {labels[0]}", f"pred {labels[1]}"])
    ax.set_yticks([0, 1]); ax.set_yticklabels([f"true {labels[0]}", f"true {labels[1]}"])
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.7)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120)
    plt.show()


# ---------------------------------------------------------------------------
# Metric CSV helper
# ---------------------------------------------------------------------------

def save_metrics(rows: Iterable[Dict], out_path: Path | str) -> pd.DataFrame:
    """Write / append rows to a metrics CSV. Returns the combined DataFrame."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_new = pd.DataFrame(list(rows))
    if out_path.exists():
        df_old = pd.read_csv(out_path)
        df = pd.concat([df_old, df_new], ignore_index=True)
    else:
        df = df_new
    df.to_csv(out_path, index=False)
    return df


_ensure_dirs()
