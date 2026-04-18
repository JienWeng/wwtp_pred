# %% [markdown]
# # 01 — Exploratory Data Analysis
# WWTP effluent (PETALING-MAWAR). Targets: BO
# D, COD, NH3N, NO3N, TSS.
# Data is weekly-resampled (`W-MON`) with linear interpolation of gaps.

# %%
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils import (  # noqa: E402
    FIG_DIR,
    METRICS_DIR,
    TARGETS,
    WEEKLY_PARQUET,
    load_clean_resample,
    train_test_split_ts,
    plot_pred,
    save_metrics,
    naive_persistence,
    seasonal_naive,
)

sns.set_theme(style="whitegrid", context="notebook")
FIG_OUT = FIG_DIR / "01_eda"
FIG_OUT.mkdir(parents=True, exist_ok=True)

# %% [markdown]
# ## Load & peek

# %%
df = load_clean_resample()
print("Shape:", df.shape)
print("Range:", df.index.min(), "->", df.index.max())
df.head()

# %%
df.describe().T

# %% [markdown]
# ## Missingness & sampling frequency

# %%
print("NaNs per column (post-resample):")
print(df.isna().sum())

raw_per_year = {}
for year in (2023, 2024, 2025):
    raw = pd.read_csv(PROJECT_ROOT / "data" / f"{year}.csv")
    raw_per_year[year] = len(raw)
print("Raw rows per year:", raw_per_year)
print("Weekly rows after resample:", len(df))

# %% [markdown]
# ## Time-series overview — all targets

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 2.3 * len(TARGETS)), sharex=True)
for ax, col in zip(axes, TARGETS):
    ax.plot(df.index, df[col], color="#4c72b0", linewidth=1.2)
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
axes[-1].set_xlabel("Date")
fig.suptitle("Weekly effluent quality — PETALING-MAWAR", y=1.01)
fig.tight_layout()
fig.savefig(FIG_OUT / "timeseries_all.png", dpi=120)
plt.show()

# %% [markdown]
# ## Distributions (histogram + boxplot)

# %%
fig, axes = plt.subplots(2, len(TARGETS), figsize=(3 * len(TARGETS), 6))
for j, col in enumerate(TARGETS):
    sns.histplot(df[col], kde=True, ax=axes[0, j], color="#4c72b0")
    axes[0, j].set_title(col)
    sns.boxplot(y=df[col], ax=axes[1, j], color="#4c72b0")
fig.tight_layout()
fig.savefig(FIG_OUT / "distributions.png", dpi=120)
plt.show()

# %% [markdown]
# ## Correlation heatmap

# %%
corr = df[TARGETS].corr()
fig, ax = plt.subplots(figsize=(5.5, 4.5))
sns.heatmap(
    corr,
    annot=True,
    fmt=".2f",
    cmap="vlag",
    center=0,
    ax=ax,
    vmin=-1,
    vmax=1,
    square=True,
    cbar_kws={"shrink": 0.8},
)
ax.set_title("Pearson correlation across targets")
fig.tight_layout()
fig.savefig(FIG_OUT / "correlation.png", dpi=120)
plt.show()

# %% [markdown]
# ## Rolling mean + std (window = 4 weeks)

# %%
fig, axes = plt.subplots(len(TARGETS), 1, figsize=(11, 2.3 * len(TARGETS)), sharex=True)
for ax, col in zip(axes, TARGETS):
    roll_mean = df[col].rolling(4, min_periods=1).mean()
    roll_std = df[col].rolling(4, min_periods=1).std()
    ax.plot(df.index, df[col], color="#b0b0b0", linewidth=0.8, label="weekly")
    ax.plot(
        df.index, roll_mean, color="#4c72b0", linewidth=1.5, label="rolling mean (4w)"
    )
    ax.fill_between(
        df.index,
        roll_mean - roll_std,
        roll_mean + roll_std,
        color="#4c72b0",
        alpha=0.2,
        label="±1σ",
    )
    ax.set_ylabel(col)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
axes[-1].set_xlabel("Date")
fig.tight_layout()
fig.savefig(FIG_OUT / "rolling_stats.png", dpi=120)
plt.show()

# %% [markdown]
# ## Seasonal decomposition (period = 52 weeks)

# %%
from statsmodels.tsa.seasonal import seasonal_decompose

for col in TARGETS:
    try:
        result = seasonal_decompose(
            df[col], model="additive", period=52, extrapolate_trend="freq"
        )
    except Exception as e:
        print(f"Decompose failed for {col}: {e}")
        continue
    fig = result.plot()
    fig.set_size_inches(11, 6)
    fig.suptitle(f"Seasonal decomposition — {col}", y=1.02)
    fig.tight_layout()
    fig.savefig(FIG_OUT / f"decompose_{col}.png", dpi=120)
    plt.show()

# %% [markdown]
# ## Stationarity — Augmented Dickey-Fuller

# %%
from statsmodels.tsa.stattools import adfuller

adf_rows = []
for col in TARGETS:
    stat, pval, usedlag, nobs, crit, _ = adfuller(df[col].dropna(), autolag="AIC")
    adf_rows.append(
        {
            "target": col,
            "adf_stat": stat,
            "p_value": pval,
            "used_lag": usedlag,
            "n_obs": nobs,
            "crit_5%": crit["5%"],
            "stationary_at_5%": pval < 0.05,
        }
    )
adf_df = pd.DataFrame(adf_rows)
adf_df

# %% [markdown]
# ## ACF / PACF (lags = 24)

# %%
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

for col in TARGETS:
    fig, axes = plt.subplots(1, 2, figsize=(11, 3))
    plot_acf(df[col].dropna(), lags=24, ax=axes[0])
    axes[0].set_title(f"ACF — {col}")
    plot_pacf(df[col].dropna(), lags=24, ax=axes[1], method="ywm")
    axes[1].set_title(f"PACF — {col}")
    fig.tight_layout()
    fig.savefig(FIG_OUT / f"acf_pacf_{col}.png", dpi=120)
    plt.show()

# %% [markdown]
# ## Baseline reference metrics (naive persistence + seasonal naive 52)
#
# Saved under `outputs/metrics/00_baselines.csv` so every downstream notebook
# has an apples-to-apples reference independent of the modelling CSV files.

# %%
BASELINE_CSV = METRICS_DIR / "00_baselines.csv"
if BASELINE_CSV.exists():
    BASELINE_CSV.unlink()
BASE_FIG = FIG_DIR / "00_baselines"
BASE_FIG.mkdir(parents=True, exist_ok=True)

train_df, test_df = train_test_split_ts(df, test_frac=0.2)
print("train", train_df.shape, "test", test_df.shape)

rows = []
for target in TARGETS:
    np_pred = naive_persistence(train_df[target], test_df[target])
    sn_pred = seasonal_naive(train_df[target], test_df[target], period=52)
    m_np = plot_pred(train_df[target], test_df[target], np_pred, target,
                     "naive_persistence", BASE_FIG)
    m_sn = plot_pred(train_df[target], test_df[target], sn_pred, target,
                     "seasonal_naive_52", BASE_FIG)
    rows.append({"target": target, "model": "naive_persistence", **m_np})
    rows.append({"target": target, "model": "seasonal_naive_52", **m_sn})

save_metrics(rows, BASELINE_CSV)
pd.DataFrame(rows)[["target", "model", "MASE", "RMSE", "MAE", "MAPE", "sMAPE", "R2", "beats_naive"]]

# %% [markdown]
# ## Save cleaned weekly frame for downstream notebooks

# %%
WEEKLY_PARQUET.parent.mkdir(parents=True, exist_ok=True)
df.to_parquet(WEEKLY_PARQUET)
print("Saved ->", WEEKLY_PARQUET)
