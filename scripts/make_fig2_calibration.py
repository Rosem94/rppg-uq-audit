"""Figure 2: uncertainty-error calibration curve.

Bin chunks by uncertainty quantile, compute empirical MAE per bin.
A perfectly-calibrated uncertainty produces monotonic increasing curve;
a flat or non-monotonic curve motivates conformal-style recalibration.
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import bootstrap

CSV_PATH = Path("results/day3_dlcn_full/dlcn_full_98p_n20.tidy.csv")
OUT_PATH = Path("results/day3_dlcn_full/fig2_calibration.png")
N_BINS = 10

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

df = pd.read_csv(CSV_PATH)
print(f"Loaded {len(df)} chunks")

# Quantile binning of uncertainty
df["unc_bin"] = pd.qcut(df.unc_mean_std, q=N_BINS, labels=False, duplicates="drop")

rng = np.random.default_rng(42)
records = []
for b in sorted(df.unc_bin.unique()):
    sub = df[df.unc_bin == b]
    ci = bootstrap(
        (sub.abs_err.values,), np.mean,
        n_resamples=1000, random_state=rng, confidence_level=0.95,
    ).confidence_interval
    records.append({
        "bin": int(b),
        "n": len(sub),
        "unc_mean": sub.unc_mean_std.mean(),
        "mae": sub.abs_err.mean(),
        "mae_lo": ci.low,
        "mae_hi": ci.high,
    })
cal = pd.DataFrame(records)
print(cal.round(3))

fig, ax = plt.subplots(figsize=(7, 4.5))

ax.errorbar(
    cal.unc_mean, cal.mae,
    yerr=[cal.mae - cal.mae_lo, cal.mae_hi - cal.mae],
    fmt="o-", color="#0072B2", capsize=4, lw=1.6, ms=7,
    label="empirical MAE per uncertainty quantile (95% bootstrap CI)",
)

# "Ideal" diagonal in DiffNormalized->BPM heuristic: if uncertainty perfectly
# tracked error, mean(|err|)~constant*unc. We show flat-baseline reference instead:
overall_mae = df.abs_err.mean()
ax.axhline(overall_mae, color="gray", lw=1.0, ls=":", label=f"overall MAE  ({overall_mae:.2f} BPM)")

# Annotate first/last bin n
# Annotate first/last bin with arrows pointing to the points (not over them)
# Annotate first/last bin in the empty corners away from the curve
ax.annotate(
    f"low-unc bin (n={cal.n.iloc[0]})",
    xy=(cal.unc_mean.iloc[0], cal.mae.iloc[0]),
    xytext=(0.05, 0.02),                         # bottom-left of axes
    textcoords="axes fraction",
    fontsize=8, ha="left", color="#0072B2",
    arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.7, alpha=0.6),
)
ax.annotate(
    f"high-unc bin (n={cal.n.iloc[-1]})",
    xy=(cal.unc_mean.iloc[-1], cal.mae.iloc[-1]),
    xytext=(0.98, 0.98),                          # top-right of axes
    textcoords="axes fraction",
    fontsize=8, ha="right", color="#0072B2",
    arrowprops=dict(arrowstyle="->", color="#0072B2", lw=0.7, alpha=0.6),
)

# Slope as crude calibration metric
slope, intercept = np.polyfit(cal.unc_mean, cal.mae, 1)
ax.plot(
    cal.unc_mean,
    slope * cal.unc_mean + intercept,
    color="#D55E00", lw=1.0, ls="--",
    label=f"linear fit  (slope = {slope:.1f} BPM per unit std)",
)

ax.set_xlabel("Mean per-chunk uncertainty in bin")
ax.set_ylabel("Empirical mean |HR error|  (BPM)")
ax.set_title(
    f"Figure 2: Calibration curve  (DLCN cross-dataset, n={len(df)} chunks)\n"
    f"Bimodal calibration: uncertainty is uninformative for bins 1-7, predictive for bins 8-10 → selective prediction works only on tail"
)
ax.legend(loc="upper left", fontsize=9, frameon=False)
fig.tight_layout()
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
plt.close(fig)