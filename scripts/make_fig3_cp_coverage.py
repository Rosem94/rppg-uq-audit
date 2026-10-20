"""Figure 3: split CP marginal vs conditional coverage on DLCN.

Three panels:
  (a) Marginal coverage across 10 seeds: both score types hit 90% target
      (sanity that CP implementation is correct)
  (b) Conditional coverage per (lighting x state) heatmap, locally_adaptive:
      L4 exercise drops to 0.74 (worst), L1 rest inflates to 0.99 (waste)
  (c) Width vs error: scatter of CI half-width vs |error| for each trial,
      with ideal y=x line. Shows that intervals are systematically too narrow
      on the upper tail (under-coverage) and too wide on lower tail (waste).
"""
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_TIDY = Path("results/day3_dlcn_full/dlcn_full_98p_n20.tidy_v2.csv")
CSV_SEEDS = Path("results/day4_cp/cp_results_seeds.csv")
CSV_COND = Path("results/day4_cp/cp_conditional_seeds.csv")
OUT_PATH = Path("results/day4_cp/fig3_cp_coverage.png")

mpl.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "figure.dpi": 130,
})

LIGHT_NAMES = {1: "FI&FP", 2: "VI&FP", 3: "FI&VP", 4: "VI&VP"}

# Load data
df_trials = pd.read_csv(CSV_TIDY)
df_seeds = pd.read_csv(CSV_SEEDS)
df_cond = pd.read_csv(CSV_COND)

# Recompute CP for one seed (locally_adaptive) to get per-trial widths
# (didn't save them earlier; recompute is cheap)
np.random.seed(0)
all_subjects = sorted(df_trials.pid.unique())
shuffled = np.random.default_rng(0).permutation(all_subjects)
cal_subs = sorted(shuffled[:30].tolist())
test_subs = sorted(shuffled[30:].tolist())
df_cal = df_trials[df_trials.pid.isin(cal_subs)]
df_test = df_trials[df_trials.pid.isin(test_subs)].copy()

s_cal = df_cal.abs_err.values / df_cal.unc_mean.values
n_cal = len(df_cal)
q_level = min(np.ceil((n_cal + 1) * 0.9) / n_cal, 1.0)
q_hat = np.quantile(s_cal, q_level, method="higher")
df_test["ci_half_width"] = q_hat * df_test.unc_mean
df_test["covered"] = (df_test.abs_err <= df_test.ci_half_width).astype(int)

print(f"Seed 0 q_hat = {q_hat:.2f}, marginal coverage = {df_test.covered.mean():.3f}")

fig = plt.figure(figsize=(13, 4.8))
gs = fig.add_gridspec(1, 3, width_ratios=[0.85, 1.05, 1.05], wspace=0.55)
ax_a = fig.add_subplot(gs[0])
ax_b = fig.add_subplot(gs[1])
ax_c = fig.add_subplot(gs[2])

# ---------- Panel (a): marginal coverage across seeds ----------
score_types = ["global", "locally_adaptive"]
positions = np.arange(len(score_types))
for i, st in enumerate(score_types):
    sub = df_seeds[df_seeds.score_type == st]
    ax_a.scatter([i] * len(sub), sub.marginal_coverage,
                 alpha=0.5, color="#0072B2", s=40)
    ax_a.errorbar([i], [sub.marginal_coverage.mean()],
                  yerr=[sub.marginal_coverage.std() / np.sqrt(len(sub))],
                  fmt="D", color="#D55E00", capsize=6, ms=9, mew=1.5)

ax_a.axhline(0.9, color="#999999", ls="--", lw=1.0, label="target 0.90")
ax_a.set_xticks(positions)
ax_a.set_xticklabels(["global\n|err|", "locally adaptive\n|err|/unc"])
ax_a.set_ylabel("Marginal coverage on test set")
ax_a.set_ylim(0.85, 0.95)
ax_a.set_title("(a) Marginal coverage across 10 seeds")
ax_a.legend(loc="lower right", fontsize=9, frameon=False)

# ---------- Panel (b): conditional coverage heatmap ----------
sub_cond = df_cond[df_cond.score_type == "locally_adaptive"]
piv = sub_cond.groupby(["state", "light"])["coverage"].mean().unstack("light")
piv = piv.reindex([1, 2], axis=0).reindex([1, 2, 3, 4], axis=1)

# Diverging colormap centered at 0.9
import matplotlib.colors as mcolors
norm = mcolors.TwoSlopeNorm(vmin=0.7, vcenter=0.9, vmax=1.0)
im = ax_b.imshow(piv.values, cmap="RdBu", aspect="auto", norm=norm)
ax_b.set_xticks(range(4))
ax_b.set_xticklabels([LIGHT_NAMES[i] for i in [1, 2, 3, 4]], fontsize=9)
ax_b.set_yticks(range(2))
ax_b.set_yticklabels(["rest", "exercise"])
ax_b.set_title("(b) Conditional coverage  (locally adaptive)\n     Target 0.90 (white). Red=under, Blue=over")

for i in range(2):
    for j in range(4):
        v = piv.values[i, j]
        ax_b.text(j, i, f"{v:.2f}",
                  ha="center", va="center", fontsize=10, fontweight="bold",
                  color="black" if 0.85 < v < 0.97 else "white")
fig.colorbar(im, ax=ax_b, pad=0.04, fraction=0.05).set_label("Coverage", fontsize=8)

# ---------- Panel (c): width vs error scatter ----------
LIGHT_COLORS = {1: "#0072B2", 2: "#009E73", 3: "#E69F00", 4: "#D55E00"}
for light in [1, 2, 3, 4]:
    sub = df_test[df_test.light == light]
    marker = "o" if (sub.state == 1).all() or (sub.state == 2).all() else "o"
    # Plot rest and exercise with different markers
    rest = sub[sub.state == 1]
    exer = sub[sub.state == 2]
    ax_c.scatter(rest.abs_err, rest.ci_half_width,
                 alpha=0.55, s=22, color=LIGHT_COLORS[light], marker="o",
                 label=f"L{light} {LIGHT_NAMES[light]} rest" if light == 1 else None,
                 edgecolors="none")
    ax_c.scatter(exer.abs_err, exer.ci_half_width,
                 alpha=0.55, s=30, color=LIGHT_COLORS[light], marker="^",
                 edgecolors="black", linewidths=0.4)

# y=x line: ideal coverage boundary (anything above is covered, below is missed)
err_max = df_test.abs_err.max()
ax_c.plot([0, err_max], [0, err_max], color="black", ls="--", lw=1.0,
          label="y=x  (above = covered)")

# Custom legend
import matplotlib.lines as mlines
handles = [mlines.Line2D([], [], color=LIGHT_COLORS[i], marker="o", ms=7, ls="",
                          label=f"L{i} {LIGHT_NAMES[i]}")
           for i in [1, 2, 3, 4]]
handles.append(mlines.Line2D([], [], color="black", marker="^", ms=8, ls="",
                              markeredgecolor="black", markerfacecolor="none",
                              label="exercise (▲)"))
handles.append(mlines.Line2D([], [], color="black", ls="--", lw=1.0, label="y=x"))
ax_c.legend(handles=handles, loc="lower right", fontsize=8, frameon=False, ncol=1)

ax_c.set_xlabel("Per-trial |HR error|  (BPM)")
ax_c.set_ylabel("CI half-width  (BPM)")
ax_c.set_title("(c) CI width vs error, per trial\n     Below dashed = uncovered")
ax_c.set_xlim(0, max(70, err_max + 2))
ax_c.set_ylim(0, df_test.ci_half_width.max() * 1.1)

plt.suptitle(
    "Split conformal prediction on DLCN: marginal coverage holds, conditional coverage fails",
    fontsize=11,
    y=1.02,
)
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(OUT_PATH, bbox_inches="tight")
print(f"Saved {OUT_PATH}")
plt.close(fig)