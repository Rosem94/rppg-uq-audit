#!/usr/bin/env python
"""Fig. 3: cell-level spread (panel A) and per-trial rank association with the ensemble error (panel B) for the
chunk-averaged dropout sigma, the trial-consistent sigma and the five-seed ensemble.

Inputs: results/day_r1_matched/matched_{dlcn,mmpd}.csv. Outputs: results/day11_figs/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib.pyplot as plt

BASE = Path("results/day_r1_matched")
OUT = Path("results/day11_figs"); OUT.mkdir(parents=True, exist_ok=True)
SEEDS = range(5)

CFG = {
    "DLCN": dict(csv=BASE / "matched_dlcn.csv", cells=["light", "state"], marker="o"),
    "MMPD": dict(csv=BASE / "matched_mmpd.csv", cells=["skin", "state"], marker="s"),
}

rows, cellstats, rhos = [], {}, {}

for ds, cfg in CFG.items():
    d = pd.read_csv(cfg["csv"])
    g = d.groupby(cfg["cells"]).agg(
        mae=("ens_err", "mean"), sE=("sigma_E", "mean"),
        sC=("sig_cons", "mean"), sK=("sig_chunk", "mean")).reset_index()
    cellstats[ds] = g

    rho_E = spearmanr(d.sigma_E, d.ens_err)[0]
    rho_K = spearmanr(d.sig_chunk, d.ens_err)[0]
    per_seed = [spearmanr(d[f"cons_{s}"], d.ens_err)[0] for s in SEEDS]
    rho_C = float(np.mean(per_seed))
    rhos[ds] = dict(E=rho_E, C=rho_C, K=rho_K,
                    C_lo=min(per_seed), C_hi=max(per_seed))

    print(f"\n{ds}: n={len(d)}, cells={len(g)}")
    print(f"  rho vs ensemble |err|:  sigma_E={rho_E:+.3f}  "
          f"sigma_cons={rho_C:+.3f} (per-seed {min(per_seed):.3f}-{max(per_seed):.3f})  "
          f"sigma_chunk={rho_K:+.3f}")
    sp = lambda v: v.max() / v.min()
    print(f"  cell spread: MAE {sp(g.mae):.2f}x | sigma_E {sp(g.sE):.2f}x | "
          f"sigma_cons {sp(g.sC):.2f}x | sigma_chunk {sp(g.sK):.2f}x")
    rows.append(dict(dataset=ds, n_trials=len(d), n_cells=len(g),
                     rho_E=rho_E, rho_cons=rho_C, rho_chunk=rho_K,
                     spread_mae=sp(g.mae), spread_E=sp(g.sE),
                     spread_cons=sp(g.sC), spread_chunk=sp(g.sK)))

pd.DataFrame(rows).to_csv(OUT / "fig6_matched_summary.csv", index=False)

plt.rcParams.update({
    "font.family": "serif", "font.size": 8.5, "axes.labelsize": 9,
    "axes.titlesize": 9, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "legend.fontsize": 7.5, "axes.linewidth": 0.7,
})
fig, axes = plt.subplots(1, 2, figsize=(7.16, 3.2),
                         gridspec_kw={"width_ratios": [1.05, 1.0], "wspace": 0.28})

# ---- Panel A ----
ax = axes[0]
COL = {"sK": "#d1604c", "sC": "#1f6fbf", "sE": "#4c9a4c"}
TICK = {"sK": r"$\sigma$ chunk-avg." + chr(10) + "(default)",
        "sC": r"$\sigma^{\mathrm{cons}}$ trial-cons." + chr(10) + "(single model)",
        "sE": r"$\sigma_E$ ensemble" + chr(10) + "(5 seeds)"}
keys = ["sK", "sC", "sE"]
rng = np.random.default_rng(0)

for xi, key in enumerate(keys):
    for ds, cfg in CFG.items():
        g = cellstats[ds]
        v = (g[key] / g[key].min()).values
        jitter = (rng.random(len(v)) - 0.5) * 0.26 + (-0.10 if ds == "DLCN" else 0.10)
        ax.scatter(np.full(len(v), xi) + jitter, v, marker=cfg["marker"],
                   s=34, alpha=0.85, facecolors=COL[key], edgecolors="black",
                   linewidths=0.45, zorder=3)

for ds, cfg in CFG.items():
    g = cellstats[ds]
    m = g.mae.max() / g.mae.min()
    ax.axhline(m, ls="--", lw=0.8, color="0.45", zorder=1)
    ax.text(-0.42, m + 0.12, ds + " cell MAE spread (" + format(m, ".2f") + "x)",
            fontsize=6.8, ha="left", va="bottom", color="0.4", style="italic")

ticklabels = []
for k in keys:
    sd = cellstats["DLCN"][k].max() / cellstats["DLCN"][k].min()
    sm = cellstats["MMPD"][k].max() / cellstats["MMPD"][k].min()
    ticklabels.append(TICK[k] + chr(10) + format(sd, ".2f") + "x / " + format(sm, ".2f") + "x")

ax.set_xticks(range(len(keys)))
ax.set_xticklabels(ticklabels, fontsize=7.2)
ax.set_ylabel("Cell-mean value / min across cells")
ax.set_ylim(0.8, 7.6)
ax.set_xlim(-0.5, 2.5)
ax.set_title("(A) Cell-level spread by definition", loc="left", pad=6)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)


# ---- Panel B ----
ax = axes[1]
labels = list(CFG.keys())
x = np.arange(len(labels)); w = 0.26
series = [("E", r"$\sigma_E$ (deep ensemble)", "#4c9a4c"),
          ("C", r"$\sigma^{\mathrm{cons}}$ (single model)", "#1f6fbf"),
          ("K", r"$\sigma$ chunk-avg. (default)", "#d1604c")]
for i, (key, lab, col) in enumerate(series):
    vals = [rhos[ds][key] for ds in labels]
    bars = ax.bar(x + (i - 1) * w, vals, w, label=lab, color=col,
                  edgecolor="black", linewidth=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.012, f"{v:.2f}",
                ha="center", va="bottom", fontsize=7.5)
ax.axhline(0, color="black", lw=0.6)
ax.set_ylabel(r"Spearman $r_s\,(\sigma,\ |\mathrm{err}|)$")
ax.set_xticks(x); ax.set_xticklabels(labels)
ax.set_ylim(0, 0.78)
ax.set_title("(B) Per-trial ranking, common target", loc="left", pad=6)
ax.legend(loc="upper center", ncol=1, frameon=True, framealpha=0.95,
          edgecolor="0.6", bbox_to_anchor=(0.5, 1.0), borderaxespad=0.3)
ax.grid(True, axis="y", alpha=0.25, linewidth=0.5)

fig.subplots_adjust(left=0.09, right=0.985, top=0.90, bottom=0.22)
plt.savefig(OUT / "fig6_ensemble.pdf", bbox_inches="tight")
plt.savefig(OUT / "fig6_ensemble.png", dpi=300, bbox_inches="tight")
print(f"\nSaved: {OUT/'fig6_ensemble.pdf'} and .png")
