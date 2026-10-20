"""Fig. 4 with 24 phase-space points: 8 PhysNet (derivation set) + 8 TSCAN + 8 PhysFormer (held-out)."""
from pathlib import Path
import pandas as pd, matplotlib as mpl, matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D

LAB = {"state": "state", "light": "light", "light_state": "l x s", "skin": "skin", "skin_state": "s x s"}
MARK = {"MMPD": "o", "DLCN": "s"}
EDGE = {"physnet": "#1a1a1a", "tscan": "#1f77b4", "physformer": "#2ca02c"}
XJ = {"physnet": 1.0, "tscan": 0.88, "physformer": 1.13}
OFF = {"dlcn_state": (22, 18), "dlcn_light": (26, 14), "dlcn_light_state": (26, 12),
       "mmpd_state": (26, 14), "mmpd_skin": (-30, -4), "mmpd_light": (-30, -14),
       "mmpd_skin_state": (26, -12), "mmpd_light_state": (26, 12)}
NAME = {"physnet": "PhysNet (derivation)", "tscan": "TSCAN (held-out)", "physformer": "PhysFormer (held-out)"}

pp = pd.concat([pd.read_csv(f"results/day_r1_matched/phase_points_{b}.csv") for b in EDGE])
mpl.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "font.size": 10,
                     "axes.labelsize": 11, "axes.titlesize": 11.5, "xtick.labelsize": 9, "ytick.labelsize": 9})
fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0), sharey=True)
cmap = plt.cm.RdBu_r


def panel(ax, col, norm, title, cb_label):
    for r in pp.itertuples():
        bb = r.backbone; main = bb == "physnet"
        color = cmap(norm(getattr(r, col)))
        ax.scatter(r.n_per_cell * XJ[bb], r.het_full, s=190 if main else 95, c=[color],
                   marker=MARK[r.dataset], edgecolors=EDGE[bb], linewidth=1.4 if main else 1.9, zorder=3 if main else 2)
        if main:
            dx, dy = OFF[r.config]
            ax.annotate(f"{r.dataset[0]}: {LAB[r.config.split('_', 1)[1]]}", (r.n_per_cell, r.het_full),
                        xytext=(dx, dy), textcoords="offset points", fontsize=8.2, color="#222222",
                        ha="left" if dx > 0 else "right", va="center", zorder=5,
                        arrowprops=dict(arrowstyle="-", lw=0.6, color="#777777", shrinkA=0, shrinkB=7),
                        bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    ax.axvline(30, color="#666666", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.axhline(1.5, color="#666666", linestyle="--", linewidth=1, alpha=0.7, zorder=1)
    ax.text(31, 11.5, r"$n_{\mathrm{trial}}/\mathrm{cell}=30$", fontsize=8.5, color="#555555", va="top")
    ax.text(210, 1.52, r"het$=1.5\times$", fontsize=8.5, color="#555555", ha="right", va="bottom")
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(8, 220); ax.set_ylim(1.0, 12)
    ax.set_xlabel("Calibration trials per cell"); ax.set_title(title, pad=8)
    ax.grid(True, which="major", alpha=0.18, zorder=0); ax.grid(True, which="minor", alpha=0.08, zorder=0)
    ax.set_xticks([10, 30, 100, 200]); ax.set_xticklabels(["10", "30", "100", "200"])
    ax.set_yticks([1, 1.5, 2, 3, 5, 10]); ax.set_yticklabels(["1", "1.5", "2", "3", "5", "10"])
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap); sm.set_array([])
    cb = plt.colorbar(sm, ax=ax, fraction=0.045, pad=0.02, extend="both"); cb.set_label(cb_label, fontsize=9)


panel(axes[0], "benefit_mondrian", TwoSlopeNorm(vmin=-2.5, vcenter=0, vmax=2.5),
      r"(A) MAE benefit at $\rho{=}0.5$  [BPM]", r"MAE benefit (Naive $-$ Mondrian)")
panel(axes[1], "abst", TwoSlopeNorm(vmin=-0.5, vcenter=0, vmax=0.5),
      r"(B) Worst-cell abstention strength at $\rho{=}0.5$", r"Naive $-$ Mondrian retention")
axes[0].set_ylabel("Cell MAE heterogeneity (max / min across cells)")
handles = [Line2D([0], [0], marker="o", color="w", markerfacecolor="#cccccc", markeredgecolor="#1a1a1a", markersize=10, label="MMPD"),
           Line2D([0], [0], marker="s", color="w", markerfacecolor="#cccccc", markeredgecolor="#1a1a1a", markersize=10, label="DLCN")]
handles += [Line2D([0], [0], marker="o", color="w", markerfacecolor="white", markeredgecolor=EDGE[b], markeredgewidth=2, markersize=9, label=NAME[b]) for b in EDGE]
axes[0].legend(handles=handles, loc="lower left", frameon=True, fontsize=8, edgecolor="#888888")
plt.tight_layout()
out = Path("results/day7"); out.mkdir(exist_ok=True)
plt.savefig(out / "fig4_phase_24.png", dpi=220, bbox_inches="tight")
plt.savefig(out / "fig4_phase_24.pdf", bbox_inches="tight")
print("Wrote results/day7/fig4_phase_24.{png,pdf}")
