"""Fig. 1: four-panel selective-prediction figure (DLCN and MMPD; retained MAE and worst-cell retention) from
the per-seed CSVs of selective_prediction.py. Shaded bands are mean +/- SEM across seeds.

Usage:
  python scripts/plot_selective_combined.py --dlcn_csv <csv> --mmpd_csv <csv> \
      --dlcn_worst "L4, S2" --mmpd_worst skin6 --out_dir out --basename fig1_selective_combined
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

CRITERIA = [
    ("naive", "Naive (raw $\\sigma$) $\\equiv$ LA-CP", "#1f77b4"),
    ("mondrian_cp", "Mondrian-CP (cell-aware)", "#d62728"),
]


def agg_curve(df, criterion, metric):
    sub = df[df.criterion == criterion].dropna(subset=[metric])
    agg = (sub.groupby("retention")[metric]
           .agg(["mean", "std", "count"]).reset_index())
    agg["sem"] = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
    return agg


def draw_row(axes_row, df, dataset_label, worst_str, panel_tags, show_legend):
    panels = [
        ("mae_overall", "Overall MAE on retained", "MAE (BPM)"),
        ("worst_cell_retention",
         f"Worst-cell retention ({worst_str})",
         "Worst-cell trials kept"),
    ]
    for ax, tag, (metric, title, ylabel) in zip(axes_row, panel_tags, panels):
        for crit, label, color in CRITERIA:
            agg = agg_curve(df, crit, metric)
            if agg.empty:
                continue
            ax.plot(agg.retention, agg["mean"], color=color, label=label,
                    linewidth=1.8, marker="o", markersize=3.5)
            ax.fill_between(agg.retention, agg["mean"] - agg["sem"],
                            agg["mean"] + agg["sem"], alpha=0.20, color=color)
        ax.set_ylabel(ylabel)
        ax.set_title(f"({tag}) {dataset_label}: {title}", fontsize="medium")
        ax.grid(alpha=0.3)
        ax.set_xlim(0.48, 1.02)

    axes_row[1].plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5, linewidth=1.0,
                     label="Proportional (random selection)")
    axes_row[1].set_ylim(-0.02, 1.05)
    if show_legend:
        axes_row[0].legend(loc="best", framealpha=0.95, fontsize="small")
        axes_row[1].legend(loc="upper left", framealpha=0.95, fontsize="small")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlcn_csv", required=True, type=Path,
                    help="selective_seeds*.csv from the DLCN (light,state) run")
    ap.add_argument("--mmpd_csv", required=True, type=Path,
                    help="selective_seeds*.csv from the MMPD (skin) run")
    ap.add_argument("--dlcn_worst", default="L4, S2")
    ap.add_argument("--mmpd_worst", default="skin=6")
    ap.add_argument("--dlcn_label", default="DLCN")
    ap.add_argument("--mmpd_label", default="MMPD")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--basename", default="fig1_selective_combined")
    ap.add_argument("--width", type=float, default=7.16,
                    help="inches; 7.16 = IEEE double-column, 3.5 = single")
    ap.add_argument("--height", type=float, default=6.0)
    ap.add_argument("--fontscale", type=float, default=1.0)
    ap.add_argument("--dpi", type=int, default=400)
    args = ap.parse_args()

    base = 9.0 * args.fontscale
    plt.rcParams.update({
        "font.size": base,
        "axes.titlesize": base * 1.05,
        "axes.labelsize": base,
        "xtick.labelsize": base * 0.9,
        "ytick.labelsize": base * 0.9,
        "legend.fontsize": base * 0.85,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })

    dlcn = pd.read_csv(args.dlcn_csv)
    mmpd = pd.read_csv(args.mmpd_csv)
    for name, d in (("DLCN", dlcn), ("MMPD", mmpd)):
        need = {"criterion", "retention", "mae_overall", "worst_cell_retention", "seed"}
        missing = need - set(d.columns)
        if missing:
            raise ValueError(f"{name} CSV missing columns: {sorted(missing)}")
        print(f"{name}: {d.seed.nunique()} seeds, criteria "
              f"{sorted(d.criterion.unique())}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(args.width, args.height),
                             sharex=True)

    draw_row(axes[0], dlcn, args.dlcn_label, args.dlcn_worst,
             ("A", "B"), show_legend=True)
    draw_row(axes[1], mmpd, args.mmpd_label, args.mmpd_worst,
             ("C", "D"), show_legend=False)

    for ax in axes[1]:
        ax.set_xlabel("Overall retention rate")

    fig.tight_layout(pad=0.6, h_pad=1.0, w_pad=1.2)
    for ext in ("pdf", "png"):
        out = args.out_dir / f"{args.basename}.{ext}"
        fig.savefig(out, dpi=args.dpi, bbox_inches="tight")
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
