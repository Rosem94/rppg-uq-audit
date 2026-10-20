"""Cross-backbone summary (Table III): per-cell MAE and mean dropout sigma for PhysNet, TSCAN and PhysFormer
on the DLCN lighting x state and MMPD Fitzpatrick cells, with the MAE ratio, the sigma ratio and their
quotient.
"""
import pandas as pd
from pathlib import Path

OUT_DIR = Path("results/day9_cross_backbone")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CONFIGS = [
    ("PhysNet", "DLCN", "results/day3_dlcn_full/dlcn_full_98p_n20.tidy_v2.csv", ["light", "state"]),
    ("PhysNet", "MMPD", "results/day5_mmpd/mmpd_full_n20.tidy.csv",            ["skin"]),
    ("TSCAN",   "DLCN", "results/day8_tscan/dlcn_full_tscan_n20.tidy.csv",     ["light", "state"]),
    ("TSCAN",   "MMPD", "results/day8_tscan/mmpd_full_tscan_n20.tidy.csv",     ["skin"]),
]

per_cell_rows, summary_rows = [], []

for backbone, dataset, path, cell_cols in CONFIGS:
    df = pd.read_csv(path)
    n_subj, n_trials = df["pid"].nunique(), len(df)
    print(f"[{backbone:7s} {dataset}] {n_trials} trials, {n_subj} subjects, cells={cell_cols}")

    cell = df.groupby(cell_cols).agg(
        n_trials=("abs_err", "size"),
        mae=("abs_err", "mean"),
        sigma=("unc_mean", "mean"),
    ).reset_index()
    cell["backbone"] = backbone
    cell["dataset"]  = dataset
    cell["cell"]     = cell[cell_cols].astype(str).agg("_".join, axis=1)
    per_cell_rows.append(cell)

    # MAE extremes (definitional anchor)
    i_best, i_worst = cell["mae"].idxmin(), cell["mae"].idxmax()
    mae_best,  mae_worst  = cell.loc[i_best, "mae"],   cell.loc[i_worst, "mae"]
    mae_ratio = mae_worst / mae_best

    # σ extremes (= σ at MAE_best vs σ at MAE_worst cell)
    sig_at_best,  sig_at_worst  = cell.loc[i_best, "sigma"], cell.loc[i_worst, "sigma"]
    sig_ratio_extremes = sig_at_worst / sig_at_best
    rho_extremes = mae_ratio / sig_ratio_extremes

    # σ global range (= σ.max anywhere / σ.min anywhere)
    sig_global_min = cell["sigma"].min()
    sig_global_max = cell["sigma"].max()
    sig_ratio_global = sig_global_max / sig_global_min
    rho_global = mae_ratio / sig_ratio_global

    summary_rows.append({
        "backbone": backbone, "dataset": dataset,
        "n_cells": len(cell), "n_trials": n_trials, "n_subjects": n_subj,
        "mae_agg":   df["abs_err"].mean(),
        "sigma_agg": df["unc_mean"].mean(),
        "best_cell": cell.loc[i_best, "cell"],
        "worst_cell": cell.loc[i_worst, "cell"],
        "mae_best": mae_best, "mae_worst": mae_worst, "mae_ratio": mae_ratio,
        "sig_at_best": sig_at_best, "sig_at_worst": sig_at_worst,
        "sig_global_min": sig_global_min, "sig_global_max": sig_global_max,
        "sig_ratio_extremes": sig_ratio_extremes,
        "sig_ratio_global":   sig_ratio_global,
        "rho_extremes": rho_extremes,
        "rho_global":   rho_global,
    })

df_per_cell = pd.concat(per_cell_rows, ignore_index=True)
df_summary  = pd.DataFrame(summary_rows)
df_per_cell.to_csv(OUT_DIR / "cross_backbone_per_cell.csv", index=False)
df_summary.to_csv(OUT_DIR / "cross_backbone_summary.csv", index=False)

# console — per-cell
print()
print("=" * 100)
print("PER-CELL")
print("=" * 100)
for (bb, ds), grp in df_per_cell.groupby(["backbone", "dataset"], sort=False):
    print(f"\n[{bb}/{ds}]")
    print(grp[["cell", "n_trials", "mae", "sigma"]].sort_values("mae")
              .to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# console — summary with both ratios
print()
print("=" * 100)
print("SUMMARY (both σ-ratio definitions)")
print("=" * 100)
cols = ["backbone", "dataset", "n_trials", "n_subjects",
        "mae_agg", "sigma_agg",
        "mae_ratio", "sig_ratio_global", "rho_global",
        "sig_ratio_extremes", "rho_extremes"]
print(df_summary[cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))

# LaTeX — uses global sigma-range definition as primary
print()
print("=" * 100)
print("LATEX TABLE (using global σ range, global sigma range)")
print("=" * 100)
lines = [
    r"\begin{table*}[t]",
    r"\centering",
    r"\caption{Cross-backbone $\sigma$-blindness diagnostics. Per-cell MAE (BPM) and mean MC dropout uncertainty $\bar{\sigma}$ on DLCN (per light $\times$ state, 8 cells) and MMPD (per Fitzpatrick skin tone, 4 cells). Spread ratios are $\text{MAE}_\text{max}/\text{MAE}_\text{min}$ and $\sigma_\text{max}/\sigma_\text{min}$ across cells. Informativeness $\rho = (\text{MAE ratio})/(\sigma\text{ ratio})$; $\rho \gg 1$ indicates $\sigma$ fails to track error. TSCAN's higher native dropout (p=0.25/0.5 vs PhysNet's p=0.1) does not improve, and arguably worsens, $\sigma$-informativeness.}",
    r"\label{tab:cross_backbone_sigma_blindness}",
    r"\begin{tabular}{l l c c c c c c}",
    r"\toprule",
    r"Backbone & Dataset & MAE & $\bar{\sigma}$ & MAE$_\text{min}$$\to$MAE$_\text{max}$ & MAE ratio & $\sigma$ ratio & $\rho$ \\",
    r"\midrule",
]
for _, r in df_summary.iterrows():
    lines.append(
        f"{r['backbone']} & {r['dataset']} & "
        f"{r['mae_agg']:.2f} & {r['sigma_agg']:.3f} & "
        f"{r['mae_best']:.2f}$\\to${r['mae_worst']:.2f} & "
        f"{r['mae_ratio']:.2f}$\\times$ & {r['sig_ratio_global']:.2f}$\\times$ & "
        f"\\textbf{{{r['rho_global']:.2f}}} \\\\"
    )
lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
latex_str = "\n".join(lines)
print(latex_str)
(OUT_DIR / "cross_backbone_table.tex").write_text(latex_str, encoding="utf-8")

print()
print(f"Saved: {OUT_DIR / 'cross_backbone_per_cell.csv'}")
print(f"Saved: {OUT_DIR / 'cross_backbone_summary.csv'}")
print(f"Saved: {OUT_DIR / 'cross_backbone_table.tex'}")
