"""Selective prediction risk-coverage curves (Sec. IV-D).

Criteria: naive ranking by the dropout sigma; LA-CP (rank-equivalent to naive, kept as a check);
Mondrian-CP (cell quantile x sigma, cells from --cell_cols); cell-MAE x sigma and cell-MAE lookup baselines.
For retention 0.5-1.0 the MAE on all retained trials and on the cell given by --worst_cell is reported.
Uses the splits of conformal_prediction.py and writes selective_seeds.csv.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm


def conformal_quantile(scores, alpha):
    """Finite-sample-corrected conformal quantile."""
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, method="higher")


def mondrian_q_hat(df_cal, cell_cols, alpha=0.1):
    """Per-cell q_hat on cal set; LA score: |err| / unc_mean."""
    out = {}
    for cell_key, grp in df_cal.groupby(cell_cols):
        scores = grp["abs_err"].values / grp["unc_mean"].values
        if not isinstance(cell_key, tuple):
            cell_key = (cell_key,)
        out[cell_key] = conformal_quantile(scores, alpha)
    return out


def cell_mae(df_cal, cell_cols):
    """Per-cell mean absolute error on the calibration set.

    Plain-statistics analog of mondrian_q_hat: the per-cell conformal
    quantile q_hat[cell] is replaced by the per-cell mean |err|. Used to
    test whether the conformal quantile is needed for the SELECTIVE ranking,
    or whether a simple historical cell-MAE lookup ranks trials as well.
    """
    out = {}
    for cell_key, grp in df_cal.groupby(cell_cols):
        if not isinstance(cell_key, tuple):
            cell_key = (cell_key,)
        out[cell_key] = grp["abs_err"].mean()
    return out


def global_la_q_hat(df_cal, alpha=0.1):
    scores = df_cal["abs_err"].values / df_cal["unc_mean"].values
    return conformal_quantile(scores, alpha)


def worst_cell_mask(df, cell_cols, worst_cell):
    """Boolean mask: True where row matches the worst-cell coords."""
    mask = pd.Series(True, index=df.index)
    for col, val in zip(cell_cols, worst_cell):
        mask &= (df[col] == val)
    return mask


def selective_curve(df_test, score_col, retentions, cell_cols, worst_cell):
    """For each retention r, MAE on top-r smallest-score subset.

    score_col may be a single column name or a list of column names
    (used for the cell-MAE lookup baseline: primary key + tiebreak).
    Returns list[dict] with overall + worst-cell metrics.
    """
    df_sorted = df_test.sort_values(score_col, ascending=True).reset_index(drop=True)
    n = len(df_sorted)
    full_worst_mask = worst_cell_mask(df_sorted, cell_cols, worst_cell)
    n_worst_total = int(full_worst_mask.sum())
    records = []
    for r in retentions:
        k = max(int(np.ceil(r * n)), 1)
        retained = df_sorted.iloc[:k]
        retained_worst_mask = full_worst_mask.iloc[:k]
        worst = retained[retained_worst_mask]
        records.append({
            "retention": r,
            "n_retained": k,
            "mae_overall": retained["abs_err"].mean(),
            "mae_worst_cell": worst["abs_err"].mean() if len(worst) > 0 else np.nan,
            "n_worst_retained": len(worst),
            "worst_cell_retention": len(worst) / n_worst_total if n_worst_total > 0 else np.nan,
        })
    return records


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_cal_subjects", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--cell_cols", nargs="+", default=["light", "state"],
                    help="Columns defining Mondrian cells. Default: ['light','state']. "
                         "Use ['state'] for motion-only coarsening, etc.")
    ap.add_argument("--worst_cell", nargs="+", type=int, required=True,
                    help="Values defining the worst-coverage cell, in order matching "
                         "--cell_cols. Example: --cell_cols light state --worst_cell 4 2 "
                         "(DLCN L4 VI&VP exercise). For motion-only: --cell_cols state "
                         "--worst_cell 4 (MMPD walking).")
    ap.add_argument("--fig_suffix", default="",
                    help="Suffix for output figure filename (e.g. '_motion_only')")
    args = ap.parse_args()

    if len(args.worst_cell) != len(args.cell_cols):
        raise ValueError(
            f"--worst_cell has {len(args.worst_cell)} values but "
            f"--cell_cols has {len(args.cell_cols)} columns. They must match."
        )
    worst_cell = tuple(args.worst_cell)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    missing = [c for c in args.cell_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--cell_cols entries not in CSV: {missing}. "
                         f"Available: {list(df.columns)}")

    all_subjects = sorted(df.pid.unique())
    n_cells = df.groupby(args.cell_cols).ngroups
    print(f"Loaded {len(df)} trials, {len(all_subjects)} subjects, alpha={args.alpha}")
    print(f"Cell cols: {args.cell_cols}  ({n_cells} cells)")
    print(f"Worst cell: {dict(zip(args.cell_cols, worst_cell))}")
    print(f"Cal/test split ratio: {args.n_cal_subjects} / "
          f"{len(all_subjects) - args.n_cal_subjects}  "
          f"(N_cal/N_cells = {args.n_cal_subjects / n_cells:.1f})")

    retentions = np.round(np.arange(0.5, 1.0 + 1e-9, 0.05), 2).tolist()

    all_records = []
    for seed in tqdm(range(args.n_seeds), desc="seeds", ncols=80):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_subjects)
        cal_subjects = sorted(shuffled[: args.n_cal_subjects].tolist())
        test_subjects = sorted(shuffled[args.n_cal_subjects:].tolist())

        df_cal = df[df.pid.isin(cal_subjects)].copy()
        df_test = df[df.pid.isin(test_subjects)].copy()

        q_hat_cell = mondrian_q_hat(df_cal, args.cell_cols, alpha=args.alpha)
        q_hat_global = global_la_q_hat(df_cal, alpha=args.alpha)
        mae_cell = cell_mae(df_cal, args.cell_cols)

        # Merge cell q_hat AND per-cell calibration MAE into test (vectorized)
        cell_records = []
        for cell_key, q in q_hat_cell.items():
            rec = {col: val for col, val in zip(args.cell_cols, cell_key)}
            rec["q_hat_cell"] = q
            rec["mae_cell"] = mae_cell.get(cell_key, np.nan)
            cell_records.append(rec)
        cell_df = pd.DataFrame(cell_records)
        df_test = df_test.merge(cell_df, on=args.cell_cols, how="left")

        # Diagnostic for sample-size hypothesis
        if seed == 0:
            print(f"\nSeed 0 q_hat / MAE_cal per cell (cell-heterogeneity check):")
            for cell_key, q in sorted(q_hat_cell.items()):
                print(f"  {dict(zip(args.cell_cols, cell_key))}: "
                      f"q_hat = {q:.4f}   MAE_cal = {mae_cell.get(cell_key, float('nan')):.3f}")

        df_test["score_naive"] = df_test["unc_mean"]
        df_test["score_la_cp"] = q_hat_global * df_test["unc_mean"]
        df_test["score_mondrian"] = df_test["q_hat_cell"] * df_test["unc_mean"]
        # Baselines: per-cell calibration MAE in place
        # of the conformal quantile.
        df_test["score_cellmae_sigma"] = df_test["mae_cell"] * df_test["unc_mean"]
        df_test["score_cellmae_only"] = df_test["mae_cell"]

        for criterion, col in [("naive", "score_naive"),
                               ("la_cp", "score_la_cp"),
                               ("mondrian_cp", "score_mondrian"),
                               ("cellmae_sigma", "score_cellmae_sigma"),
                               ("cellmae_lookup",
                                ["score_cellmae_only", "unc_mean"])]:
            for rec in selective_curve(df_test, col, retentions,
                                       args.cell_cols, worst_cell):
                rec["seed"] = seed
                rec["criterion"] = criterion
                all_records.append(rec)

    out_df = pd.DataFrame(all_records)
    out_csv = args.out_dir / f"selective_seeds{args.fig_suffix}.csv"
    out_df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")

    # ===== Sanity check: naive == la_cp =====
    naive = out_df[out_df.criterion == "naive"].set_index(["seed", "retention"])["mae_overall"]
    la_cp = out_df[out_df.criterion == "la_cp"].set_index(["seed", "retention"])["mae_overall"]
    diff = (naive - la_cp).abs().max()
    print(f"\nSanity: max |MAE_naive - MAE_LA-CP| = {diff:.2e} "
          f"({'OK, rank-equivalent' if diff < 1e-9 else 'WARNING: differ!'})")

    # ===== Diagnostic: worst-cell n_retained =====
    worst_str = ", ".join(f"{c}={v}" for c, v in zip(args.cell_cols, worst_cell))
    print(f"\nWorst-cell ({worst_str}) n_retained — min across seeds:")
    diag = (out_df.groupby(["criterion", "retention"])["n_worst_retained"]
            .min().unstack("retention"))
    print(diag.to_string())

    # ===== Headline comparison at retention rho = 0.5 (for section IV-D) =====
    print(f"\n=== At retention rho=0.5 (mean over {args.n_seeds} seeds) ===")
    r05 = out_df[np.isclose(out_df.retention, 0.5)]
    summ = (r05.groupby("criterion")
                .agg(mae_overall=("mae_overall", "mean"),
                     worst_cell_retention=("worst_cell_retention", "mean")))
    for crit in ["naive", "mondrian_cp", "cellmae_sigma", "cellmae_lookup"]:
        if crit in summ.index:
            row = summ.loc[crit]
            print(f"  {crit:16s}: MAE_overall = {row.mae_overall:7.3f} BPM   "
                  f"worst-cell retention = {row.worst_cell_retention:.3f}")
    if {"naive", "mondrian_cp", "cellmae_sigma"}.issubset(summ.index):
        base = summ.loc["naive", "mae_overall"]
        print(f"  MAE benefit vs naive:  "
              f"Mondrian-CP = {base - summ.loc['mondrian_cp','mae_overall']:+.3f} BPM   "
              f"cell-MAE x sigma = {base - summ.loc['cellmae_sigma','mae_overall']:+.3f} BPM   "
              f"cell-MAE lookup = {base - summ.loc['cellmae_lookup','mae_overall']:+.3f} BPM")

    # ===== Plot (naive vs Mondrian-CP, unchanged) =====
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.3), sharex=True)
    plot_criteria = [
        ("naive", "Naive (raw σ) ≡ LA-CP", "#1f77b4"),
        ("mondrian_cp", "Mondrian-CP (cell-aware)", "#d62728"),
    ]
    panels = [
        ("mae_overall", "(A) Overall MAE on retained", "MAE (BPM)"),
        ("worst_cell_retention",
         f"(B) Worst-cell retention\n({worst_str})",
         "Fraction of worst-cell trials kept"),
    ]
    for ax, (metric, title, ylabel) in zip(axes, panels):
        for crit, label, color in plot_criteria:
            sub = out_df[out_df.criterion == crit].dropna(subset=[metric])
            agg = sub.groupby("retention")[metric].agg(["mean", "std", "count"]).reset_index()
            se = agg["std"] / np.sqrt(agg["count"].clip(lower=1))
            ax.plot(agg.retention, agg["mean"], color=color, label=label,
                    linewidth=2, marker="o", markersize=5)
            ax.fill_between(agg.retention, agg["mean"] - se, agg["mean"] + se,
                            alpha=0.20, color=color)
        ax.set_xlabel("Overall retention rate")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(alpha=0.3)
        ax.legend(loc="best", framealpha=0.95)

    # Panel B: diagonal reference (random selection baseline)
    axes[1].plot([0.5, 1.0], [0.5, 1.0], "k--", alpha=0.5,
                 linewidth=1.2, label="Proportional (random selection)")
    axes[1].set_ylim(-0.02, 1.05)
    axes[1].legend(loc="upper left", framealpha=0.95, fontsize=9)

    plt.tight_layout()
    for ext in ("png", "pdf"):
        out_fig = args.out_dir / f"fig4_selective{args.fig_suffix}.{ext}"
        plt.savefig(out_fig, dpi=200, bbox_inches="tight")
    print(f"\nSaved fig4_selective{args.fig_suffix}.png + .pdf to {args.out_dir}")


if __name__ == "__main__":
    main()