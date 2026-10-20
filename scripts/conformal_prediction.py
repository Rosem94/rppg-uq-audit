"""Split conformal prediction on trial-level rPPG results.

Ten subject-disjoint calibration/test splits (seed k: np.random.default_rng(k).permutation(sorted(pid))).
Global and locally adaptive scores; marginal and cell-conditional coverage for the cells in --cell_cols.
--sigma_col selects the uncertainty proxy and --sigma_floor its lower bound; --shuffle_cells gives the
permutation control. Writes per-seed, per-cell and per-trial coverage files and a text summary.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

# Pretty names for known DLCN columns. Fallback to f"{col}={val}" for unknown.
LIGHT_NAMES = {1: "FI&FP", 2: "VI&FP", 3: "FI&VP", 4: "VI&VP"}
STATE_NAMES = {1: "rest", 2: "exercise"}


def split_conformal(df_cal, df_test, alpha=0.1, score_type="locally_adaptive", sigma_col="unc_mean", sigma_floor=0.0):
    """Returns df_test with added columns: ci_low, ci_high, ci_width, covered.

    score_type:
      - 'global':            s = |err|              (Romano et al. 2019 baseline)
      - 'locally_adaptive':  s = |err| / unc_mean   (Lei et al. 2018 LACP variant)
    """
    if score_type == "global":
        s_cal = df_cal["abs_err"].values
        u_cal = np.ones(len(df_cal))
        u_test = np.ones(len(df_test))
    elif score_type == "locally_adaptive":
        u_cal = np.maximum(df_cal[sigma_col].values, sigma_floor)
        u_test = np.maximum(df_test[sigma_col].values, sigma_floor)
        s_cal = df_cal["abs_err"].values / u_cal
    else:
        raise ValueError(f"Unknown score_type: {score_type}")

    n_cal = len(df_cal)
    # Conformal quantile with finite-sample correction
    q_level = np.ceil((n_cal + 1) * (1 - alpha)) / n_cal
    q_level = min(q_level, 1.0)
    q_hat = np.quantile(s_cal, q_level, method="higher")

    df_out = df_test.copy()
    half_width = q_hat * u_test
    df_out["ci_low"] = df_out["hr_pred"] - half_width
    df_out["ci_high"] = df_out["hr_pred"] + half_width
    df_out["ci_width"] = 2 * half_width
    df_out["covered"] = (
        (df_out["hr_true"] >= df_out["ci_low"])
        & (df_out["hr_true"] <= df_out["ci_high"])
    ).astype(int)
    df_out["q_hat"] = q_hat
    return df_out


def evaluate_split(df, cal_subjects, test_subjects, alpha, score_type, cell_cols, sigma_col="unc_mean", sigma_floor=0.0, shuffle_cells=False, shuffle_seed=0):
    df_cal = df[df.pid.isin(cal_subjects)].copy()
    df_test = df[df.pid.isin(test_subjects)].copy()
    if shuffle_cells:
        # Null control: permute cell labels among test trials, preserving group
        # sizes. Coverage per (permuted) cell then reflects only the sampling
        # variability of grouping n_test trials into these bin sizes.
        rng_null = np.random.default_rng(10_000 + shuffle_seed)
        for c in cell_cols:
            df_test[c] = rng_null.permutation(df_test[c].values)
    res = split_conformal(df_cal, df_test, alpha=alpha, score_type=score_type, sigma_col=sigma_col, sigma_floor=sigma_floor)

    out = {
        "score_type": score_type,
        "n_cal_subjects": len(cal_subjects),
        "n_test_subjects": len(test_subjects),
        "n_cal_trials": len(df_cal),
        "n_test_trials": len(df_test),
        "q_hat": res.q_hat.iloc[0],
        "marginal_coverage": res.covered.mean(),
        "mean_width": res.ci_width.mean(),
        "median_width": res.ci_width.median(),
    }

    # Conditional coverage per cell — auto-detect cell values via groupby
    cond_records = []
    for cell_key, cell_df in res.groupby(cell_cols):
        if not isinstance(cell_key, tuple):
            cell_key = (cell_key,)
        rec = {col: val for col, val in zip(cell_cols, cell_key)}
        rec.update({
            "n": len(cell_df),
            "coverage": cell_df.covered.mean(),
            "mean_width": cell_df.ci_width.mean(),
            "mean_abs_err": cell_df.abs_err.mean(),
        })
        cond_records.append(rec)
    return out, cond_records, res


def format_cell_desc(cell_cols, row):
    """Pretty-print a cell row using known name dicts where available."""
    parts = []
    for c in cell_cols:
        v = int(row[c])
        if c == "light" and v in LIGHT_NAMES:
            parts.append(f"L{v} {LIGHT_NAMES[v]}")
        elif c == "state" and v in STATE_NAMES:
            parts.append(f"{STATE_NAMES[v]}")
        else:
            parts.append(f"{c}={v}")
    return " | ".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path,
                    help="trial-level tidy CSV (e.g. dlcn / mmpd / ubfc)")
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1, help="Miscoverage rate")
    ap.add_argument("--n_cal_subjects", type=int, default=30,
                    help="Number of calibration subjects (rest go to test)")
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--sigma_col", type=str, default="unc_mean",
                    help="Column used as the per-trial uncertainty proxy u(x).")
    ap.add_argument("--shuffle_cells", action="store_true",
                    help="Null control: permute cell labels within the test set, "
                         "preserving group sizes, before computing conditional coverage.")
    ap.add_argument("--sigma_floor", type=float, default=0.0,
                    help="Lower bound on u(x); BPM-scale sigmas need >0 because "
                         "spectral quantization can make all passes identical.")
    ap.add_argument("--cell_cols", nargs="+", default=["light", "state"],
                    help="Columns defining cells for conditional coverage. "
                         "Default: ['light', 'state'] (DLCN/MMPD regime). "
                         "Use ['state'] for motion-only coarsening, "
                         "['skin'] for skin-tone fairness, etc.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)

    # Validate cell_cols exist in CSV
    missing = [c for c in args.cell_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--cell_cols entries not in CSV: {missing}. "
                         f"Available: {list(df.columns)}")

    all_subjects = sorted(df.pid.unique())
    print(f"Loaded {len(df)} trials, {len(all_subjects)} subjects")
    print(f"Target coverage: {(1 - args.alpha) * 100:.0f}%")
    print(f"Calibration: {args.n_cal_subjects} subjects, "
          f"Test: {len(all_subjects) - args.n_cal_subjects} subjects")
    print(f"Cell cols: {args.cell_cols}  ({df.groupby(args.cell_cols).ngroups} cells)")
    print(f"Seeds: {args.n_seeds}\n")

    seed_records = []

    per_trial_records = []
    cond_records_all = []

    for seed in tqdm(range(args.n_seeds), desc="CP seeds", ncols=80):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_subjects)
        cal_subjects = sorted(shuffled[: args.n_cal_subjects].tolist())
        test_subjects = sorted(shuffled[args.n_cal_subjects:].tolist())

        for score_type in ["global", "locally_adaptive"]:
            summary, cond, _per_trial = evaluate_split(
                df, cal_subjects, test_subjects,
                alpha=args.alpha, score_type=score_type, sigma_col=args.sigma_col, sigma_floor=args.sigma_floor, shuffle_cells=args.shuffle_cells, shuffle_seed=seed,
                cell_cols=args.cell_cols,
            )
            summary["seed"] = seed
            seed_records.append(summary)
            _pt = _per_trial.copy()
            _pt["seed"] = seed
            _pt["score_type"] = score_type
            per_trial_records.append(_pt)
            for c in cond:
                c["seed"] = seed
                c["score_type"] = score_type
                cond_records_all.append(c)

    if per_trial_records:
        pd.concat(per_trial_records, ignore_index=True).to_csv(
            args.out_dir / 'cp_per_trial_coverage.csv', index=False)
    seed_df = pd.DataFrame(seed_records)
    cond_df = pd.DataFrame(cond_records_all)

    seed_df.to_csv(args.out_dir / "cp_results_seeds.csv", index=False)
    cond_df.to_csv(args.out_dir / "cp_conditional_seeds.csv", index=False)
    print(f"Saved per-seed results to {args.out_dir}")

    # Summary across seeds
    print("\n" + "=" * 70)
    print(f"SUMMARY across {args.n_seeds} seeds (mean +/- SE)")
    print("=" * 70)
    for score_type in ["global", "locally_adaptive"]:
        sub = seed_df[seed_df.score_type == score_type]
        print(f"\nScore type: {score_type}")
        print(f"  q_hat:               {sub.q_hat.mean():.3f} +/- {sub.q_hat.std() / np.sqrt(len(sub)):.3f}")
        print(f"  Marginal coverage:   {sub.marginal_coverage.mean():.3f} "
              f"+/- {sub.marginal_coverage.std() / np.sqrt(len(sub)):.3f}  "
              f"(target {1 - args.alpha:.2f})")
        print(f"  Mean interval width: {sub.mean_width.mean():.2f} "
              f"+/- {sub.mean_width.std() / np.sqrt(len(sub)):.2f}  BPM")
        print(f"  Median width:        {sub.median_width.mean():.2f}  BPM")

    # Conditional coverage table (avg across seeds)
    print("\n" + "=" * 70)
    cell_label = " x ".join(args.cell_cols)
    print(f"CONDITIONAL COVERAGE per ({cell_label}), avg across seeds")
    print(f"(target {1 - args.alpha:.2f}; >0.05 deviation -> conditional miscalibration)")
    print("=" * 70)
    for score_type in ["global", "locally_adaptive"]:
        print(f"\n  {score_type}:")
        agg = (cond_df[cond_df.score_type == score_type]
               .groupby(args.cell_cols)
               .agg(coverage=("coverage", "mean"),
                    width=("mean_width", "mean"),
                    err=("mean_abs_err", "mean"),
                    n=("n", "mean"))
               .reset_index())
        for _, row in agg.iterrows():
            desc = format_cell_desc(args.cell_cols, row)
            cov_dev = abs(row.coverage - (1 - args.alpha))
            flag = "  <-- miscalibrated" if cov_dev > 0.05 else ""
            print(f"    {desc:32s}: cov={row.coverage:.3f}  width={row.width:6.2f}  "
                  f"err_mean={row.err:5.2f}  n={row.n:5.1f}{flag}")

    # Save text summary
    summary_txt = args.out_dir / "cp_summary.txt"
    with open(summary_txt, "w") as f:
        f.write(f"cell_cols: {args.cell_cols}\n\n")
        f.write(seed_df.groupby("score_type").agg(
            mean=("marginal_coverage", "mean"),
            se=("marginal_coverage", lambda x: x.std() / np.sqrt(len(x))),
            mean_width=("mean_width", "mean"),
        ).to_string())
        f.write("\n\nConditional coverage by score_type:\n\n")
        f.write(cond_df.groupby(["score_type"] + args.cell_cols).agg(
            coverage=("coverage", "mean"),
            width=("mean_width", "mean"),
            err=("mean_abs_err", "mean"),
        ).to_string())
    print(f"\nSaved {summary_txt}")


if __name__ == "__main__":
    main()