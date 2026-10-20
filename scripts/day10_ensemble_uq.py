#!/usr/bin/env python
"""Five-seed deep ensemble versus MC dropout (Sec. IV-F): per-cell and per-trial comparison of the ensemble
standard deviation with the single-model dropout proxies.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr


def _stat(res):
    return (res.statistic, res.pvalue) if hasattr(res, 'statistic') else (res[0], res[1])


def load_dataset(dataset, in_dir, n_seeds=5):
    dfs = []
    for s in range(n_seeds):
        if dataset == "dlcn":
            csv_path = in_dir / f"dlcn_seed{s}_n20.tidy.csv"
        else:
            csv_path = in_dir / f"mmpd_seed{s}" / "mmpd_full_n20.tidy.csv"
        df = pd.read_csv(csv_path)
        df["seed"] = s
        dfs.append(df)
    long_df = pd.concat(dfs, ignore_index=True)

    if dataset == "dlcn":
        trial_keys = ["pid", "trial"]
        cell_keys = ["light", "state"]
        meta_cols = ["light", "state"]
    else:
        trial_keys = ["pid", "light", "state", "exercise"]
        cell_keys = ["skin", "state"]
        meta_cols = ["light", "state", "exercise", "skin", "gender",
                     "glasser", "hair_cover", "makeup"]
    return long_df, trial_keys, cell_keys, meta_cols


def aggregate_per_trial(long_df, trial_keys, meta_cols):
    agg = long_df.groupby(trial_keys).agg(
        hr_ensemble=("hr_pred", "mean"),
        sigma_ensemble=("hr_pred", "std"),
        hr_true=("hr_true", "first"),
        sigma_mcd_avg=("unc_mean", "mean"),
        abs_err_single_avg=("abs_err", "mean"),
    ).reset_index()
    agg["abs_err_ensemble"] = (agg["hr_ensemble"] - agg["hr_true"]).abs()
    extra = [c for c in meta_cols if c not in trial_keys]
    if extra:
        meta_first = long_df.groupby(trial_keys)[extra].first().reset_index()
        agg = agg.merge(meta_first, on=trial_keys, how="left")
    return agg


def report_dataset(dataset, in_dir, out_dir, n_seeds=5):
    print(f"\n{'='*60}")
    print(f"  {dataset.upper()} ENSEMBLE UQ")
    print(f"{'='*60}")
    long_df, trial_keys, cell_keys, meta_cols = load_dataset(dataset, in_dir, n_seeds)
    n_trials = long_df.groupby(trial_keys).ngroups
    print(f"Trials: {n_trials}, seeds: {n_seeds}, total rows: {len(long_df)}")

    per_trial = aggregate_per_trial(long_df, trial_keys, meta_cols)
    out_csv = out_dir / f"{dataset}_ensemble_per_trial.csv"
    per_trial.to_csv(out_csv, index=False)
    print(f"Saved per-trial: {out_csv} ({len(per_trial)} rows)")

    sE = per_trial['sigma_ensemble'].mean()
    sM = per_trial['sigma_mcd_avg'].mean()
    mE = per_trial['abs_err_ensemble'].mean()
    mS = per_trial['abs_err_single_avg'].mean()
    print(f"\n--- HEADLINE ---")
    print(f"Mean sigma_MCD (avg over seeds):  {sM:.4f}")
    print(f"Mean sigma_E (across-seed std):   {sE:.4f}")
    print(f"sigma_E / sigma_MCD ratio:        {sE/sM:.2f}x")
    print(f"Ensemble MAE:                     {mE:.3f} BPM")
    print(f"Single-model MAE (avg of 5):      {mS:.3f} BPM")
    print(f"Ensemble benefit:                 {mS-mE:+.3f} BPM ({(mS-mE)/mS*100:+.1f}%)")

    pr, pp = _stat(pearsonr(per_trial['sigma_ensemble'], per_trial['sigma_mcd_avg']))
    sr, sp = _stat(spearmanr(per_trial['sigma_ensemble'], per_trial['sigma_mcd_avg']))
    print(f"\n--- sigma_E vs sigma_MCD (per-trial, n={len(per_trial)}) ---")
    print(f"Pearson r  = {pr:+.3f} (p={pp:.2e})")
    print(f"Spearman r = {sr:+.3f} (p={sp:.2e})")

    err = per_trial['abs_err_ensemble']
    rE, pE = _stat(spearmanr(per_trial['sigma_ensemble'], err))
    rM, pM = _stat(spearmanr(per_trial['sigma_mcd_avg'], err))
    print(f"\n--- Calibration (sigma vs |error|, Spearman) ---")
    print(f"sigma_E    vs |err_ensemble|: r = {rE:+.3f} (p={pE:.2e})")
    print(f"sigma_MCD  vs |err_ensemble|: r = {rM:+.3f} (p={pM:.2e})")

    print(f"\n--- Per-cell sigma_E vs sigma_MCD (cells: {cell_keys}) ---")
    cell_agg = per_trial.groupby(cell_keys).agg(
        n=("hr_ensemble", "size"),
        mae_ens=("abs_err_ensemble", "mean"),
        sigma_E=("sigma_ensemble", "mean"),
        sigma_MCD=("sigma_mcd_avg", "mean"),
    ).reset_index()
    cell_agg["ratio_E_MCD"] = cell_agg["sigma_E"] / cell_agg["sigma_MCD"]
    print(cell_agg.to_string(index=False))

    rC, pC = _stat(spearmanr(cell_agg["sigma_E"], cell_agg["sigma_MCD"]))
    rCE, pCE = _stat(spearmanr(cell_agg["sigma_E"], cell_agg["mae_ens"]))
    rCM, pCM = _stat(spearmanr(cell_agg["sigma_MCD"], cell_agg["mae_ens"]))
    print(f"\nCell-level sigma_E vs sigma_MCD: r = {rC:+.3f} (p={pC:.2e})")
    print(f"Cell-level sigma_E vs cell_MAE:  r = {rCE:+.3f} (p={pCE:.2e})")
    print(f"Cell-level sigma_MCD vs cell_MAE:r = {rCM:+.3f} (p={pCM:.2e})")

    cell_csv = out_dir / f"{dataset}_ensemble_per_cell.csv"
    cell_agg.to_csv(cell_csv, index=False)
    print(f"Saved per-cell: {cell_csv}")

    return {
        "dataset": dataset, "n_trials": n_trials, "n_seeds": n_seeds,
        "sigma_E_mean": sE, "sigma_MCD_mean": sM, "ratio_E_over_MCD": sE/sM,
        "mae_ensemble": mE, "mae_single_avg": mS,
        "sigma_corr_pearson": pr, "sigma_corr_spearman": sr,
        "sigma_E_calib_spearman": rE, "sigma_MCD_calib_spearman": rM,
        "cell_sigma_rank_spearman": rC,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_dir", type=Path, default=Path("results/day10_ensemble"))
    ap.add_argument("--out_dir", type=Path, default=Path("results/day10_ensemble/agg"))
    ap.add_argument("--n_seeds", type=int, default=5)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    summaries = []
    for ds in ["dlcn", "mmpd"]:
        summaries.append(report_dataset(ds, args.in_dir, args.out_dir, args.n_seeds))

    sdf = pd.DataFrame(summaries)
    spath = args.out_dir / "ensemble_uq_summary.csv"
    sdf.to_csv(spath, index=False)
    print(f"\n{'='*60}")
    print(f"  OVERALL SUMMARY -> {spath}")
    print(f"{'='*60}")
    print(sdf.to_string(index=False))


if __name__ == "__main__":
    main()
