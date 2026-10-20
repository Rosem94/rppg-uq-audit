"""Operating points for selective prediction (Table VIII).

Same splits, scores and criteria as selective_prediction.py. For each retention rho: MAE on retained trials,
missed failures (retained with |err| > 5 and > 10 BPM), false abstentions (abstained with |err| <= 5 BPM),
failure capture and per-group retention; mean over seeds.
"""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

RHOS = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
THR_LOW, THR_HIGH = 5.0, 10.0


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, method="higher")


def per_cell(df_cal, cell_cols, alpha):
    q, m = {}, {}
    for key, grp in df_cal.groupby(cell_cols):
        if not isinstance(key, tuple):
            key = (key,)
        q[key] = conformal_quantile(grp["abs_err"].values / grp["unc_mean"].values, alpha)
        m[key] = grp["abs_err"].mean()
    return q, m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_cal_subjects", type=int, required=True)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--cell_cols", nargs="+", required=True)
    ap.add_argument("--group_col", required=True,
                    help="column for per-group reporting, e.g. skin or light")
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(args.csv)
    subjects = sorted(df.pid.unique())
    groups = sorted(df[args.group_col].unique())
    print(f"Loaded {len(df)} trials, {len(subjects)} subjects; cells {args.cell_cols}; group {args.group_col} {groups}")

    rows, trial_rows = [], []
    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(subjects)
        cal = sorted(shuffled[: args.n_cal_subjects].tolist())
        test = sorted(shuffled[args.n_cal_subjects:].tolist())
        d_cal = df[df.pid.isin(cal)].copy()
        d_test = df[df.pid.isin(test)].copy()
        q_cell, mae_cell = per_cell(d_cal, args.cell_cols, args.alpha)
        cell_df = pd.DataFrame([{**dict(zip(args.cell_cols, k)), "q_hat_cell": q_cell[k],
                                 "mae_cell": mae_cell[k]} for k in q_cell])
        d_test = d_test.merge(cell_df, on=args.cell_cols, how="left")
        d_test["score_naive"] = d_test["unc_mean"]
        d_test["score_mondrian_cp"] = d_test["q_hat_cell"] * d_test["unc_mean"]
        d_test["score_cellmae_sigma"] = d_test["mae_cell"] * d_test["unc_mean"]
        n = len(d_test)
        n_fail_high = int((d_test.abs_err > THR_HIGH).sum())

        for crit in ["naive", "mondrian_cp", "cellmae_sigma"]:
            order = d_test.sort_values(f"score_{crit}", ascending=True).reset_index(drop=True)
            for rho in RHOS:
                k = max(int(np.ceil(rho * n)), 1)
                ret, abst = order.iloc[:k], order.iloc[k:]
                rec = {
                    "seed": seed, "criterion": crit, "rho": rho, "n_retained": k,
                    "mae_retained": ret.abs_err.mean(),
                    "missed_gt5_pct": 100 * (ret.abs_err > THR_LOW).mean(),
                    "missed_gt10_pct": 100 * (ret.abs_err > THR_HIGH).mean(),
                    "false_abst_pct": 100 * (abst.abs_err <= THR_LOW).mean() if len(abst) else np.nan,
                    "fail_capture_pct": 100 * (abst.abs_err > THR_HIGH).sum() / n_fail_high if n_fail_high else np.nan,
                }
                for g in groups:
                    tot = (order[args.group_col] == g).sum()
                    rg = ret[ret[args.group_col] == g]
                    rec[f"ret_{args.group_col}{g}_pct"] = 100 * len(rg) / tot if tot else np.nan
                    rec[f"mae_{args.group_col}{g}"] = rg.abs_err.mean() if len(rg) else np.nan
                    rec[f"missed_gt10_{args.group_col}{g}_pct"] = 100 * (rg.abs_err > THR_HIGH).mean() if len(rg) else np.nan
                rows.append(rec)
            order["rank_frac"] = (np.arange(n) + 1) / n
            trial_rows.append(order.assign(seed=seed, criterion=crit)[
                ["seed", "criterion", "pid", *list(dict.fromkeys(args.cell_cols + [args.group_col])), "abs_err",
                 f"score_{crit}", "rank_frac"]].rename(columns={f"score_{crit}": "score"}))

    res = pd.DataFrame(rows)
    res.to_csv(args.out_dir / "operating_point_seeds.csv", index=False)
    pd.concat(trial_rows).to_csv(args.out_dir / "operating_point_per_trial.csv", index=False)
    summ = res.drop(columns=["seed"]).groupby(["criterion", "rho"]).mean().round(2)
    summ.to_csv(args.out_dir / "operating_point_summary.csv")

    print("\n=== Sanity vs Table V (rho=0.5, MAE retained) ===")
    for crit in ["naive", "mondrian_cp", "cellmae_sigma"]:
        print(f"  {crit:14s}: {summ.loc[(crit, 0.5), 'mae_retained']:.2f} BPM")

    core = ["mae_retained", "missed_gt5_pct", "missed_gt10_pct", "false_abst_pct", "fail_capture_pct"]
    grp_cols = [c for c in summ.columns if c.startswith("ret_")]
    print("\n=== Operating points (mean over seeds) ===")
    print(summ[core + grp_cols].to_string())
    print(f"\nSaved to {args.out_dir}")


if __name__ == "__main__":
    main()
