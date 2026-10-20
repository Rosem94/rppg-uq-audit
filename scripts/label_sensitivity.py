"""Sensitivity of Mondrian-CP to cell labels at test time (Sec. V-c).

A fraction of test-trial labels is corrupted, coarsened (--coarse_map) or withheld while calibration keeps the
true labels; per-cell coverage and the selective-prediction benefit are reported over ten splits and five
draws.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, method="higher")


def cell_keys(df, cols):
    """List of tuple keys per row, robust to pandas groupby key conventions."""
    return list(zip(*[df[c] for c in cols]))


def as_tuple(key):
    return key if isinstance(key, tuple) else (key,)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_cal_subjects", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--n_corrupt_reps", type=int, default=5,
                    help="Independent corruption draws per split seed.")
    ap.add_argument("--fractions", nargs="+", type=float,
                    default=[0.0, 0.1, 0.2, 0.3])
    ap.add_argument("--sigma_col", type=str, default="unc_mean")
    ap.add_argument("--cell_cols", nargs="+", default=["light", "state"])
    ap.add_argument("--worst_cell", nargs="+", type=int, required=True,
                    help="True-label coordinates of the worst cell, matching --cell_cols.")
    ap.add_argument("--retention", type=float, default=0.5)
    ap.add_argument("--missing", action="store_true", help="Withhold the label of corrupted test trials (assign a value unseen in calibration) instead of relabelling; fallback quantile is max_r q_hat.")
    ap.add_argument("--coarse_map", type=str, default=None,
                    help="Scenario B: 'orig:merged,...' applied to a SINGLE cell col "
                         "on cal+test (e.g. '3:0,4:0,5:1,6:1' for two-level skin). "
                         "Fractions are ignored in this mode.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    missing = [c for c in args.cell_cols if c not in df.columns]
    if missing:
        raise ValueError(f"--cell_cols not in CSV: {missing}")
    if args.coarse_map and len(args.cell_cols) != 1:
        raise ValueError("--coarse_map supports a single cell column only")

    worst_cell = tuple(args.worst_cell)
    all_subjects = sorted(df.pid.unique())
    n_cells = df.groupby(args.cell_cols).ngroups
    print(f"Loaded {len(df)} trials, {len(all_subjects)} subjects, "
          f"{n_cells} cells on {args.cell_cols}, sigma={args.sigma_col}")

    coarse_map = None
    if args.coarse_map:
        coarse_map = {int(k): int(v) for k, v in
                      (p.split(":") for p in args.coarse_map.split(","))}
        print(f"Coarse map on {args.cell_cols[0]}: {coarse_map}")
        fractions = [None]           # single deterministic scenario
    else:
        fractions = args.fractions

    cell_records, sel_records = [], []

    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_subjects)
        cal_subjects = sorted(shuffled[: args.n_cal_subjects].tolist())
        test_subjects = sorted(shuffled[args.n_cal_subjects:].tolist())
        df_cal = df[df.pid.isin(cal_subjects)].copy()
        df_test = df[df.pid.isin(test_subjects)].copy()

        for f in fractions:
            reps = 1 if (f is None or f == 0.0) else args.n_corrupt_reps
            for rep in range(reps):
                cal = df_cal.copy()
                test = df_test.copy()

                if coarse_map is not None:
                    c = args.cell_cols[0]
                    cal["_lab"] = cal[c].map(coarse_map)
                    test["_lab"] = test[c].map(coarse_map)
                    lab_cols = ["_lab"]
                else:
                    # assigned labels start as true labels
                    for c in args.cell_cols:
                        cal[f"_lab_{c}"] = cal[c]
                        test[f"_lab_{c}"] = test[c]
                    lab_cols = [f"_lab_{c}" for c in args.cell_cols]
                    if f and f > 0:
                        rng_c = np.random.default_rng(
                            50_000 + 1000 * seed + 10 * rep + int(round(100 * f)))
                        n_corrupt = int(round(f * len(test)))
                        idx = rng_c.choice(test.index.values, size=n_corrupt,
                                           replace=False)
                        uniq = [sorted(df[c].unique()) for c in args.cell_cols]
                        for i in idx:
                            if args.missing:
                                new = tuple(-9 for _ in args.cell_cols)
                            else:
                              while True:
                                new = tuple(rng_c.choice(u) for u in uniq)
                                old = tuple(test.loc[i, f"_lab_{c}"]
                                            for c in args.cell_cols)
                                if new != old:
                                    break
                            for c, v in zip(args.cell_cols, new):
                                test.loc[i, f"_lab_{c}"] = v

                # per-cell quantities on calibration (assigned = true there,
                # or coarse in scenario B)
                sigma_cal = cal[args.sigma_col].values
                q_hat, mae_cell = {}, {}
                for key, grp in cal.groupby(lab_cols):
                    scores = grp["abs_err"].values / grp[args.sigma_col].values
                    q_hat[as_tuple(key)] = conformal_quantile(scores, args.alpha)
                    mae_cell[as_tuple(key)] = grp["abs_err"].mean()

                keys_test = cell_keys(test, lab_cols)
                test["_q"] = [q_hat.get(k, np.nan) for k in keys_test]
                test["_mae_cal"] = [mae_cell.get(k, np.nan) for k in keys_test]
                # Trials whose assigned label was unseen in calibration:
                #  - for COVERAGE they fall back to the global LA quantile
                #    (a deployed system must still emit an interval);
                #  - for ABSTENTION ranking they are treated as maximally
                #    uncertain (sorted last), matching selective_prediction.py
                #    where unmapped scores are NaN and sort to the end.
                g_scores = cal["abs_err"].values / cal[args.sigma_col].values
                g_q = conformal_quantile(g_scores, args.alpha)
                test["_q_rank"] = test["_q"].fillna(np.inf)
                test["_mae_rank"] = test["_mae_cal"].fillna(np.inf)
                fb_q = max(q_hat.values()) if args.missing else g_q
                fb_m = max(mae_cell.values()) if args.missing else cal["abs_err"].mean()
                test["_q"] = test["_q"].fillna(fb_q)
                test["_mae_cal"] = test["_mae_cal"].fillna(fb_m)

                half = test["_q"] * test[args.sigma_col]
                test["_covered"] = (test["abs_err"] <= half).astype(int)
                test["_width"] = 2 * half

                # ---- coverage per TRUE (fine) cell ----
                for key, grp in test.groupby(args.cell_cols):
                    key = as_tuple(key)
                    cell_records.append({
                        **{c: v for c, v in zip(args.cell_cols, key)},
                        "seed": seed, "rep": rep,
                        "fraction": -1.0 if f is None else f,
                        "scenario": "coarse" if coarse_map else "relabel",
                        "n": len(grp),
                        "coverage": grp["_covered"].mean(),
                        "mean_width": grp["_width"].mean(),
                    })
                cell_records.append({
                    **{c: -1 for c in args.cell_cols},
                    "seed": seed, "rep": rep,
                    "fraction": -1.0 if f is None else f,
                    "scenario": "coarse" if coarse_map else "relabel",
                    "n": len(test),
                    "coverage": test["_covered"].mean(),
                    "mean_width": test["_width"].mean(),
                })

                # ---- selective prediction at rho ----
                test["_s_naive"] = test[args.sigma_col]
                test["_s_mond"] = test["_q_rank"] * test[args.sigma_col]
                test["_s_cmae"] = test["_mae_rank"] * test[args.sigma_col]
                n = len(test)
                k = max(int(np.ceil(args.retention * n)), 1)
                wmask = pd.Series(True, index=test.index)
                for c, v in zip(args.cell_cols, worst_cell):
                    wmask &= (test[c] == v)
                n_worst = int(wmask.sum())
                rec = {"seed": seed, "rep": rep,
                       "fraction": -1.0 if f is None else f,
                       "scenario": "coarse" if coarse_map else "relabel"}
                for name, col in [("naive", "_s_naive"), ("mondrian", "_s_mond"),
                                  ("cellmae_sigma", "_s_cmae")]:
                    order = test.sort_values(col, ascending=True).iloc[:k]
                    rec[f"mae_{name}"] = order["abs_err"].mean()
                    kept_worst = int(wmask.loc[order.index].sum())
                    rec[f"worst_ret_{name}"] = (kept_worst / n_worst
                                                if n_worst else np.nan)
                sel_records.append(rec)

    cell_df = pd.DataFrame(cell_records)
    sel_df = pd.DataFrame(sel_records)
    cell_df.to_csv(args.out_dir / "label_sens_cells.csv", index=False)
    sel_df.to_csv(args.out_dir / "label_sens_selective.csv", index=False)

    # ---------- summary ----------
    lines = []
    lines.append(f"csv: {args.csv}")
    lines.append(f"cell_cols: {args.cell_cols}  sigma: {args.sigma_col}  "
                 f"alpha: {args.alpha}  n_cal_subjects: {args.n_cal_subjects}  "
                 f"seeds: {args.n_seeds}  reps: {args.n_corrupt_reps}")
    if coarse_map:
        lines.append(f"coarse_map: {coarse_map}")
    lines.append("")
    marg_mask = (cell_df[args.cell_cols] == -1).all(axis=1)
    for f in sorted(cell_df.fraction.unique()):
        sub = cell_df[cell_df.fraction == f]
        tag = "coarse(2-level)" if f == -1.0 else f"f={f:.1f}"
        marg = sub[marg_mask.reindex(sub.index, fill_value=False)]
        cells = sub[~marg_mask.reindex(sub.index, fill_value=False)]
        agg = cells.groupby(args.cell_cols).agg(
            coverage=("coverage", "mean"), width=("mean_width", "mean"))
        worst_key = agg.coverage.idxmin()
        lines.append(f"[{tag}] marginal={marg.coverage.mean():.3f}  "
                     f"width={marg.mean_width.mean():.1f}  "
                     f"worst-cell={agg.coverage.min():.3f} @ {worst_key}  "
                     f"cells<0.85: {(agg.coverage < 0.85).sum()}/{len(agg)}")
        for key, row in agg.iterrows():
            if not isinstance(key, tuple):
                key = (key,)
            kstr = ",".join(f"{c}={v}" for c, v in zip(args.cell_cols, key))
            lines.append(f"    {kstr:24s} cov={row.coverage:.3f} width={row.width:6.1f}")
        s = sel_df[sel_df.fraction == f]
        lines.append(f"    selective @rho={args.retention}: "
                     f"naive MAE {s.mae_naive.mean():.2f} | "
                     f"Mondrian {s.mae_mondrian.mean():.2f} "
                     f"(benefit {s.mae_naive.mean() - s.mae_mondrian.mean():+.2f}) | "
                     f"cellMAExsigma {s.mae_cellmae_sigma.mean():.2f} "
                     f"(benefit {s.mae_naive.mean() - s.mae_cellmae_sigma.mean():+.2f})")
        lines.append(f"    worst-cell retention: naive {s.worst_ret_naive.mean():.3f} | "
                     f"Mondrian {s.worst_ret_mondrian.mean():.3f} | "
                     f"cellMAExsigma {s.worst_ret_cellmae_sigma.mean():.3f}")
        lines.append("")
    out_txt = args.out_dir / "label_sens_summary.txt"
    out_txt.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved {out_txt}")


if __name__ == "__main__":
    main()
