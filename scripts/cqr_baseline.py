"""Conformalized quantile regression baseline (Sec. IV-C5).

Gradient-boosted 5% and 95% quantile regressors on per-trial features (point prediction, dropout summaries,
sigma_cons, chunk count, one-hot cell metadata). The calibration cohort is split subject-disjointly in half
for fitting and conformalization; ten splits, alpha = 0.10. The backbone is not retrained.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor


def conformal_quantile(scores, alpha):
    n = len(scores)
    q_level = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(scores, q_level, method="higher")


NUMERIC_CANDIDATES = ["hr_pred", "unc_mean", "unc_max", "unc_p90",
                      "unc_consistent", "n_chunks"]


def build_X(df, num_cols, meta_cols, meta_levels):
    parts = [df[num_cols].to_numpy(dtype=float)]
    for c in meta_cols:
        levels = meta_levels[c]
        onehot = np.zeros((len(df), len(levels)))
        for j, v in enumerate(levels):
            onehot[:, j] = (df[c] == v).astype(float)
        parts.append(onehot)
    return np.hstack(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    ap.add_argument("--alpha", type=float, default=0.1)
    ap.add_argument("--n_cal_subjects", type=int, default=30)
    ap.add_argument("--n_seeds", type=int, default=10)
    ap.add_argument("--cell_cols", nargs="+", default=["light", "state"],
                    help="Cells for CONDITIONAL coverage reporting.")
    ap.add_argument("--meta_cols", nargs="+", default=None,
                    help="Categorical metadata one-hot encoded as features "
                         "(default: same as --cell_cols).")
    ap.add_argument("--n_estimators", type=int, default=300)
    ap.add_argument("--max_depth", type=int, default=3)
    ap.add_argument("--learning_rate", type=float, default=0.05)
    ap.add_argument("--min_samples_leaf", type=int, default=10)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv)
    meta_cols = args.meta_cols or args.cell_cols
    num_cols = [c for c in NUMERIC_CANDIDATES if c in df.columns]
    meta_levels = {c: sorted(df[c].unique()) for c in meta_cols}
    all_subjects = sorted(df.pid.unique())
    lo_q, hi_q = args.alpha / 2, 1 - args.alpha / 2
    print(f"Loaded {len(df)} trials, {len(all_subjects)} subjects; "
          f"numeric feats {num_cols}; one-hot {meta_cols}; "
          f"report cells on {args.cell_cols}")

    cell_records, marg_records = [], []

    for seed in range(args.n_seeds):
        rng = np.random.default_rng(seed)
        shuffled = rng.permutation(all_subjects)
        cal_subjects = shuffled[: args.n_cal_subjects].tolist()
        test_subjects = shuffled[args.n_cal_subjects:].tolist()
        # subject-disjoint train/conformalization split inside the cal cohort
        rng2 = np.random.default_rng(1000 + seed)
        cal_perm = rng2.permutation(sorted(cal_subjects))
        n_tr = len(cal_perm) // 2
        tr_subjects = set(cal_perm[:n_tr].tolist())
        cf_subjects = set(cal_perm[n_tr:].tolist())

        d_tr = df[df.pid.isin(tr_subjects)]
        d_cf = df[df.pid.isin(cf_subjects)]
        d_te = df[df.pid.isin(test_subjects)].copy()

        X_tr = build_X(d_tr, num_cols, meta_cols, meta_levels)
        X_cf = build_X(d_cf, num_cols, meta_cols, meta_levels)
        X_te = build_X(d_te, num_cols, meta_cols, meta_levels)
        y_tr, y_cf, y_te = (d["hr_true"].to_numpy()
                            for d in (d_tr, d_cf, d_te))

        kw = dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
                  learning_rate=args.learning_rate,
                  min_samples_leaf=args.min_samples_leaf, random_state=seed)
        m_lo = GradientBoostingRegressor(loss="quantile", alpha=lo_q, **kw)
        m_hi = GradientBoostingRegressor(loss="quantile", alpha=hi_q, **kw)
        m_lo.fit(X_tr, y_tr)
        m_hi.fit(X_tr, y_tr)

        # conformalization (Romano et al. 2019)
        lo_cf, hi_cf = m_lo.predict(X_cf), m_hi.predict(X_cf)
        E = np.maximum(lo_cf - y_cf, y_cf - hi_cf)
        Q = conformal_quantile(E, args.alpha)

        lo_te = m_lo.predict(X_te) - Q
        hi_te = m_hi.predict(X_te) + Q
        d_te["_covered"] = ((y_te >= lo_te) & (y_te <= hi_te)).astype(int)
        d_te["_width"] = hi_te - lo_te

        marg_records.append({
            "seed": seed, "coverage": d_te["_covered"].mean(),
            "mean_width": d_te["_width"].mean(),
            "median_width": d_te["_width"].median(),
            "Q": Q, "n_train": len(d_tr), "n_conf": len(d_cf),
            "n_test": len(d_te),
        })
        for key, grp in d_te.groupby(args.cell_cols):
            if not isinstance(key, tuple):
                key = (key,)
            cell_records.append({
                **{c: v for c, v in zip(args.cell_cols, key)},
                "seed": seed, "n": len(grp),
                "coverage": grp["_covered"].mean(),
                "mean_width": grp["_width"].mean(),
            })

    marg_df = pd.DataFrame(marg_records)
    cell_df = pd.DataFrame(cell_records)
    marg_df.to_csv(args.out_dir / "cqr_marginal_seeds.csv", index=False)
    cell_df.to_csv(args.out_dir / "cqr_cells_seeds.csv", index=False)

    lines = [f"csv: {args.csv}",
             f"features: {num_cols} + one-hot{meta_cols}  "
             f"cells: {args.cell_cols}  alpha: {args.alpha}  "
             f"n_cal_subjects: {args.n_cal_subjects} (half train / half conf)  "
             f"seeds: {args.n_seeds}",
             f"GBR: n_estimators={args.n_estimators} max_depth={args.max_depth} "
             f"lr={args.learning_rate} min_samples_leaf={args.min_samples_leaf}",
             "",
             f"marginal coverage = {marg_df.coverage.mean():.3f} "
             f"+/- {marg_df.coverage.sem():.3f}",
             f"mean width  = {marg_df.mean_width.mean():.1f} BPM   "
             f"median width = {marg_df.median_width.mean():.1f} BPM",
             ""]
    agg = cell_df.groupby(args.cell_cols).agg(
        coverage=("coverage", "mean"), width=("mean_width", "mean"),
        n=("n", "mean"))
    worst = agg.coverage.idxmin()
    lines.append(f"worst-cell coverage = {agg.coverage.min():.3f} @ {worst}   "
                 f"range [{agg.coverage.min():.3f}, {agg.coverage.max():.3f}]  "
                 f"cells outside +/-0.05: "
                 f"{((agg.coverage - (1 - args.alpha)).abs() > 0.05).sum()}/{len(agg)}")
    for key, row in agg.iterrows():
        if not isinstance(key, tuple):
            key = (key,)
        kstr = ",".join(f"{c}={v}" for c, v in zip(args.cell_cols, key))
        lines.append(f"    {kstr:24s} cov={row.coverage:.3f} "
                     f"width={row.width:6.1f}  n~{row.n:.0f}")
    out = args.out_dir / "cqr_summary.txt"
    out.write_text("\n".join(lines))
    print("\n".join(lines))
    print(f"Saved {out}")


if __name__ == "__main__":
    main()
