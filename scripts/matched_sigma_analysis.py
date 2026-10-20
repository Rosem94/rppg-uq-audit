import pandas as pd
import numpy as np
from scipy.stats import spearmanr
from pathlib import Path

OUT = Path("results/day_r1_matched")
SEEDS = [0, 1, 2, 3, 4]

def load(dataset):
    dfs = []
    for s in SEEDS:
        if dataset == "dlcn":
            f = OUT / f"dlcn_seed{s}.tidy.csv"
            keys = ["pid", "trial", "light", "state"]
        else:
            f = OUT / f"mmpd_seed{s}" / "mmpd_full_n20.tidy.csv"
            keys = ["pid", "light", "state", "exercise", "skin"]
        d = pd.read_csv(f)
        d["seed"] = s
        dfs.append((d, keys))
    return dfs

def analyze(dataset, cell_cols_list):
    dfs = load(dataset)
    keys = dfs[0][1]
    base = dfs[0][0][keys + ["hr_true"]].copy()

    # sanity: identical trial sets and labels across seeds
    for d, _ in dfs[1:]:
        m = base.merge(d[keys + ["hr_true"]], on=keys, suffixes=("", "_b"))
        assert len(m) == len(base) == len(d), f"{dataset}: trial sets differ across seeds"
        assert np.allclose(m.hr_true, m.hr_true_b), f"{dataset}: hr_true mismatch"

    wide = base.copy()
    for d, _ in dfs:
        s = d["seed"].iloc[0]
        cols = d[keys + ["hr_pred", "unc_consistent", "unc_mean", "abs_err"]].rename(
            columns={"hr_pred": f"pred_{s}", "unc_consistent": f"cons_{s}",
                     "unc_mean": f"chunk_{s}", "abs_err": f"err_{s}"})
        wide = wide.merge(cols, on=keys)

    P = wide[[f"pred_{s}" for s in SEEDS]].values
    wide["ens_pred"] = P.mean(axis=1)
    wide["sigma_E"] = P.std(axis=1, ddof=1)
    wide["ens_err"] = (wide.ens_pred - wide.hr_true).abs()
    wide["sig_cons"] = wide[[f"cons_{s}" for s in SEEDS]].mean(axis=1)
    wide["sig_chunk"] = wide[[f"chunk_{s}" for s in SEEDS]].mean(axis=1)

    print(f"\n{'='*70}\n{dataset.upper()}  (n_trials = {len(wide)})\n{'='*70}")

    print("\nPer-seed MAE (BPM):")
    for s in SEEDS:
        print(f"  seed{s}: {wide[f'err_{s}'].mean():6.2f}")
    print(f"  ensemble-mean prediction MAE: {wide.ens_err.mean():6.2f}")

    print("\n--- MATCHED per-trial Spearman rho(sigma, |err|), same target = ensemble |err| ---")
    for name, col in [("sigma_E (5 seeds)", "sigma_E"),
                      ("sigma_MCD trial-consistent (seed-avg)", "sig_cons"),
                      ("sigma_MCD chunk-avg [old, confounded]", "sig_chunk")]:
        rho, p = spearmanr(wide[col], wide.ens_err)
        print(f"  {name:42s} rho = {rho:+.3f}  (p = {p:.2e})")

    print("\n--- Per-seed self-consistency: rho(cons_s, err_s), each seed vs own error ---")
    rhos = []
    for s in SEEDS:
        rho, _ = spearmanr(wide[f"cons_{s}"], wide[f"err_{s}"])
        rhos.append(rho)
        print(f"  seed{s}: rho = {rho:+.3f}")
    print(f"  mean over seeds: {np.mean(rhos):+.3f}")

    for cell_cols in cell_cols_list:
        g = wide.groupby(cell_cols).agg(
            mae=("ens_err", "mean"), sE=("sigma_E", "mean"),
            sC=("sig_cons", "mean"), sK=("sig_chunk", "mean"),
            n=("ens_err", "size"))
        def spread(v):
            v = v[np.isfinite(v) & (v > 0)]
            return v.max() / v.min()
        print(f"\n--- Cell-level spread, cells = {cell_cols} ({len(g)} cells) ---")
        print(f"  MAE spread:              {spread(g.mae.values):5.2f}x")
        print(f"  sigma_E spread:          {spread(g.sE.values):5.2f}x")
        print(f"  sigma_cons spread:       {spread(g.sC.values):5.2f}x")
        print(f"  sigma_chunk spread [old]:{spread(g.sK.values):5.2f}x")
        rho_E, _ = spearmanr(g.mae, g.sE)
        rho_C, _ = spearmanr(g.mae, g.sC)
        print(f"  cell-rank rho(MAE, sigma_E)    = {rho_E:+.3f}")
        print(f"  cell-rank rho(MAE, sigma_cons) = {rho_C:+.3f}")

    keep = keys + ["hr_true", "ens_pred", "sigma_E", "ens_err", "sig_cons", "sig_chunk"] + \
           [c for c in wide.columns if c.startswith(("pred_", "cons_", "err_"))]
    wide[keep].to_csv(OUT / f"matched_{dataset}.csv", index=False)
    print(f"\nSaved: {OUT / f'matched_{dataset}.csv'}")

analyze("dlcn", [["light", "state"]])
analyze("mmpd", [["skin"], ["skin", "state"], ["light", "state"]])
