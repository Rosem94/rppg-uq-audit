#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Trial-consistent uncertainty (Eq. 3).

The default proxy averages chunk-level waveform dispersion. sigma_cons passes every stochastic pass through
the same trial-level spectral HR pipeline as the point estimate and takes the standard deviation of the
resulting HR values, in BPM. The two proxies are compared per cell and per trial (Sec. IV-C4, IV-E2, IV-F).
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

SIGMA_COLS = ["unc_mean", "unc_consistent"]


def spread_ratio(values):
    """max/min over strictly positive, finite cell means."""
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v) & (v > 0)]
    return v.max() / v.min() if len(v) > 1 else float("nan")


def report(name, df, cell_cols):
    print("=" * 72)
    print(name)
    print(f"  trials = {len(df)}   cells = {cell_cols}   "
          f"n_cells = {df.groupby(cell_cols).ngroups}")
    print("=" * 72)

    cols = [c for c in SIGMA_COLS if c in df.columns]
    if "unc_consistent" not in df.columns:
        print("  [!] 'unc_consistent' not in CSV -- re-run the patched "
              "analyze_*.py first.\n")

    # Per-cell table: MAE and both sigma definitions side by side.
    agg = {"n": ("abs_err", "size"), "mae": ("abs_err", "mean")}
    for c in cols:
        agg[c] = (c, "mean")
    g = df.groupby(cell_cols).agg(**agg).reset_index()
    print(g.round(4).to_string(index=False))
    print()

    # Diagnostics per sigma definition.
    for c in cols:
        gc = df.groupby(cell_cols).agg(mae=("abs_err", "mean"),
                                       sig=(c, "mean"))
        mae_r = spread_ratio(gc["mae"].values)
        sig_r = spread_ratio(gc["sig"].values)
        rho = mae_r / sig_r if np.isfinite(sig_r) and sig_r > 0 else float("nan")
        r_p, p_p = pearsonr(df[c], df["abs_err"])
        r_s, p_s = spearmanr(df[c], df["abs_err"])
        print(f"  [{c}]")
        print(f"     per-cell MAE ratio   : {mae_r:6.2f}x")
        print(f"     per-cell sigma ratio : {sig_r:6.2f}x")
        print(f"     informativeness rho  : {rho:6.2f}   "
              f"(= MAE ratio / sigma ratio; larger = blinder)")
        print(f"     trial sigma~|err|    : Pearson r={r_p:+.3f} (p={p_p:.2g})"
              f"   Spearman r={r_s:+.3f} (p={p_s:.2g})")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dlcn", type=Path, required=True,
                    help="DLCN tidy CSV regenerated with the patched script")
    ap.add_argument("--mmpd", type=Path, required=True,
                    help="MMPD tidy CSV regenerated with the patched script")
    args = ap.parse_args()

    dlcn = pd.read_csv(args.dlcn)
    mmpd = pd.read_csv(args.mmpd)

    report("DLCN  (UBFC-trained PhysNet + MC dropout)", dlcn, ["light", "state"])
    report("MMPD  (skin-tone cells -- fairness axis)", mmpd, ["skin"])
    report("MMPD  (light x state grid)", mmpd, ["light", "state"])

    print("=" * 72)
    print("READING THE RESULT")
    print("-" * 72)
    print("If unc_consistent gives a sigma ratio and rho close to unc_mean,")
    print("sigma-blindness is robust to the sigma definition: it is not an")
    print("artifact of the chunk/trial timescale mismatch (comment 3), and the")
    print("ablation strengthens the paper. If unc_consistent tracks error")
    print("markedly better, the per-cell numbers in section IV-B need revising.")
    print("=" * 72)


if __name__ == "__main__":
    main()