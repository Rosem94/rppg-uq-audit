"""DLCN hetero tidy CSV: aleatoric (log_var) + epistemic (MC dropout) side by side.

Extends analyze_dlcn_full_v2.py to the PhysNetHetero pickle. Adds alea_mean
(= sqrt(mean_t(exp(log_var))) per chunk, averaged over chunks) next to the
epistemic columns, and reports whether sigma_A tracks cross-dataset error /
regime structure better than sigma_MCD.

Output: results/day5_dlcn/dlcn_full_hetero_n20.tidy.csv
"""
import argparse
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
from scipy.signal import butter
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr
from tqdm import tqdm

FS = 30.0
LOW_PASS = 0.6
HIGH_PASS = 3.3
N_ALEA = 20
RNG = np.random.default_rng(0)
LIGHT_NAMES = {1: "FI&FP", 2: "VI&FP", 3: "FI&VP", 4: "VI&VP"}


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(signal, lambda_value=100):
    n = signal.shape[0]
    ones = np.ones(n); minus_twos = -2 * np.ones(n)
    D = spdiags(np.array([ones, minus_twos, ones]),
                np.array([0, 1, 2]), n - 2, n).tocsc()
    A = speye(n, format="csc") + (lambda_value ** 2) * (D.T @ D)
    return signal - spsolve(A, signal)


def calc_hr_toolbox(signal, fs=FS, diff_flag=True):
    if diff_flag:
        signal = _detrend(np.cumsum(signal), 100)
    else:
        signal = _detrend(signal, 100)
    [b, a] = butter(1, [LOW_PASS / fs * 2, HIGH_PASS / fs * 2], btype="bandpass")
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    sig_2d = np.expand_dims(signal, 0)
    N = _next_power_of_2(sig_2d.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(sig_2d, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f_ppg >= LOW_PASS) & (f_ppg <= HIGH_PASS))
    return np.take(np.take(f_ppg, fmask),
                   np.argmax(np.take(pxx_ppg, fmask), 0))[0] * 60.0


def parse_subj(subj_id):
    m = re.match(r"P(\d+)_T(\d+)", str(subj_id))
    if not m:
        return None
    p, t = int(m.group(1)), int(m.group(2))
    light = ((t - 1) % 4) + 1
    state = 1 if t <= 4 else 2
    return p, t, light, state


def chunk_alea_sigma(payload):
    """Per-chunk aleatoric sigma = sqrt(mean_t(exp(log_var)))."""
    lv = np.asarray(payload["log_var"]).astype(np.float64).flatten()
    return float(np.sqrt(np.mean(np.exp(lv))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    records = []
    for subj_id, chunks in tqdm(data.items(), desc="Per-trial HR", ncols=80):
        parsed = parse_subj(subj_id)
        if parsed is None:
            continue
        pid, trial, light, state = parsed
        sorted_chunks = sorted(chunks.items(), key=lambda x: int(x[0]))
        full_pred = np.concatenate([
            np.asarray(p["mean"]).flatten() for _, p in sorted_chunks
        ])
        full_label = np.concatenate([
            np.asarray(p["label"]).flatten() for _, p in sorted_chunks
        ])
        chunk_uncs = np.array([
            np.asarray(p["std"]).flatten().mean() for _, p in sorted_chunks
        ])
        alea_chunks = np.array([chunk_alea_sigma(p) for _, p in sorted_chunks])
        n_chunks = len(sorted_chunks)
        hr_pred = calc_hr_toolbox(full_pred, diff_flag=True)
        hr_true = calc_hr_toolbox(full_label, diff_flag=True)
        sample_stack = np.stack([
            np.asarray(p["samples"]) for _, p in sorted_chunks
        ], axis=1)                               # (N_passes, n_chunks, T)
        hr_per_pass = np.array([
            calc_hr_toolbox(sample_stack[k].reshape(-1), diff_flag=True)
            for k in range(sample_stack.shape[0])
        ])
        logvar_full = np.concatenate([np.asarray(p["log_var"]).astype(np.float64).flatten() for _, p in sorted_chunks])
        assert logvar_full.shape[0] == full_pred.shape[0]
        alea_sd = np.exp(0.5 * logvar_full)
        eps = RNG.standard_normal((N_ALEA, full_pred.shape[0]))
        hr_alea = np.array([calc_hr_toolbox(full_pred + alea_sd * eps[k], diff_flag=True) for k in range(N_ALEA)])
        flat_samples = sample_stack.reshape(sample_stack.shape[0], -1)
        n_tot = min(N_ALEA, flat_samples.shape[0])
        hr_tot = np.array([calc_hr_toolbox(flat_samples[k] + alea_sd * eps[k], diff_flag=True) for k in range(n_tot)])
        records.append({
            "pid": pid, "trial": trial, "light": light, "state": state,
            "n_chunks": n_chunks,
            "hr_pred": hr_pred, "hr_true": hr_true,
            "abs_err": abs(hr_pred - hr_true),
            "unc_mean": chunk_uncs.mean(),
            "unc_max": chunk_uncs.max(),
            "unc_consistent": float(hr_per_pass.std()),
            "alea_consistent": float(hr_alea.std()),
            "tot_consistent": float(hr_tot.std()),
            "alea_mean": alea_chunks.mean(),
            "alea_max": alea_chunks.max(),
        })

    df = pd.DataFrame(records)
    print(f"\nPer-trial records: {len(df)}")
    print(f"Unique participants: {df.pid.nunique()}\n")

    print("=== Overall (per-trial, toolbox-style HR) ===")
    print(f"  HR MAE:          {df.abs_err.mean():.2f}  BPM")
    print(f"  sigma_MCD range: [{df.unc_mean.min():.4f}, {df.unc_mean.max():.4f}]")
    print(f"  sigma_A   range: [{df.alea_mean.min():.4f}, {df.alea_mean.max():.4f}]\n")

    print("=== Per (lighting x state): MAE vs sigma_MCD vs sigma_A ===")
    cell = (df.groupby(["light", "state"])
              .agg(n=("abs_err", "size"),
                   mae=("abs_err", "mean"),
                   sigma_MCD=("unc_mean", "mean"),
                   sigma_A=("alea_mean", "mean"),
                   sigma_cons=("unc_consistent", "mean"),
                   sigma_A_bpm=("alea_consistent", "mean"))
              .round(4).reset_index())
    cell["light_name"] = cell.light.map(LIGHT_NAMES)
    cell["state_name"] = cell.state.map({1: "rest", 2: "exercise"})
    print(cell[["light_name", "state_name", "n", "mae", "sigma_MCD", "sigma_A", "sigma_cons", "sigma_A_bpm"]].to_string(index=False))

    mae_ratio = cell["mae"].max() / cell["mae"].min()
    mcd_ratio = cell["sigma_MCD"].max() / cell["sigma_MCD"].min()
    a_ratio   = cell["sigma_A"].max() / cell["sigma_A"].min()
    print(f"\n  Across-cell MAE spread:        {mae_ratio:.2f}x")
    print(f"  Across-cell sigma_MCD spread:  {mcd_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/mcd_ratio:.2f}; ~1=tracks, >1=under-scales)")
    print(f"  Across-cell sigma_A spread:    {a_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/a_ratio:.2f})")
    c_ratio = cell["sigma_cons"].max() / cell["sigma_cons"].min()
    ab_ratio = cell["sigma_A_bpm"].max() / cell["sigma_A_bpm"].min()
    print(f"  Across-cell sigma_cons spread: {c_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/c_ratio:.2f})")
    print(f"  Across-cell sigma_A_bpm spread:{ab_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/ab_ratio:.2f})")

    print(f"\n=== Trial-level rank correlation with |error| (n={len(df)}) ===")
    for col, label in [("unc_mean", "sigma_MCD (chunk-avg)"),
                       ("unc_consistent", "sigma_MCD (trial-consistent)"),
                       ("alea_mean", "sigma_A (aleatoric)"),
                       ("alea_consistent", "sigma_A (trial-consistent, BPM)"),
                       ("tot_consistent", "sigma_E+A (trial-consistent, BPM)")]:
        rho, p = spearmanr(df[col], df.abs_err)
        print(f"  {label:30s}: Spearman rho={rho:+.3f}  p={p:.2g}")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"\nSaved {args.out_csv}")


if __name__ == "__main__":
    main()
