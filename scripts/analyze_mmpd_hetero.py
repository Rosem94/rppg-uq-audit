"""MMPD hetero tidy CSV: aleatoric (log_var) + epistemic (MC dropout) side by side.

Extends analyze_mmpd_full.py to the PhysNetHetero pickle, which adds a per-chunk
'log_var' aleatoric head output. Adds an aleatoric sigma column (alea_mean) next
to the epistemic sigma columns (unc_mean = sigma_MCD chunk-avg, unc_consistent =
trial-consistent sigma), so we can ask whether the aleatoric head recovers the
demographic / regime structure that MC dropout is blind to.

sigma_A per chunk = sqrt(mean_t(exp(log_var)))  -- waveform-amplitude units, the
same footing as unc_mean (also a per-timestep waveform std), so the comparison is
apples-to-apples via rank correlation and relative cell ratios.

Output: results/day5_mmpd/mmpd_full_hetero_n20.tidy.csv
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

KEY_RE = re.compile(
    r'(subject\d+)_L(\d+)_MO(\d+)_E(\d+)_S(\d+)_GE(\d+)_GL(\d+)_H(\d+)_MA(\d+)'
)
LIGHT_NAMES  = {1: "LED-low", 2: "LED-high", 3: "Incandescent", 4: "Nature"}
MOTION_NAMES = {1: "Stationary", 2: "Rotation", 3: "Talking", 4: "Walking"}


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


def parse_key(key):
    m = KEY_RE.match(key)
    if not m:
        return None
    pid, L, MO, E, S, GE, GL, H, MA = m.groups()
    return {
        "pid": pid, "light": int(L), "state": int(MO),  # state := motion
        "exercise": int(E), "skin": int(S),
        "gender": int(GE), "glasser": int(GL),
        "hair_cover": int(H), "makeup": int(MA),
    }


def chunk_alea_sigma(payload):
    """Per-chunk aleatoric sigma = sqrt(mean_t(exp(log_var)))."""
    lv = np.asarray(payload["log_var"]).astype(np.float64).flatten()
    return float(np.sqrt(np.mean(np.exp(lv))))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trial-entries from {args.pkl}")

    records = []
    skipped = 0
    for sid in tqdm(sorted(data.keys()), desc="trials"):
        meta = parse_key(sid)
        if meta is None:
            skipped += 1
            continue

        chunks = data[sid]
        sorted_items = sorted(chunks.items(), key=lambda kv: kv[0])
        means  = np.concatenate([np.asarray(c['mean']).flatten()  for _, c in sorted_items])
        labels = np.concatenate([np.asarray(c['label']).flatten() for _, c in sorted_items])

        hr_pred = calc_hr_toolbox(means,  fs=FS, diff_flag=True)
        hr_true = calc_hr_toolbox(labels, fs=FS, diff_flag=True)

        sigma_chunks = np.array([np.asarray(c['std']).flatten().mean() for _, c in sorted_items])
        alea_chunks  = np.array([chunk_alea_sigma(c) for _, c in sorted_items])

        sample_stack = np.stack([
            np.asarray(c['samples']) for _, c in sorted_items
        ], axis=1)                               # (N_passes, n_chunks, T)
        hr_per_pass = np.array([
            calc_hr_toolbox(sample_stack[k].reshape(-1), fs=FS, diff_flag=True)
            for k in range(sample_stack.shape[0])
        ])
        logvar_full = np.concatenate([np.asarray(c['log_var']).astype(np.float64).flatten() for _, c in sorted_items])
        assert logvar_full.shape[0] == means.shape[0]
        alea_sd = np.exp(0.5 * logvar_full)
        eps = RNG.standard_normal((N_ALEA, means.shape[0]))
        hr_alea = np.array([calc_hr_toolbox(means + alea_sd * eps[k], fs=FS, diff_flag=True) for k in range(N_ALEA)])
        flat_samples = sample_stack.reshape(sample_stack.shape[0], -1)
        n_tot = min(N_ALEA, flat_samples.shape[0])
        hr_tot = np.array([calc_hr_toolbox(flat_samples[k] + alea_sd * eps[k], fs=FS, diff_flag=True) for k in range(n_tot)])

        records.append({
            **meta,
            "n_chunks": len(chunks),
            "hr_pred": hr_pred,
            "hr_true": hr_true,
            "abs_err": abs(hr_pred - hr_true),
            "unc_mean": sigma_chunks.mean(),
            "unc_max":  sigma_chunks.max(),
            "unc_consistent": float(hr_per_pass.std()),
            "alea_consistent": float(hr_alea.std()),
            "tot_consistent": float(hr_tot.std()),
            "alea_mean": alea_chunks.mean(),
            "alea_max":  alea_chunks.max(),
        })

    if skipped > 0:
        print(f"Warning: {skipped} keys did not match parser")

    df = pd.DataFrame(records)
    print(f"\n=== MMPD cross-dataset (UBFC-trained PhysNetHetero) ===")
    print(f"Total trials:    {len(df)}")
    print(f"Unique subjects: {df.pid.nunique()}")
    print(f"MAE:             {df.abs_err.mean():.3f} BPM")
    print(f"sigma_MCD range: [{df.unc_mean.min():.4f}, {df.unc_mean.max():.4f}]")
    print(f"sigma_A   range: [{df.alea_mean.min():.4f}, {df.alea_mean.max():.4f}]")

    print(f"\n=== Per-skin (Fitzpatrick): does sigma_A track the demographic gap? ===")
    skin = (df.groupby("skin")
              .agg(n=("abs_err", "size"),
                   mae=("abs_err", "mean"),
                   sigma_MCD=("unc_mean", "mean"),
                   sigma_A=("alea_mean", "mean"),
                   sigma_cons=("unc_consistent", "mean"),
                   sigma_A_bpm=("alea_consistent", "mean"))
              .round(4))
    print(skin.to_string())

    sk = skin.sort_index()
    mae_ratio = sk["mae"].max() / sk["mae"].min()
    mcd_ratio = sk["sigma_MCD"].max() / sk["sigma_MCD"].min()
    a_ratio   = sk["sigma_A"].max() / sk["sigma_A"].min()
    print(f"\n  Across-skin MAE spread:        {mae_ratio:.2f}x")
    print(f"  Across-skin sigma_MCD spread:  {mcd_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/mcd_ratio:.2f}; ~1=tracks, >1=under-scales)")
    print(f"  Across-skin sigma_A spread:    {a_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/a_ratio:.2f})")
    c_ratio = sk["sigma_cons"].max() / sk["sigma_cons"].min()
    ab_ratio = sk["sigma_A_bpm"].max() / sk["sigma_A_bpm"].min()
    print(f"  Across-skin sigma_cons spread: {c_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/c_ratio:.2f})")
    print(f"  Across-skin sigma_A_bpm spread:{ab_ratio:.2f}x   (MAE/sigma ratio {mae_ratio/ab_ratio:.2f})")

    print(f"\n=== Per-(light, motion) cell ===")
    cell = (df.groupby(["light", "state"])
              .agg(n=("abs_err", "size"),
                   mae=("abs_err", "mean"),
                   sigma_MCD=("unc_mean", "mean"),
                   sigma_A=("alea_mean", "mean"),
                   sigma_cons=("unc_consistent", "mean"),
                   sigma_A_bpm=("alea_consistent", "mean"))
              .round(3).reset_index())
    cell["light_name"]  = cell.light.map(LIGHT_NAMES)
    cell["motion_name"] = cell.state.map(MOTION_NAMES)
    print(cell[["light_name", "motion_name", "n", "mae", "sigma_MCD", "sigma_A", "sigma_cons", "sigma_A_bpm"]].to_string(index=False))

    print(f"\n=== Trial-level rank correlation with |error| (n={len(df)}) ===")
    for col, label in [("unc_mean", "sigma_MCD (chunk-avg)"),
                       ("unc_consistent", "sigma_MCD (trial-consistent)"),
                       ("alea_mean", "sigma_A (aleatoric)"),
                       ("alea_consistent", "sigma_A (trial-consistent, BPM)"),
                       ("tot_consistent", "sigma_E+A (trial-consistent, BPM)")]:
        rho, p = spearmanr(df[col], df.abs_err)
        print(f"  {label:30s}: Spearman rho={rho:+.3f}  p={p:.2g}")

    out_csv = args.out_dir / "mmpd_full_hetero_n20.tidy.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()
