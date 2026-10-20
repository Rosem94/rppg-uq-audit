"""Trial-level HR and uncertainty for DLCN from MC-dropout inference pickles.

Per trial the chunks are concatenated, the DiffNormalized signal is integrated, detrended (smoothness prior,
lambda = 100), band-pass filtered to 0.6-3.3 Hz, and the HR is the peak of a zero-padded periodogram, as in
rPPG-Toolbox. Writes a tidy per-trial CSV with the HR error, the chunk-averaged dropout sigma and the
trial-consistent sigma (every stochastic pass through the same pipeline).
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
from scipy.stats import bootstrap, mannwhitneyu, pearsonr, spearmanr
from tqdm import tqdm

FS = 30.0
LOW_PASS = 0.6
HIGH_PASS = 3.3
LIGHT_NAMES = {1: "FI&FP", 2: "VI&FP", 3: "FI&VP", 4: "VI&VP"}


# --- Adapted from rPPG-Toolbox evaluation/post_process.py ---


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(signal, lambda_value=100):
    """Detrend via sparse solve. Mathematically equivalent to toolbox dense inv,
    but ~100x faster: matrix (H + lambda^2 D'D) is band-sparse, solved in O(n)."""
    n = signal.shape[0]
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, n - 2, n).tocsc()
    A = speye(n, format="csc") + (lambda_value ** 2) * (D.T @ D)
    # toolbox computes (H - inv(A)) @ signal = signal - inv(A) @ signal
    return signal - spsolve(A, signal)


def calc_hr_toolbox(signal, fs=FS, diff_flag=True):
    """HR computation matching toolbox calculate_metric_per_video + _calculate_fft_hr."""
    if diff_flag:
        signal = _detrend(np.cumsum(signal), 100)
    else:
        signal = _detrend(signal, 100)
    # Bandpass
    [b, a] = butter(1, [LOW_PASS / fs * 2, HIGH_PASS / fs * 2], btype="bandpass")
    signal = scipy.signal.filtfilt(b, a, np.double(signal))
    # FFT (periodogram with zero-padding)
    sig_2d = np.expand_dims(signal, 0)
    N = _next_power_of_2(sig_2d.shape[1])
    f_ppg, pxx_ppg = scipy.signal.periodogram(sig_2d, fs=fs, nfft=N, detrend=False)
    fmask = np.argwhere((f_ppg >= LOW_PASS) & (f_ppg <= HIGH_PASS))
    mask_freq = np.take(f_ppg, fmask)
    mask_pxx = np.take(pxx_ppg, fmask)
    return np.take(mask_freq, np.argmax(mask_pxx, 0))[0] * 60.0


def parse_subj(subj_id):
    m = re.match(r"P(\d+)_T(\d+)", str(subj_id))
    if not m:
        return None
    p, t = int(m.group(1)), int(m.group(2))
    light = ((t - 1) % 4) + 1
    state = 1 if t <= 4 else 2
    return p, t, light, state


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()

    with open(args.pkl, "rb") as f:
        data = pickle.load(f)

    # Aggregate per-trial: concat chunks in chunk_id order
    records = []
    for subj_id, chunks in tqdm(data.items(), desc="Per-trial HR", ncols=80):
        parsed = parse_subj(subj_id)
        if parsed is None:
            continue
        pid, trial, light, state = parsed
        # Sort chunks by integer chunk_id
        sorted_chunks = sorted(chunks.items(), key=lambda x: int(x[0]))
        full_pred = np.concatenate([
            np.asarray(payload["mean"]).flatten() for _, payload in sorted_chunks
        ])
        full_label = np.concatenate([
            np.asarray(payload["label"]).flatten() for _, payload in sorted_chunks
        ])
        # Mean per-chunk uncertainty (averaged over chunks of trial)
        chunk_uncs = np.array([
            np.asarray(payload["std"]).flatten().mean() for _, payload in sorted_chunks
        ])
        n_chunks = len(sorted_chunks)
        hr_pred = calc_hr_toolbox(full_pred, diff_flag=True)
        hr_true = calc_hr_toolbox(full_label, diff_flag=True)
        # Consistent sigma: run the SAME trial-level HR
        # pipeline on each MC-dropout pass, then take the std of the per-pass
        # trial HRs. Uncertainty in BPM, on the same quantity as the point
        # estimate and abs_err, unlike unc_mean (a per-timestep waveform std).
        sample_stack = np.stack([
            np.asarray(payload["samples"]) for _, payload in sorted_chunks
        ], axis=1)                               # (N_passes, n_chunks, T)
        hr_per_pass = np.array([
            calc_hr_toolbox(sample_stack[k].reshape(-1), diff_flag=True)
            for k in range(sample_stack.shape[0])
        ])
        records.append({
            "pid": pid, "trial": trial, "light": light, "state": state,
            "n_chunks": n_chunks,
            "hr_pred": hr_pred, "hr_true": hr_true,
            "abs_err": abs(hr_pred - hr_true),
            "unc_mean": chunk_uncs.mean(),
            "unc_max": chunk_uncs.max(),
            "unc_p90": np.quantile(chunk_uncs, 0.9),
            "unc_consistent": float(hr_per_pass.std()),
            "hr_pred_passmean": float(hr_per_pass.mean()),
        })

    df = pd.DataFrame(records)
    print(f"\nPer-trial records: {len(df)} (vs 8568 chunks)")
    print(f"Unique participants: {df.pid.nunique()}, trials: {sorted(df.trial.unique())}\n")

    print("=== Overall (per-trial, toolbox-style HR) ===")
    rng = np.random.default_rng(42)
    mae_ci = bootstrap(
        (df.abs_err.values,), np.mean,
        n_resamples=2000, random_state=rng, confidence_level=0.95
    ).confidence_interval
    print(f"  HR MAE:    {df.abs_err.mean():.2f}  [95% CI {mae_ci.low:.2f}, {mae_ci.high:.2f}]  BPM")
    print(f"  Median:    {df.abs_err.median():.2f}  BPM")
    print(f"  Pearson:   {df[['hr_pred', 'hr_true']].corr().iloc[0, 1]:.3f}")
    print(f"  Mean unc:  {df.unc_mean.mean():.4f} (averaged across chunks of each trial)\n")

    print("=== Per (lighting × state) ===")
    for state in [1, 2]:
        s_name = ["rest", "exercise"][state - 1]
        for light in [1, 2, 3, 4]:
            sub = df[(df.light == light) & (df.state == state)]
            if len(sub) == 0:
                continue
            print(f"  L{light} {LIGHT_NAMES[light]:5s} | {s_name:8s}: "
                  f"n={len(sub):3d}  MAE={sub.abs_err.mean():6.2f}  "
                  f"unc={sub.unc_mean.mean():.4f}")
    print()

    print("=== Trial-level correlation (n=780) unc_mean vs |error| ===")
    r_p, p_p = pearsonr(df.unc_mean, df.abs_err)
    r_s, p_s = spearmanr(df.unc_mean, df.abs_err)
    print(f"  Pearson:  r={r_p:+.3f}  p={p_p:.3g}")
    print(f"  Spearman: r={r_s:+.3f}  p={p_s:.3g}\n")

    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved {args.out_csv}")


if __name__ == "__main__":
    main()