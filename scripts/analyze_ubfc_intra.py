"""Trial-level HR and uncertainty for the UBFC-rPPG intra-dataset test split, with the pipeline of
analyze_dlcn_full_v2.py.
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import torch
from scipy.signal import butter
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve
from tqdm import tqdm

FS = 30.0
LOW_PASS = 0.6
HIGH_PASS = 3.3


def _next_power_of_2(x):
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def _detrend(signal, lambda_value=100):
    n = signal.shape[0]
    ones = np.ones(n)
    minus_twos = -2 * np.ones(n)
    diags_data = np.array([ones, minus_twos, ones])
    diags_index = np.array([0, 1, 2])
    D = spdiags(diags_data, diags_index, n - 2, n).tocsc()
    A = speye(n, format="csc") + (lambda_value ** 2) * (D.T @ D)
    return signal - spsolve(A, signal)


def calc_hr_toolbox(signal, fs=FS, diff_flag=True):
    """HR via toolbox-style: cumsum + detrend + bandpass + periodogram."""
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
    mask_freq = np.take(f_ppg, fmask)
    mask_pxx = np.take(pxx_ppg, fmask)
    return np.take(mask_freq, np.argmax(mask_pxx, 0))[0] * 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out_dir", required=True, type=Path)
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.pkl, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} subjects from {args.pkl}")

    records = []
    for sid in tqdm(sorted(data.keys()), desc="subjects"):
        chunks = data[sid]
        sorted_items = sorted(chunks.items(), key=lambda kv: kv[0])
        means  = torch.cat([c['mean']  for _, c in sorted_items]).numpy()
        stds   = torch.cat([c['std']   for _, c in sorted_items]).numpy()
        labels = torch.cat([c['label'] for _, c in sorted_items]).numpy()

        # HR over concatenated chunks (single periodogram per subject)
        hr_pred = calc_hr_toolbox(means,  fs=FS, diff_flag=True)
        hr_true = calc_hr_toolbox(labels, fs=FS, diff_flag=True)

        # Uncertainty: mean-per-chunk-std, then aggregate across chunks
        sigma_chunks = np.array([c['std'].mean().item() for _, c in sorted_items])

        records.append({
            "pid":      sid,
            "n_chunks": len(chunks),
            "hr_pred":  hr_pred,
            "hr_true":  hr_true,
            "abs_err":  abs(hr_pred - hr_true),
            "unc_mean": sigma_chunks.mean(),
            "unc_max":  sigma_chunks.max(),
            "unc_p90":  float(np.percentile(sigma_chunks, 90)),
        })

    df = pd.DataFrame(records)
    print("\n=== Per-subject results ===")
    print(df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\n=== UBFC intra-dataset (n={len(df)} test subjects) ===")
    print(f"MAE:        {df.abs_err.mean():.3f} BPM")
    print(f"RMSE:       {(df.abs_err ** 2).mean() ** 0.5:.3f} BPM")
    print(f"Median err: {df.abs_err.median():.3f} BPM")
    print(f"unc range:  [{df.unc_mean.min():.4f}, {df.unc_mean.max():.4f}]")

    out_csv = args.out_dir / "ubfc_intra_n20.tidy.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()