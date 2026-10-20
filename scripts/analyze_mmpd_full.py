"""Trial-level HR and uncertainty for MMPD from MC-dropout inference pickles, with the pipeline of
analyze_dlcn_full_v2.py. MMPD file names are parsed into light, motion (state), exercise, skin tone and the
remaining metadata fields.
"""
import argparse
import pickle
import re
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
        means  = torch.cat([c['mean']  for _, c in sorted_items]).numpy()
        labels = torch.cat([c['label'] for _, c in sorted_items]).numpy()

        hr_pred = calc_hr_toolbox(means,  fs=FS, diff_flag=True)
        hr_true = calc_hr_toolbox(labels, fs=FS, diff_flag=True)
        sigma_chunks = np.array([c['std'].mean().item() for _, c in sorted_items])

        # Consistent sigma: same trial-level HR pipeline
        # applied to each MC-dropout pass, std over per-pass trial HRs (BPM).
        sample_stack = np.stack([
            np.asarray(c['samples']) for _, c in sorted_items
        ], axis=1)                               # (N_passes, n_chunks, T)
        hr_per_pass = np.array([
            calc_hr_toolbox(sample_stack[k].reshape(-1), fs=FS, diff_flag=True)
            for k in range(sample_stack.shape[0])
        ])

        records.append({
            **meta,
            "n_chunks": len(chunks),
            "hr_pred": hr_pred,
            "hr_true": hr_true,
            "abs_err": abs(hr_pred - hr_true),
            "unc_mean": sigma_chunks.mean(),
            "unc_max":  sigma_chunks.max(),
            "unc_p90":  float(np.percentile(sigma_chunks, 90)),
            "unc_consistent": float(hr_per_pass.std()),
            "hr_pred_passmean": float(hr_per_pass.mean()),
        })

    if skipped > 0:
        print(f"Warning: {skipped} keys did not match parser")

    df = pd.DataFrame(records)
    print(f"\n=== MMPD cross-dataset (UBFC-trained PhysNet+MCDropout) ===")
    print(f"Total trials:    {len(df)}")
    print(f"Unique subjects: {df.pid.nunique()}")
    print(f"MAE:             {df.abs_err.mean():.3f} BPM")
    print(f"RMSE:            {(df.abs_err ** 2).mean() ** 0.5:.3f} BPM")
    print(f"Median err:      {df.abs_err.median():.3f} BPM")
    print(f"unc range:       [{df.unc_mean.min():.4f}, {df.unc_mean.max():.4f}]")

    print(f"\n=== Per-(light, motion) cell ===")
    cell = (df.groupby(["light", "state"])
              .agg(n=("abs_err", "size"),
                   mae=("abs_err", "mean"),
                   median_err=("abs_err", "median"),
                   unc=("unc_mean", "mean"))
              .round(2).reset_index())
    cell["light_name"]  = cell.light.map(LIGHT_NAMES)
    cell["motion_name"] = cell.state.map(MOTION_NAMES)
    print(cell[["light", "light_name", "state", "motion_name",
                "n", "mae", "median_err", "unc"]].to_string(index=False))

    print(f"\n=== Per-exercise ===")
    print(df.groupby("exercise").agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean")
    ).round(2).to_string())

    print(f"\n=== Per-skin (Fitzpatrick) ===")
    print(df.groupby("skin").agg(
        n=("abs_err", "size"),
        mae=("abs_err", "mean"),
        unc=("unc_mean", "mean")
    ).round(2).to_string())

    out_csv = args.out_dir / "mmpd_full_n20.tidy.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved {out_csv}")


if __name__ == "__main__":
    main()