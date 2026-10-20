"""In-distribution control on DLCN (Sec. IV-B4): tidy per-trial CSV and cell-stratified summary for the
DLCN-trained PhysNet MC-dropout model on its subject-disjoint test split.
"""
import argparse
import os
import pickle
import re

import numpy as np
import pandas as pd
import torch
from scipy.signal import periodogram, butter, filtfilt

FS = 30
HR_LOW_HZ = 0.75   # 45 BPM
HR_HIGH_HZ = 2.5   # 150 BPM

# DLCN preprocessed filenames: P{participant}_T{trial}_L{lighting}_S{state}
KEY_RE = re.compile(r"P(\d+)_T(\d+)_L(\d+)_S(\d+)")


def bandpass_filter(signal, fs=FS, low=HR_LOW_HZ, high=HR_HIGH_HZ):
    nyq = fs / 2.0
    b, a = butter(1, [low / nyq, high / nyq], btype="bandpass")
    return filtfilt(b, a, signal)


def fft_hr_bpm(ppg_signal, fs=FS):
    if len(ppg_signal) < 32 or np.all(ppg_signal == 0) or np.std(ppg_signal) < 1e-8:
        return np.nan
    try:
        filt = bandpass_filter(ppg_signal, fs)
    except Exception:
        return np.nan
    freqs, power = periodogram(filt, fs=fs, detrend=False)
    valid = (freqs >= HR_LOW_HZ) & (freqs <= HR_HIGH_HZ)
    if not np.any(valid) or np.all(power[valid] == 0):
        return np.nan
    peak_freq = freqs[valid][np.argmax(power[valid])]
    return float(peak_freq * 60)


def to_numpy(arr):
    if isinstance(arr, torch.Tensor):
        return arr.detach().cpu().numpy()
    return np.asarray(arr)


def parse_key(subject_key):
    """Parse P{p}_T{t}_L{l}_S{s}. Falls back to deriving light/state from
    the trial index when the L/S suffix is absent (older cache naming)."""
    m = KEY_RE.match(str(subject_key))
    if m:
        p, t, light, state = (int(x) for x in m.groups())
        return {"subject_num": p, "trial": t, "light": light, "state": state}
    m2 = re.match(r"P(\d+)_T?(\d+)", str(subject_key))
    if not m2:
        return {}
    p, t = int(m2.group(1)), int(m2.group(2))
    return {
        "subject_num": p,
        "trial": t,
        "light": ((t - 1) % 4) + 1,
        "state": 1 if t <= 4 else 2,
    }


def cell_report(valid, axis_cols, fh, min_n=10):
    """Per-cell MAE / sigma table + max/min ratios along one metadata axis."""
    name = "x".join(axis_cols)
    cells = valid.groupby(axis_cols, as_index=False).agg(
        n_chunks=("abs_err", "count"),
        n_subj=("subject_num", "nunique"),
        mae=("abs_err", "mean"),
        sigma=("sigma_mean", "mean"),
    )
    cells["mae"] = cells["mae"].round(3)
    cells["sigma"] = cells["sigma"].round(4)
    lines = [f"\n--- Cells by {name} ---", cells.to_string(index=False)]
    big = cells[cells["n_chunks"] >= min_n]
    if len(big) < 2:
        lines.append(f"[{name}] fewer than 2 cells with n>={min_n}; ratios not computed")
    elif big["mae"].min() <= 0 or big["sigma"].min() <= 0:
        lines.append(f"[{name}] a cell has zero MAE/sigma; ratios not computed")
    else:
        mae_ratio = big["mae"].max() / big["mae"].min()
        sigma_ratio = big["sigma"].max() / big["sigma"].min()
        lines.append(
            f"[{name}] cells with n>={min_n}: {len(big)}/{len(cells)}; "
            f"MAE ratio {mae_ratio:.2f}x, sigma ratio {sigma_ratio:.2f}x, "
            f"eta = {mae_ratio / sigma_ratio:.2f}"
        )
    out = "\n".join(lines)
    print(out)
    fh.write(out + "\n")
    return cells


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pickle", required=True)
    ap.add_argument("--out_dir", default="results/r4_8_indist_dlcn")
    ap.add_argument("--min_n", type=int, default=10,
                    help="Min chunks per cell to enter the ratio computation.")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    with open(args.pickle, "rb") as f:
        data = pickle.load(f)
    print(f"Loaded {len(data)} trial keys from {args.pickle}")

    rows = []
    n_unparsed = 0
    for subject_key, chunks in data.items():
        meta = parse_key(subject_key)
        if not meta:
            n_unparsed += 1
            if n_unparsed <= 5:
                print(f"  WARNING: unparsed key {subject_key}")
        for chunk_idx, chunk in chunks.items():
            mean_sig = to_numpy(chunk["mean"])
            std_sig = to_numpy(chunk["std"])
            label_sig = to_numpy(chunk["label"])
            hr_pred = fft_hr_bpm(mean_sig)
            hr_true = fft_hr_bpm(label_sig)
            row = {
                "subject_key": subject_key,
                "chunk_idx": int(chunk_idx),
                "hr_pred": hr_pred,
                "hr_true": hr_true,
                "abs_err": (abs(hr_pred - hr_true)
                            if not (np.isnan(hr_pred) or np.isnan(hr_true)) else np.nan),
                "sigma_mean": float(np.mean(std_sig)),
                "chunk_len": len(mean_sig),
            }
            row.update(meta)
            rows.append(row)
    if n_unparsed:
        print(f"  WARNING: {n_unparsed} keys could not be parsed")

    df = pd.DataFrame(rows)
    csv_path = os.path.join(args.out_dir, "dlcn_trained_dlcn_test_n20.tidy.csv")
    df.to_csv(csv_path, index=False)
    print(f"Wrote {len(df)} rows -> {csv_path}")

    summary_path = os.path.join(args.out_dir, "summary_indist_dlcn.txt")
    valid = df.dropna(subset=["abs_err", "subject_num"])
    with open(summary_path, "w") as fh:
        subj_list = sorted(valid["subject_num"].unique().tolist())
        head = (
            f"In-distribution DLCN-trained -> DLCN test (participant-disjoint)\n"
            f"pickle: {args.pickle}\n"
            f"chunks: {len(valid)}/{len(df)} valid; "
            f"participants: {len(subj_list)} {subj_list}\n"
            f"Aggregate MAE {valid['abs_err'].mean():.3f} BPM "
            f"(std {valid['abs_err'].std():.3f}); "
            f"mean sigma {valid['sigma_mean'].mean():.4f}\n"
            f"HR true mean {valid['hr_true'].mean():.1f}, "
            f"range [{valid['hr_true'].min():.1f}, {valid['hr_true'].max():.1f}]\n"
        )
        print(head)
        fh.write(head)

        per_subj = valid.groupby("subject_num", as_index=False).agg(
            n_chunks=("abs_err", "count"),
            mae=("abs_err", "mean"), sigma=("sigma_mean", "mean"),
        ).round(4)
        block = "\n--- Per held-out participant ---\n" + per_subj.to_string(index=False)
        print(block)
        fh.write(block + "\n")

        # Primary axes for DLCN: lighting scenario and physiological state,
        # plus the full 4x2 cell grid used in paper Table II.
        cell_report(valid, ["light"], fh, args.min_n)
        cell_report(valid, ["state"], fh, args.min_n)
        cell_report(valid, ["light", "state"], fh, args.min_n)
        # Spearman rank correlation sigma vs |err| at chunk level.
        rho = valid[["sigma_mean", "abs_err"]].corr(method="spearman").iloc[0, 1]
        tail = f"\nChunk-level Spearman rho(sigma, |err|) = {rho:.3f}\n"
        print(tail)
        fh.write(tail)

    print(f"Summary -> {summary_path}")


if __name__ == "__main__":
    main()
