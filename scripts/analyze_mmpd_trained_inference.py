"""Convert MC inference pickles -> tidy CSV with per-chunk HR_pred, HR_true, |error|, sigma_mean.

For Priority 1 cross-training-source replication of paper Section IV-B:
training source = MMPD, test sources = UBFC + DLCN. Computes the same per-trial
HR + sigma metrics as the original UBFC-trained pipeline so MAE/sigma ratios
can be compared cell-by-cell with paper Table III.
"""
import pickle
import re
import os
import numpy as np
import torch
import pandas as pd
from scipy.signal import periodogram, butter, filtfilt

FS = 30
HR_LOW_HZ = 0.75   # 45 BPM
HR_HIGH_HZ = 2.5   # 150 BPM


def bandpass_filter(signal, fs=FS, low=HR_LOW_HZ, high=HR_HIGH_HZ):
    nyq = fs / 2.0
    b, a = butter(1, [low/nyq, high/nyq], btype="bandpass")
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


def extract_dlcn_metadata(subject_key):
    m = re.match(r"P(\d+)_T(\d+)_L(\d+)_S(\d+)", subject_key)
    if not m:
        return {}
    return {
        "subject_num": int(m.group(1)),
        "task": int(m.group(2)),
        "light": int(m.group(3)),
        "state": int(m.group(4)),
    }


def analyze_pickle(pickle_path, output_csv, dataset):
    print(f"\n=== {dataset}: {pickle_path}")
    with open(pickle_path, "rb") as f:
        data = pickle.load(f)
    print(f"  {len(data)} subjects/trials")
    
    rows = []
    for subject_key, chunks in data.items():
        for chunk_idx, chunk in chunks.items():
            mean_sig = to_numpy(chunk["mean"])
            std_sig = to_numpy(chunk["std"])
            label_sig = to_numpy(chunk["label"])
            
            hr_pred = fft_hr_bpm(mean_sig)
            hr_true = fft_hr_bpm(label_sig)
            sigma_mean = float(np.mean(std_sig))
            
            row = {
                "subject_key": subject_key,
                "chunk_idx": int(chunk_idx),
                "hr_pred": hr_pred,
                "hr_true": hr_true,
                "abs_err": (abs(hr_pred - hr_true) if not (np.isnan(hr_pred) or np.isnan(hr_true)) else np.nan),
                "sigma_mean": sigma_mean,
                "chunk_len": len(mean_sig),
            }
            
            if dataset == "DLCN":
                row.update(extract_dlcn_metadata(subject_key))
            elif dataset == "UBFC":
                m = re.match(r"subject(\d+)", subject_key)
                if m:
                    row["subject_num"] = int(m.group(1))
            
            rows.append(row)
    
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    print(f"  Wrote {len(df)} rows -> {output_csv}")
    
    valid = df.dropna(subset=["abs_err"])
    print(f"\n  --- Aggregate (n={len(valid)}/{len(df)}) ---")
    print(f"  MAE:         {valid['abs_err'].mean():.3f} BPM (std {valid['abs_err'].std():.3f})")
    print(f"  Mean sigma:  {valid['sigma_mean'].mean():.4f} (std {valid['sigma_mean'].std():.4f})")
    print(f"  HR true:     mean {valid['hr_true'].mean():.1f}, range [{valid['hr_true'].min():.1f}, {valid['hr_true'].max():.1f}]")
    
    if dataset == "DLCN":
        cells = valid.groupby(["light", "state"]).agg(
            count=("abs_err", "count"),
            mae=("abs_err", "mean"),
            sigma=("sigma_mean", "mean"),
        ).reset_index()
        cells["mae"] = cells["mae"].round(3)
        cells["sigma"] = cells["sigma"].round(4)
        print(f"\n  --- Per (light, state) cells (DLCN) ---")
        print(cells.to_string(index=False))
        
        if len(cells) > 1:
            mae_ratio = cells["mae"].max() / cells["mae"].min()
            sigma_ratio = cells["sigma"].max() / cells["sigma"].min()
            print(f"\n  Cell MAE ratio (max/min):   {mae_ratio:.2f}x")
            print(f"  Cell sigma ratio (max/min): {sigma_ratio:.2f}x")
            print(f"  Informativeness rho = MAE_ratio / sigma_ratio = {mae_ratio/sigma_ratio:.2f}")
    
    return df


os.makedirs("results/priority1", exist_ok=True)

ubfc_df = analyze_pickle(
    r"mc_results\mmpd_trained_ubfc_full_n20.pkl",
    r"results\priority1\mmpd_trained_ubfc_n20.tidy.csv",
    dataset="UBFC",
)

dlcn_df = analyze_pickle(
    r"mc_results\mmpd_trained_dlcn_full_n20.pkl",
    r"results\priority1\mmpd_trained_dlcn_n20.tidy.csv",
    dataset="DLCN",
)

print(f"\n=== Done ===")
print(f"Two tidy CSVs in results\\priority1\\")
