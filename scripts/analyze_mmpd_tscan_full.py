"""Analyze TSCAN MC inference pickle on MMPD; produce trial-level tidy CSV.

Mirrors analyze_dlcn_tscan_full.py for DLCN, with MMPD-specific metadata
parsing (8 fields encoded in filename per MMPDLoader convention).

Filename convention (from MMPDLoader.preprocess_dataset):
    subject{N}_L{light}_MO{motion}_E{exercise}_S{skin}_GE{gender}_GL{glasses}_H{hair_cover}_MA{makeup}
e.g.,
    subject10_L1_MO1_E1_S4_GE2_GL2_H2_MA2

Pipeline (per trial, matches paper Section III-A):
    1. concatenate per-chunk MC mean predictions in sort order
    2. cumulative sum (integrate DiffNormalized to BVP)
    3. smoothness-prior detrend (Tarvainen 2002, lambda=100)
    4. Butterworth bandpass [0.6, 3.3] Hz, order 2
    5. zero-padded periodogram, argmax in band -> HR (BPM)

Output CSV schema matches the existing PhysNet pipeline output for direct
comparison, with columns:
    pid, light, state, exercise, skin, gender, glasser, hair_cover, makeup,
    n_chunks, hr_pred, hr_true, abs_err, unc_mean, unc_max, unc_p90

Note: 'state' here is motion type (1-4), reusing the column name from
analyze_dlcn_tscan_full.py for downstream-script compatibility (selective
prediction and conformal prediction scripts both expect 'state').

Usage:
    python scripts/analyze_mmpd_tscan_full.py \
        --pickle mc_results/mmpd_full_tscan_n20.pkl \
        --output results/day8_tscan/mmpd_full_tscan_n20.tidy.csv
"""

import argparse
import os
import pickle
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse
from scipy.signal import butter, filtfilt, periodogram
from tqdm import tqdm


# ----------------------------------------------------------------------------
# Signal-processing helpers (verbatim from analyze_dlcn_tscan_full.py)
# ----------------------------------------------------------------------------

def detrend_smoothness_prior(signal, lambda_val=100.0):
    """Smoothness-prior detrending (Tarvainen 2002), matches rPPG-Toolbox."""
    n = signal.shape[0]
    if n < 3:
        return signal.astype(np.float64).copy()
    H = np.identity(n)
    ones = np.ones(n)
    diags_data = np.array([ones, -2 * ones, ones])
    diags_index = np.array([0, 1, 2])
    D = scipy.sparse.spdiags(diags_data, diags_index, n - 2, n).toarray()
    filtered = np.dot(
        (H - np.linalg.inv(H + (lambda_val ** 2) * np.dot(D.T, D))),
        signal.astype(np.float64),
    )
    return filtered


def hr_from_bvp(signal, fs=30, low_cut=0.6, high_cut=3.3):
    """HR estimation: cumsum -> detrend -> bandpass -> periodogram -> argmax."""
    sig = np.asarray(signal, dtype=np.float64).flatten()
    if len(sig) < int(fs * 2):
        return np.nan
    sig = np.cumsum(sig)
    sig = detrend_smoothness_prior(sig, lambda_val=100.0)
    nyq = 0.5 * fs
    try:
        b, a = butter(2, [low_cut / nyq, high_cut / nyq], btype="band")
        sig = filtfilt(b, a, sig)
    except ValueError:
        return np.nan
    n_fft = max(2 ** 14, int(2 ** np.ceil(np.log2(len(sig) + 1))))
    freqs, psd = periodogram(sig, fs=fs, nfft=n_fft)
    mask = (freqs >= low_cut) & (freqs <= high_cut)
    if not np.any(mask):
        return np.nan
    f_peak = freqs[mask][np.argmax(psd[mask])]
    return 60.0 * float(f_peak)


# ----------------------------------------------------------------------------
# MMPD trial-key parsing
# ----------------------------------------------------------------------------

# Captures all 8 MMPD metadata fields from MMPDLoader filename convention.
_MMPD_RE = re.compile(
    r"(?P<pid>(?:subject|S)\d+)"
    r".*?_L(?P<light>\d+)"
    r".*?_MO(?P<motion>\d+)"
    r".*?_E(?P<exercise>\d+)"
    r".*?_S(?P<skin>\d+)"
    r".*?_GE(?P<gender>\d+)"
    r".*?_GL(?P<glasser>\d+)"
    r".*?_H(?P<hair_cover>\d+)"
    r".*?_MA(?P<makeup>\d+)"
)


def parse_mmpd_key(key):
    """Parse a MMPD pickle key into metadata dict, or None if it doesn't match."""
    s = str(key)
    m = _MMPD_RE.search(s)
    if m is None:
        return None
    return {
        "pid":        m.group("pid"),
        "light":      int(m.group("light")),
        "state":      int(m.group("motion")),  # motion -> 'state' for downstream compat
        "exercise":   int(m.group("exercise")),
        "skin":       int(m.group("skin")),
        "gender":     int(m.group("gender")),
        "glasser":    int(m.group("glasser")),
        "hair_cover": int(m.group("hair_cover")),
        "makeup":     int(m.group("makeup")),
    }


# ----------------------------------------------------------------------------
# Per-trial aggregation
# ----------------------------------------------------------------------------

def aggregate_trial(chunks_dict):
    """Concatenate chunks in sort order; return (mean, label, chunk_unc, n_chunks)."""
    sort_order = sorted(chunks_dict.keys())
    mean_parts, label_parts, chunk_unc = [], [], []
    for k in sort_order:
        ch = chunks_dict[k]
        mean_parts.append(np.asarray(ch["mean"]).flatten())
        label_parts.append(np.asarray(ch["label"]).flatten())
        chunk_unc.append(float(np.asarray(ch["std"]).flatten().mean()))
    return (
        np.concatenate(mean_parts),
        np.concatenate(label_parts),
        np.array(chunk_unc, dtype=np.float64),
        len(sort_order),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trial-level tidy CSV from TSCAN MC inference pickle on MMPD."
    )
    parser.add_argument("--pickle", required=True,
                        help="MC inference pickle from tools/run_tscan_mc_inference.py")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (parent dir created if missing)")
    parser.add_argument("--fs", type=float, default=30.0,
                        help="Sampling frequency (Hz). Default 30 (matches MMPD).")
    args = parser.parse_args()

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    print(f"=> Loading pickle: {args.pickle}")
    with open(args.pickle, "rb") as f:
        results = pickle.load(f)

    n_trials = len(results)
    n_chunks_total = sum(len(v) for v in results.values())
    print(f"=> Loaded {n_chunks_total} chunks across {n_trials} trial keys.")

    rows = []
    skipped_keys = []
    for key in tqdm(sorted(results.keys()), desc="Trials"):
        meta = parse_mmpd_key(key)
        if meta is None:
            skipped_keys.append(key)
            continue

        try:
            mean_sig, label_sig, chunk_unc, n_chunks = aggregate_trial(results[key])
        except Exception as e:
            warnings.warn(f"Aggregation failed for {key}: {e}")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            hr_pred = hr_from_bvp(mean_sig, fs=args.fs)
            hr_true = hr_from_bvp(label_sig, fs=args.fs)

        if np.isnan(hr_pred) or np.isnan(hr_true):
            warnings.warn(f"NaN HR for trial {key}; kept with NaN err.")
            abs_err = np.nan
        else:
            abs_err = abs(hr_pred - hr_true)

        row = dict(meta)
        row.update({
            "n_chunks":  n_chunks,
            "hr_pred":   hr_pred,
            "hr_true":   hr_true,
            "abs_err":   abs_err,
            "unc_mean":  float(chunk_unc.mean())             if len(chunk_unc) else np.nan,
            "unc_max":   float(chunk_unc.max())              if len(chunk_unc) else np.nan,
            "unc_p90":   float(np.percentile(chunk_unc, 90)) if len(chunk_unc) else np.nan,
        })
        rows.append(row)

    # Match column order to existing PhysNet output for diff-friendly comparison.
    df = pd.DataFrame(rows, columns=[
        "pid", "light", "state", "exercise", "skin", "gender",
        "glasser", "hair_cover", "makeup",
        "n_chunks", "hr_pred", "hr_true", "abs_err",
        "unc_mean", "unc_max", "unc_p90",
    ])
    df = df.sort_values(["pid", "light", "state", "exercise"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\n=> Wrote {len(df)} trial rows to {out}")

    # ---- Diagnostics for the paper ----
    valid = df.dropna(subset=["abs_err"])
    print(f"\n=== Diagnostics (TSCAN cross-dataset on MMPD) ===")
    print(f"  Trials with valid HR:  {len(valid)} / {len(df)}")
    if len(valid):
        print(f"  Aggregate MAE:         {valid['abs_err'].mean():.3f} BPM")
        print(f"  Aggregate median err:  {valid['abs_err'].median():.3f} BPM")
        print(f"  Aggregate RMSE:        {np.sqrt((valid['abs_err']**2).mean()):.3f} BPM")
        print(f"  Mean per-trial unc:    {valid['unc_mean'].mean():.4f}")
        print(f"  Pearson(HR_pred, HR_true): "
              f"{valid[['hr_pred','hr_true']].corr().iloc[0,1]:.3f}")

        # Per-skin breakdown -- the headline analysis for paper Section IV-E
        print(f"\n  *** Per-Fitzpatrick breakdown (paper Section IV-E claim) ***")
        skin_summary = (valid.groupby("skin")
                              .agg(n=("abs_err", "count"),
                                   MAE=("abs_err", "mean"),
                                   RMSE=("abs_err", lambda x: np.sqrt((x**2).mean())),
                                   sigma=("unc_mean", "mean"))
                              .reset_index())
        for _, r in skin_summary.iterrows():
            print(f"    Fitz {int(r['skin'])}: n={int(r['n']):3d}, "
                  f"MAE={r['MAE']:6.2f} BPM, RMSE={r['RMSE']:6.2f}, "
                  f"sigma={r['sigma']:.4f}")

        # Sigma-blindness diagnostic: ratio of sigma range vs MAE range across skin tones
        if len(skin_summary) >= 2:
            mae_ratio = skin_summary["MAE"].max() / skin_summary["MAE"].min()
            sigma_ratio = skin_summary["sigma"].max() / skin_summary["sigma"].min()
            sigma_pct = 100 * (skin_summary["sigma"].max() - skin_summary["sigma"].min()) / skin_summary["sigma"].mean()
            print(f"\n  Sigma-blindness check across skin tones:")
            print(f"    MAE   range ratio: {mae_ratio:.2f}x")
            print(f"    Sigma range ratio: {sigma_ratio:.3f}x  ({sigma_pct:.1f}% relative spread)")
            print(f"    => If sigma_ratio << MAE_ratio, sigma-blindness pattern reproduces.")

        # Per-(light, state) breakdown (16 cells) for cross-cell heterogeneity
        print(f"\n  Per-(lighting x motion) MAE summary:")
        cell_mae = (valid.groupby(["light", "state"])["abs_err"]
                       .agg(["count", "mean"])
                       .rename(columns={"count": "n", "mean": "MAE"})
                       .reset_index())
        for _, r in cell_mae.iterrows():
            print(f"    L{int(r['light'])}, M{int(r['state'])}: "
                  f"n={int(r['n']):3d}, MAE={r['MAE']:6.2f} BPM")

    if skipped_keys:
        print(f"\n  WARNING: {len(skipped_keys)} keys could not be parsed:")
        for k in skipped_keys[:5]:
            print(f"    {k!r}")
        if len(skipped_keys) > 5:
            print(f"    ... and {len(skipped_keys) - 5} more")


if __name__ == "__main__":
    main()