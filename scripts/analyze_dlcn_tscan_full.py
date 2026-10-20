"""Trial-level HR and uncertainty for DLCN from TSCAN MC-dropout pickles, with the pipeline of
analyze_dlcn_full_v2.py.

DLCN trials 1-4 are recorded at rest and 5-8 after exercise; within each block the lighting scenarios are
FI&FP, VI&FP, FI&VP and VI&VP (light codes 1-4).
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
# rPPG-Toolbox-equivalent signal-processing helpers
# ----------------------------------------------------------------------------

def detrend_smoothness_prior(signal, lambda_val=100.0):
    """Smoothness-prior detrending (Tarvainen 2002).
    Verbatim port of evaluation/post_process.py::_detrend in rPPG-Toolbox.
    """
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
    if len(sig) < int(fs * 2):  # need at least ~2 sec
        return np.nan
    # 1. integrate (DiffNormalized -> BVP)
    sig = np.cumsum(sig)
    # 2. detrend
    sig = detrend_smoothness_prior(sig, lambda_val=100.0)
    # 3. bandpass
    nyq = 0.5 * fs
    try:
        b, a = butter(2, [low_cut / nyq, high_cut / nyq], btype="band")
        sig = filtfilt(b, a, sig)
    except ValueError:
        return np.nan
    # 4. periodogram (zero-padded for spectral resolution)
    n_fft = max(2 ** 14, int(2 ** np.ceil(np.log2(len(sig) + 1))))
    freqs, psd = periodogram(sig, fs=fs, nfft=n_fft)
    mask = (freqs >= low_cut) & (freqs <= high_cut)
    if not np.any(mask):
        return np.nan
    f_peak = freqs[mask][np.argmax(psd[mask])]
    return 60.0 * float(f_peak)


# ----------------------------------------------------------------------------
# DLCN trial-key parsing
# ----------------------------------------------------------------------------

# Robust regex: matches keys like "P10_3", "subject10_3", "P10_T3_L3_S1", etc.
# Captures the LAST numeric trial number after the subject id.
_KEY_RE = re.compile(r"(?P<sid>(?:P|subject)?\d+).*?_(?:T)?(?P<trial>\d+)")


def parse_dlcn_key(key):
    """Parse a DLCN pickle key into (subject_id_str, trial_int, light, state).

    Returns None if the key doesn't match the expected DLCN convention.
    """
    s = str(key)
    m = _KEY_RE.match(s)
    if m is None:
        return None
    sid_raw = m.group("sid")
    # normalize to "PNN" form for consistency with paper notation
    sid_num = re.search(r"(\d+)", sid_raw).group(1)
    sid = f"P{int(sid_num)}"
    try:
        trial = int(m.group("trial"))
    except (TypeError, ValueError):
        return None
    if not (1 <= trial <= 8):
        return None
    light = ((trial - 1) % 4) + 1
    state = ((trial - 1) // 4) + 1
    return sid, trial, light, state


# ----------------------------------------------------------------------------
# Per-trial aggregation
# ----------------------------------------------------------------------------

def aggregate_trial(chunks_dict):
    """Concatenate chunks in sort order; return (mean_signal, label_signal,
    chunk_unc_means, n_chunks)."""
    sort_order = sorted(chunks_dict.keys())
    mean_parts, label_parts, chunk_unc = [], [], []
    for k in sort_order:
        ch = chunks_dict[k]
        m = np.asarray(ch["mean"]).flatten()
        l = np.asarray(ch["label"]).flatten()
        s = np.asarray(ch["std"]).flatten()
        mean_parts.append(m)
        label_parts.append(l)
        chunk_unc.append(float(s.mean()))
    return (
        np.concatenate(mean_parts),
        np.concatenate(label_parts),
        np.array(chunk_unc, dtype=np.float64),
        len(sort_order),
    )


def main():
    parser = argparse.ArgumentParser(
        description="Trial-level tidy CSV from TSCAN MC inference pickle on DLCN."
    )
    parser.add_argument("--pickle", required=True,
                        help="MC inference pickle from tools/run_tscan_mc_inference.py")
    parser.add_argument("--output", required=True,
                        help="Output CSV path (parent dir created if missing)")
    parser.add_argument("--fs", type=float, default=30.0,
                        help="Sampling frequency (Hz). Default 30 (matches DLCN).")
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
        parsed = parse_dlcn_key(key)
        if parsed is None:
            skipped_keys.append(key)
            continue
        sid, trial, light, state = parsed

        try:
            mean_sig, label_sig, chunk_unc, n_chunks = aggregate_trial(results[key])
        except Exception as e:
            warnings.warn(f"Aggregation failed for {key}: {e}")
            continue

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # filtfilt edge warnings on short trials
            hr_pred = hr_from_bvp(mean_sig, fs=args.fs)
            hr_true = hr_from_bvp(label_sig, fs=args.fs)

        if np.isnan(hr_pred) or np.isnan(hr_true):
            warnings.warn(f"NaN HR for trial {key} (n_chunks={n_chunks}); kept with NaN err.")
            abs_err = np.nan
        else:
            abs_err = abs(hr_pred - hr_true)

        rows.append({
            "pid":       sid,
            "trial":     trial,
            "light":     light,
            "state":     state,
            "n_chunks":  n_chunks,
            "hr_pred":   hr_pred,
            "hr_true":   hr_true,
            "abs_err":   abs_err,
            "unc_mean":  float(chunk_unc.mean())               if len(chunk_unc) else np.nan,
            "unc_max":   float(chunk_unc.max())                if len(chunk_unc) else np.nan,
            "unc_p90":   float(np.percentile(chunk_unc, 90))   if len(chunk_unc) else np.nan,
        })

    df = pd.DataFrame(rows).sort_values(["pid", "trial"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"\n=> Wrote {len(df)} trial rows to {out}")

    # Summary diagnostics for the paper
    valid = df.dropna(subset=["abs_err"])
    print(f"\n=== Diagnostics (TSCAN cross-dataset on DLCN) ===")
    print(f"  Trials with valid HR:  {len(valid)} / {len(df)}")
    if len(valid):
        print(f"  Aggregate MAE:         {valid['abs_err'].mean():.3f} BPM")
        print(f"  Aggregate median err:  {valid['abs_err'].median():.3f} BPM")
        print(f"  Aggregate RMSE:        {np.sqrt((valid['abs_err']**2).mean()):.3f} BPM")
        print(f"  Mean per-trial unc:    {valid['unc_mean'].mean():.4f}")
        print(f"  Pearson(HR_pred, HR_true): "
              f"{valid[['hr_pred','hr_true']].corr().iloc[0,1]:.3f}")
        print()
        print("  MAE by (lighting, state):")
        cell_mae = (valid.groupby(["light", "state"])["abs_err"]
                       .agg(["count", "mean"])
                       .rename(columns={"count": "n", "mean": "MAE"})
                       .reset_index())
        for _, r in cell_mae.iterrows():
            print(f"    L{int(r['light'])}, S{int(r['state'])}: "
                  f"n={int(r['n']):3d}, MAE={r['MAE']:6.2f} BPM")
        print()
        print("  Mean per-trial uncertainty by (lighting, state):")
        cell_unc = (valid.groupby(["light", "state"])["unc_mean"]
                       .mean().reset_index())
        for _, r in cell_unc.iterrows():
            print(f"    L{int(r['light'])}, S{int(r['state'])}: "
                  f"unc={r['unc_mean']:.4f}")

    if skipped_keys:
        print(f"\n  WARNING: {len(skipped_keys)} keys could not be parsed:")
        for k in skipped_keys[:5]:
            print(f"    {k!r}")
        if len(skipped_keys) > 5:
            print(f"    ... and {len(skipped_keys) - 5} more")


if __name__ == "__main__":
    main()