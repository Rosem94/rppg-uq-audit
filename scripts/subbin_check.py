"""Sub-bin peak interpolation check for the trial-consistent uncertainty (Reviewer 1, comment 3).

Reads an MC-dropout inference pickle (with per-pass "samples"), recomputes the trial HR of every stochastic
pass with the toolbox periodogram (bin = 0.879 BPM) and with Gaussian (log-parabolic) interpolation of the
periodogram peak, and compares the two versions of sigma_cons. Nothing in the repository is modified.

Usage:  python subbin_check.py --pkl <inference.pkl> --out_csv subbin_dlcn.csv
"""
import argparse
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.signal
from scipy.signal import butter
from scipy.sparse import spdiags, eye as speye
from scipy.sparse.linalg import spsolve
from scipy.stats import spearmanr

FS, LOW, HIGH = 30.0, 0.6, 3.3


def _detrend(x, lam=100):
    n = x.shape[0]
    o = np.ones(n)
    D = spdiags(np.array([o, -2 * o, o]), np.array([0, 1, 2]), n - 2, n).tocsc()
    A = speye(n, format="csc") + (lam ** 2) * (D.T @ D)
    return x - spsolve(A, x)


def hr_both(sig):
    """Return (toolbox HR, interpolated HR, bin width in BPM) for a DiffNormalized trial signal."""
    x = _detrend(np.cumsum(np.asarray(sig, dtype=float)), 100)
    b, a = butter(1, [LOW / FS * 2, HIGH / FS * 2], btype="bandpass")
    x = scipy.signal.filtfilt(b, a, x)
    n = x.shape[0]
    nfft = 1 if n == 0 else 2 ** (n - 1).bit_length()
    f, p = scipy.signal.periodogram(x, fs=FS, nfft=nfft, detrend=False)
    idx = np.where((f >= LOW) & (f <= HIGH))[0]
    i = idx[np.argmax(p[idx])]
    df = f[1] - f[0]
    hr_bin = f[i] * 60.0
    delta = 0.0
    if 0 < i < len(p) - 1 and min(p[i - 1], p[i], p[i + 1]) > 0:
        la, lb, lc = np.log(p[i - 1]), np.log(p[i]), np.log(p[i + 1])
        den = la - 2 * lb + lc
        if den < 0:
            delta = float(np.clip(0.5 * (la - lc) / den, -0.5, 0.5))
    return hr_bin, (f[i] + delta * df) * 60.0, df * 60.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pkl", required=True, type=Path)
    ap.add_argument("--out_csv", required=True, type=Path)
    args = ap.parse_args()
    with open(args.pkl, "rb") as fh:
        data = pickle.load(fh)

    rows = []
    for k, (sid, chunks) in enumerate(data.items()):
        ch = sorted(chunks.items(), key=lambda t: int(t[0]))
        if "samples" not in ch[0][1]:
            raise SystemExit("This pickle has no per-pass 'samples'; use the inference pickle "
                             "behind results/day_r1_matched.")
        cat = lambda key: np.concatenate([np.asarray(c[key]).reshape(-1) for _, c in ch])
        p_bin, p_int, binw = hr_both(cat("mean"))
        t_bin, t_int, _ = hr_both(cat("label"))
        stack = np.stack([np.asarray(c["samples"]) for _, c in ch], axis=1)
        per = np.array([hr_both(stack[j].reshape(-1))[:2] for j in range(stack.shape[0])])
        rows.append(dict(trial_id=str(sid), n_pass=stack.shape[0], bin_bpm=binw,
                         abs_err=abs(p_bin - t_bin), abs_err_interp=abs(p_int - t_int),
                         unc_mean=float(np.mean([np.asarray(c["std"]).mean() for _, c in ch])),
                         sig_cons=float(per[:, 0].std()), sig_cons_interp=float(per[:, 1].std())))
        if (k + 1) % 100 == 0:
            print(f"  {k + 1}/{len(data)} trials", flush=True)

    d = pd.DataFrame(rows)
    d.to_csv(args.out_csv, index=False)
    binw = d.bin_bpm.median()
    floored = np.maximum(d.sig_cons, binw)
    rs = lambda a, b: spearmanr(a, b)[0]
    print()
    print(f"file: {args.pkl.name}   trials: {len(d)}   passes: {int(d.n_pass.median())}")
    print(f"spectral bin (median over trials): {binw:.3f} BPM   "
          f"trials with a different bin: {(~np.isclose(d.bin_bpm, binw)).sum()}")
    print(f"MAE, toolbox estimator:           {d.abs_err.mean():.2f} BPM")
    print(f"MAE, interpolated point estimate: {d.abs_err_interp.mean():.2f} BPM")
    print(f"sigma_cons == 0, toolbox:         {(d.sig_cons == 0).mean() * 100:.1f} % of trials")
    print(f"sigma_cons == 0, interpolated:    {(d.sig_cons_interp == 0).mean() * 100:.1f} % of trials")
    print(f"sigma_cons_interp < one bin:      {(d.sig_cons_interp < binw).mean() * 100:.1f} % of trials")
    print(f"smallest sigma_cons_interp:       {d.sig_cons_interp.min():.4f} BPM")
    print(f"mean sigma_cons  toolbox / floored / interpolated: "
          f"{d.sig_cons.mean():.2f} / {floored.mean():.2f} / {d.sig_cons_interp.mean():.2f} BPM")
    print("Spearman r_s with |error| of the toolbox point estimate:")
    print(f"   default proxy (unc_mean):      {rs(d.unc_mean, d.abs_err):+.3f}")
    print(f"   sigma_cons, toolbox:           {rs(d.sig_cons, d.abs_err):+.3f}")
    print(f"   sigma_cons, floored:           {rs(floored, d.abs_err):+.3f}")
    print(f"   sigma_cons, interpolated:      {rs(d.sig_cons_interp, d.abs_err):+.3f}")
    print("Spearman r_s when the point estimate is interpolated too:")
    print(f"   sigma_cons, interpolated:      {rs(d.sig_cons_interp, d.abs_err_interp):+.3f}")
    print(f"Spearman between toolbox and interpolated sigma_cons: {rs(d.sig_cons, d.sig_cons_interp):+.3f}")
    print(f"saved {args.out_csv}")


if __name__ == "__main__":
    main()
