import pickle, numpy as np, pandas as pd, sys
from pathlib import Path
from scipy.stats import spearmanr

sys.path.insert(0, "scripts")
from analyze_dlcn_full_v2 import calc_hr_toolbox

rng = np.random.default_rng(42)
N_SUB = 10  # random 5-pass subsets per trial

def subsample_sigma(pkl_path):
    with open(pkl_path, "rb") as f:
        d = pickle.load(f)
    rows = []
    for key, chunks in d.items():
        sorted_chunks = sorted(chunks.items()) if isinstance(chunks, dict) else list(enumerate(chunks))
        stack = np.stack([np.asarray(p["samples"]) for _, p in sorted_chunks], axis=1)  # (20, n_chunks, T)
        hr20 = np.array([calc_hr_toolbox(stack[k].reshape(-1), diff_flag=True) for k in range(stack.shape[0])])
        sig5 = np.mean([hr20[rng.choice(20, 5, replace=False)].std() for _ in range(N_SUB)])
        rows.append({"key": str(key), "sig20": hr20.std(), "sig5": sig5})
    return pd.DataFrame(rows)

for ds, pkl, csv in [("dlcn", "dlcn_seed0_n20.pkl", "dlcn_seed0.tidy.csv"),
                     ("mmpd", "mmpd_seed0_n20.pkl", "mmpd_seed0/mmpd_full_n20.tidy.csv")]:
    base = Path("results/day_r1_matched")
    sub = subsample_sigma(base / pkl)
    tidy = pd.read_csv(base / csv)
    assert len(sub) == len(tidy), f"{ds}: row mismatch {len(sub)} vs {len(tidy)}"
    # rows are per-trial in insertion order on both sides; verify via sig20 ~ unc_consistent
    check = np.corrcoef(sub.sig20.values, tidy.unc_consistent.values)[0, 1]
    print(f"\n{ds.upper()}  (alignment check corr sig20 vs unc_consistent: {check:.4f})")
    if check < 0.999:
        print("  WARNING: row order mismatch, need key-based merge; stopping interpretation here")
        continue
    err = tidy.abs_err.values
    r20, _ = spearmanr(sub.sig20, err)
    r5, _ = spearmanr(sub.sig5, err)
    print(f"  rho(sigma_cons 20 passes, |err|) = {r20:+.3f}")
    print(f"  rho(sigma_cons  5 passes, |err|) = {r5:+.3f}   <- compute-matched to 5-seed ensemble")
