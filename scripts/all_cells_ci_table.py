"""Subject-cluster bootstrap 95% CIs for LACP coverage in every MMPD skin-tone x motion cell (Table VII).

Input: results/day_r1_matched/mmpd_per_trial_lacp.csv. Each trial's coverage is averaged over the splits
in which it was tested and subjects are resampled 20,000 times. A subject annotated under two skin tones
is counted in both rows.
"""
import pandas as pd, numpy as np

INPUT_CSV = r"results/day_r1_matched/mmpd_per_trial_lacp.csv"
COL_SUBJECT, COL_FITZ, COL_CELL, COL_COVERED, COL_SPLIT = "pid", "skin", "state", "covered", "seed"
COL_LIGHT, COL_EXERCISE = "light", "exercise"
MOTION = {1: "stationary", 2: "rotation", 3: "talking", 4: "walking"}
ORDER = ["all", "stationary", "rotation", "talking", "walking"]

d = pd.read_csv(INPUT_CSV)

# One row per real trial = unique (pid, skin, state, light, exercise).
# `covered` is averaged over whichever calibration splits (seeds) that
# trial happened to appear in -- this is the nuisance-average over the
# calibration draw, matching subject_level_stats.py's stated intent.
trial_keys = [COL_SUBJECT, COL_FITZ, COL_CELL, COL_LIGHT, COL_EXERCISE]
collapsed = d.groupby(trial_keys, as_index=False)[COL_COVERED].mean()

assert collapsed.shape[0] == d.groupby(trial_keys).ngroups
n_total_trials = len(collapsed)

rng = np.random.default_rng(0)


def boot_ci(sub, n=20000):
    st = sub.groupby(COL_SUBJECT)[COL_COVERED].agg(s="sum", n="size")
    s, nn = st["s"].values, st["n"].values
    k = len(st)
    if k < 2:
        return np.nan, np.nan, (s.sum() / nn.sum() if nn.sum() else np.nan), k
    idx = rng.integers(0, k, size=(n, k))
    boot = s[idx].sum(1) / nn[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return lo, hi, s.sum() / nn.sum(), k


rows = []
for skin, g in collapsed.groupby(COL_FITZ):
    lo, hi, pt, k = boot_ci(g)
    rows.append((skin, "all", k, len(g), pt, lo, hi))
for (skin, st), g in collapsed.groupby([COL_FITZ, COL_CELL]):
    lo, hi, pt, k = boot_ci(g)
    rows.append((skin, MOTION[st], k, len(g), pt, lo, hi))

rows.sort(key=lambda r: (r[0], ORDER.index(r[1])))

print("skin & motion & $N_{subj}$ & $N_{tri}$ & coverage & 95\\% CI \\\\")
for r in rows:
    motion = "--" if r[1] == "all" else r[1]
    ci = "[%.2f, %.2f]" % (r[5], r[6]) if np.isfinite(r[5]) else "--"
    print("%d & %s & %d & %d & %.3f & %s \\\\" % (r[0], motion, r[2], r[3], r[4], ci))

OUT_CSV = r"results/day_r1_matched/all_cells_ci.csv"
pd.DataFrame(rows, columns=["skin", "motion", "n_subjects", "n_trials", "coverage", "ci_low", "ci_high"]).to_csv(OUT_CSV, index=False)
print(f"\nSaved {OUT_CSV}")

print("\n--- sanity check: all vs sum(motion) ---")
for skin in sorted(collapsed[COL_FITZ].unique()):
    n_all = len(collapsed[collapsed[COL_FITZ] == skin])
    n_sum_by_state = sum(
        len(collapsed[(collapsed[COL_FITZ] == skin) & (collapsed[COL_CELL] == st)]) for st in MOTION
    )
    print(f"Fitz {skin}: all={n_all}, sum(motion)={n_sum_by_state}, match={n_all == n_sum_by_state}")

print(f"\n--- sanity check: total unique trials = {n_total_trials} (paper states 647 MMPD trials) ---")
