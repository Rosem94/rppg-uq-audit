"""Recompute the eight Fig. 4 phase-space points from CSVs and check Table II aggregation."""
import numpy as np, pandas as pd
from pathlib import Path

TIDY = {"DLCN": ("results/day_r1_matched/dlcn_main.tidy.csv", 30),
        "MMPD": ("results/day5_mmpd_v3/mmpd_full_n20.tidy.csv", 12)}
CONF = [("DLCN", "dlcn_state", ["state"]), ("DLCN", "dlcn_light", ["light"]),
        ("DLCN", "dlcn_light_state", ["light", "state"]),
        ("MMPD", "mmpd_state", ["state"]), ("MMPD", "mmpd_light", ["light"]),
        ("MMPD", "mmpd_skin", ["skin"]), ("MMPD", "mmpd_skin_state", ["skin", "state"]),
        ("MMPD", "mmpd_light_state", ["light", "state"])]
HARD = {"dlcn_state": (120.0, 1.86, 0.66, 0.45), "dlcn_light": (60.0, 3.12, 0.08, 0.17),
        "dlcn_light_state": (30.0, 6.76, 0.19, 0.15), "mmpd_state": (58.8, 2.16, 1.22, 0.43),
        "mmpd_light": (58.8, 1.35, -0.39, 0.10), "mmpd_skin": (58.8, 1.67, 1.32, 0.49),
        "mmpd_skin_state": (14.7, 4.20, -0.15, 0.30), "mmpd_light_state": (14.7, 2.76, -0.74, 0.02)}

rows = []
for ds, name, cols in CONF:
    df = pd.read_csv(TIDY[ds][0]); n_cal = TIDY[ds][1]
    subjects = sorted(df.pid.unique())
    het_full = df.groupby(cols).abs_err.mean().pipe(lambda s: s.max() / s.min())
    het_cal, ncell = [], []
    for seed in range(10):
        sh = np.random.default_rng(seed).permutation(subjects)
        cal = df[df.pid.isin(sh[:n_cal])]
        g = cal.groupby(cols).abs_err
        het_cal.append(g.mean().max() / g.mean().min()); ncell.append(g.size().mean())
    sel = pd.read_csv(f"results/day12_baseline/{name}/selective_seeds_{name}.csv")
    r05 = sel[np.isclose(sel.retention, 0.5)].groupby("criterion").mean(numeric_only=True)
    ben_m = r05.loc["naive", "mae_overall"] - r05.loc["mondrian_cp", "mae_overall"]
    ben_c = r05.loc["naive", "mae_overall"] - r05.loc["cellmae_sigma", "mae_overall"]
    abst = r05.loc["naive", "worst_cell_retention"] - r05.loc["mondrian_cp", "worst_cell_retention"]
    h = HARD[name]
    rows.append(dict(dataset=ds, config=name, n_per_cell=np.mean(ncell), n_hard=h[0],
                     het_full=het_full, het_cal=np.mean(het_cal), het_hard=h[1],
                     benefit_mondrian=ben_m, ben_hard=h[2], benefit_cellmae=ben_c,
                     abst=abst, abst_hard=h[3]))
out = pd.DataFrame(rows).round(3)
out.to_csv("results/day_r1_matched/phase_points.csv", index=False)
print(out.to_string(index=False))

print("\n=== Table II check: DLCN PhysNet per-cell MAE, trial-level (dlcn_main) ===")
d = pd.read_csv(TIDY["DLCN"][0])
c = d.groupby(["light", "state"]).abs_err.mean()
print(c.round(2).to_string()); print("spread %.2fx  overall MAE %.2f" % (c.max() / c.min(), d.abs_err.mean()))
try:
    ch = pd.read_csv("results/day3_dlcn_full/dlcn_full_98p_n20.tidy.csv")
    print("chunk-level columns:", list(ch.columns))
    ecol = [x for x in ch.columns if "err" in x.lower()][0]
    cc = ch.groupby(["light", "state"])[ecol].mean()
    print("=== chunk-level per-cell ==="); print(cc.round(2).to_string())
    print("spread %.2fx  overall %.2f" % (cc.max() / cc.min(), ch[ecol].mean()))
except Exception as e:
    print("chunk-level check failed:", e)

print("\n=== matched_dlcn.csv columns (B8) ===")
m = pd.read_csv("results/day_r1_matched/matched_dlcn.csv"); print(list(m.columns))
for k in [x for x in m.columns if "err" in x.lower()]:
    print(f"  {k}: mean {m[k].mean():.2f}")
