import pandas as pd, numpy as np, glob, os

pats = ["results/day12_baseline/*/selective_seeds_*.csv",
        "results/day6/*/selective_seeds_*.csv",
        "results/day7/*/selective_seeds*.csv",
        "results/day9_cross_backbone/*/selective_seeds_*.csv"]

files = sorted({os.path.normpath(p) for pat in pats for p in glob.glob(pat)})
rows = []
for f in files:
    d = pd.read_csv(f)
    if not {"retention","mae_overall","criterion"}.issubset(d.columns):
        continue
    s = d[np.isclose(d.retention, 0.5)].groupby("criterion")["mae_overall"].mean()
    if "naive" not in s.index:
        continue
    rec = {"file": os.path.relpath(f)}
    for c in ["mondrian_cp","cellmae_sigma","cellmae_lookup"]:
        rec[c] = round(s["naive"] - s[c], 3) if c in s.index else None
    rec["naive_mae"] = round(s["naive"], 3)
    rows.append(rec)

print(pd.DataFrame(rows).to_string(index=False))
