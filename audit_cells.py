import numpy as np, pandas as pd

def audit(name, csv, cells, n_cal, nmin=9):
    d = pd.read_csv(csv)
    pids = sorted(d.pid.unique())
    allc = d.groupby(cells).size().index
    per_split = []
    for s in range(10):
        cal = np.random.default_rng(s).permutation(pids)[:n_cal]
        per_split.append(d[d.pid.isin(cal)].groupby(cells).size().reindex(allc, fill_value=0))
    t = pd.concat(per_split, axis=1)
    out = pd.DataFrame({"n_min": t.min(axis=1), "n_max": t.max(axis=1),
                        "splits_below_9": (t < nmin).sum(axis=1),
                        "splits_empty": (t == 0).sum(axis=1)})
    print()
    print("==", name, "| cells:", cells, "| n_cal:", n_cal, "| subjects:", len(pids), "| trials:", len(d))
    print("splits with at least one cell below 9:", int((t < nmin).any(axis=0).sum()), "of 10")
    bad = out[out.splits_below_9 > 0]
    print(bad.to_string() if len(bad) else "no cell below 9 in any split")

m = "results/day_r1_matched/mmpd_main/mmpd_full_n20.tidy.csv"
dl = "results/day_r1_matched/dlcn_main.tidy.csv"
audit("MMPD skin x state", m, ["skin", "state"], 12)
audit("MMPD light x state", m, ["light", "state"], 12)
audit("MMPD state only", m, ["state"], 12)
audit("MMPD light only", m, ["light"], 12)
for n in (6, 9, 12, 15):
    audit("MMPD skin only", m, ["skin"], n)
audit("DLCN light x state", dl, ["light", "state"], 30)
audit("DLCN state only", dl, ["state"], 30)
audit("DLCN light only", dl, ["light"], 30)
