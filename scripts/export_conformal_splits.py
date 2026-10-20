import json, pathlib, numpy as np, pandas as pd
out = pathlib.Path("splits_conformal"); out.mkdir(exist_ok=True)
for name, csv, n_cal in [("dlcn","results/day_r1_matched/dlcn_main.tidy.csv",30),("mmpd","results/day5_mmpd/mmpd_full_n20.tidy.csv",12)]:
    pids = sorted(pd.read_csv(csv).pid.unique()); res = {}
    for seed in range(10):
        sh = np.random.default_rng(seed).permutation(pids)
        res[seed] = {"cal": sorted(map(str, sh[:n_cal])), "test": sorted(map(str, sh[n_cal:]))}
    (out/f"{name}_cal{n_cal}_seeds0-9.json").write_text(json.dumps(res, indent=1)); print(name, len(pids), "subjects written")