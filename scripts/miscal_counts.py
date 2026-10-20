import pandas as pd, sys
for name in ["cp_ls","cp_st","cp_li","cp_sk","cp_ss","cp_ls_null","cp_ss_null"]:
    d = pd.read_csv(f"results/r2/{name}/cp_conditional_seeds.csv")
    d = d[d.score_type=="locally_adaptive"]
    cells = [c for c in d.columns if c in ("light","state","skin")]
    per_split = d.assign(mis=(d.coverage-0.9).abs()>0.05).groupby("seed").mis.sum()
    split_avg = d.groupby(cells).coverage.mean()
    n_mis = int(((split_avg-0.9).abs()>0.05).sum())
    print(f"{name}: split-avg {n_mis}/{len(split_avg)}  per-split mean {per_split.mean():.1f}  range {split_avg.min():.3f}-{split_avg.max():.3f}")
