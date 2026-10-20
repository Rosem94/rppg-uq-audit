import pandas as pd, glob, os
rows = []
for axis in ["skin", "state", "light"]:
    for kind in ["real", "shuf"]:
        d = os.path.join("results", "day_r1_matched", f"null_{axis}_{kind}")
        f = [x for x in glob.glob(os.path.join(d, "*.csv")) if "cond" in os.path.basename(x)]
        if not f:
            print("no conditional csv in", d); continue
        c = pd.read_csv(f[0])
        c = c[c.score_type == "locally_adaptive"]
        key = [x for x in ["skin", "state", "light"] if x in c.columns][0]
        g = c.groupby(["seed", key]).coverage.mean().reset_index()
        per_seed = g.groupby("seed").apply(
            lambda s: pd.Series({"lo": s.coverage.min(), "hi": s.coverage.max(),
                                 "mis": int(((s.coverage - 0.90).abs() > 0.05).sum())}))
        rows.append(dict(axis=axis, kind=kind,
                         lo=per_seed.lo.mean(), hi=per_seed.hi.mean(),
                         mis=per_seed.mis.mean()))
print(pd.DataFrame(rows).round(3).to_string(index=False))
