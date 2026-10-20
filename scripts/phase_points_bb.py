"""Phase-space points (8 partitions) for any backbone from its trial-level tidy CSV."""
import argparse, numpy as np, pandas as pd
from pathlib import Path

PARTS = [("DLCN", "dlcn_state", ["state"]), ("DLCN", "dlcn_light", ["light"]),
         ("DLCN", "dlcn_light_state", ["light", "state"]),
         ("MMPD", "mmpd_state", ["state"]), ("MMPD", "mmpd_light", ["light"]),
         ("MMPD", "mmpd_skin", ["skin"]), ("MMPD", "mmpd_skin_state", ["skin", "state"]),
         ("MMPD", "mmpd_light_state", ["light", "state"])]
NCAL = {"DLCN": 30, "MMPD": 12}


def cq(s, alpha):
    n = len(s); q = min(np.ceil((n + 1) * (1 - alpha)) / n, 1.0)
    return np.quantile(s, q, method="higher")


ap = argparse.ArgumentParser()
ap.add_argument("--dlcn_csv", required=True); ap.add_argument("--mmpd_csv", required=True)
ap.add_argument("--tag", required=True); ap.add_argument("--alpha", type=float, default=0.1)
a = ap.parse_args()
CSV = {"DLCN": a.dlcn_csv, "MMPD": a.mmpd_csv}
rows = []
for ds, name, cols in PARTS:
    df = pd.read_csv(CSV[ds]); subs = sorted(df.pid.unique()); ncal = NCAL[ds]
    cell_mae = df.groupby(cols).abs_err.mean()
    worst = cell_mae.idxmax(); worst = worst if isinstance(worst, tuple) else (worst,)
    het = cell_mae.max() / cell_mae.min()
    rec = []
    for seed in range(10):
        sh = np.random.default_rng(seed).permutation(subs)
        cal = df[df.pid.isin(sh[:ncal])].copy(); te = df[df.pid.isin(sh[ncal:])].copy()
        q, m = {}, {}
        for k, g in cal.groupby(cols):
            k = k if isinstance(k, tuple) else (k,)
            q[k] = cq(g.abs_err.values / g.unc_mean.values, a.alpha); m[k] = g.abs_err.mean()
        cd = pd.DataFrame([{**dict(zip(cols, k)), "q": q[k], "m": m[k]} for k in q])
        te = te.merge(cd, on=cols, how="left")
        te["s_naive"] = te.unc_mean
        te["s_mondrian_cp"] = te.q * te.unc_mean
        te["s_cellmae_sigma"] = te.m * te.unc_mean
        n = len(te); k = int(np.ceil(0.5 * n)); out = {"rho1": (te.abs_err.mean(), 1.0)}
        for cr in ["naive", "mondrian_cp", "cellmae_sigma"]:
            o = te.sort_values(f"s_{cr}").reset_index(drop=True)
            wm = np.ones(len(o), bool)
            for c, v in zip(cols, worst):
                wm &= (o[c] == v).values
            out[cr] = (o.iloc[:k].abs_err.mean(), wm[:k].sum() / max(wm.sum(), 1))
        rec.append(out)
    mae = lambda cr: np.mean([r[cr][0] for r in rec])
    wret = lambda cr: np.mean([r[cr][1] for r in rec])
    rows.append(dict(backbone=a.tag, dataset=ds, config=name,
                     n_per_cell=round(ncal * df.groupby("pid").size().mean() / cell_mae.size, 1),
                     het_full=het, worst_cell=str(worst), mae_rho1=mae("rho1"), mae_naive=mae("naive"),
                     mae_mondrian=mae("mondrian_cp"), mae_cellmae=mae("cellmae_sigma"),
                     vs_random_mondrian=mae("rho1") - mae("mondrian_cp"),
                     vs_random_cellmae=mae("rho1") - mae("cellmae_sigma"),
                     benefit_mondrian=mae("naive") - mae("mondrian_cp"),
                     benefit_cellmae=mae("naive") - mae("cellmae_sigma"),
                     abst=wret("naive") - wret("mondrian_cp"),
                     abst_cellmae=wret("naive") - wret("cellmae_sigma")))
out = pd.DataFrame(rows).round(3)
Path("results/day_r1_matched").mkdir(exist_ok=True)
out.to_csv(f"results/day_r1_matched/phase_points_{a.tag}.csv", index=False)
print(out.to_string(index=False))
