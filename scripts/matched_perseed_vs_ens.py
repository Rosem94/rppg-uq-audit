import pandas as pd
from scipy.stats import spearmanr

for ds in ["dlcn", "mmpd"]:
    w = pd.read_csv(rf"results\day_r1_matched\matched_{ds}.csv")
    print(f"\n{ds.upper()}: per-seed rho(cons_s, ensemble |err|)  [single model, matched target]")
    rhos = []
    for s in range(5):
        rho, _ = spearmanr(w[f"cons_{s}"], w.ens_err)
        rhos.append(rho)
        print(f"  seed{s}: {rho:+.3f}")
    print(f"  mean: {sum(rhos)/5:+.3f}   vs sigma_E: {spearmanr(w.sigma_E, w.ens_err)[0]:+.3f}")
