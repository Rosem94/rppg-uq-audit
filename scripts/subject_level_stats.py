#!/usr/bin/env python3
"""Subject-level statistics for the MMPD skin-tone coverage analysis.

Input: per-trial LACP records written by conformal_prediction.py, one row per
(trial, calibration split) in which the trial was a test trial.

1. A trial is identified by (pid, skin, state, light, exercise); its coverage
   is the mean of `covered` over the splits in which it was tested.
2. The subject is the resampling and permutation unit. Skin tone is annotated
   per video: a subject with trials under two tones contributes each trial
   under its own annotation and is excluded from any contrast in which it
   would fall in both groups.
3. Primary contrast, Fitzpatrick 3 minus 6: trial-weighted and
   equal-subject-weighted estimates, stratified subject-cluster bootstrap CI,
   one-sided subject-level permutation p, GEE logistic regression with
   subject-clustered robust standard errors.
4. Pairwise contrasts, Fitzpatrick 3 minus 4, 5 and 6 (equal subject weight),
   one-sided permutation p, Holm adjustment over the family of three.
5. Coverage of the (Fitzpatrick 6, walking) cell with a subject-cluster
   bootstrap CI.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

TRIAL_KEYS = ["pid", "skin", "state", "light", "exercise"]
REFERENCE_TONE = 3
COMPARISON_TONES = (4, 5, 6)
PRIMARY_TONE = 6
WALKING = 4
FAMILY_ALPHA = 0.05


def load_trials(path):
    raw = pd.read_csv(path)
    missing = set(TRIAL_KEYS + ["covered", "seed"]) - set(raw.columns)
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    if "score_type" in raw.columns:
        types = set(raw["score_type"].dropna().unique())
        if types != {"locally_adaptive"}:
            raise ValueError(f"expected locally_adaptive records only, found {types}")
    if raw.duplicated(TRIAL_KEYS + ["seed"]).any():
        raise ValueError("duplicate (trial, seed) records")
    if not raw["covered"].isin([0, 1]).all():
        raise ValueError("covered must be 0/1")
    trials = (raw.groupby(TRIAL_KEYS, as_index=False)["covered"].mean()
                 .rename(columns={"covered": "cov"}))
    return raw, trials


def subject_table(trials):
    g = trials.groupby("pid")["cov"]
    return pd.DataFrame({"s": g.sum(), "n": g.size().astype(float), "m": g.mean()})


def contrast_groups(trials, tone_a, tone_b):
    a = trials[trials["skin"] == tone_a]
    b = trials[trials["skin"] == tone_b]
    overlap = sorted(set(a["pid"]) & set(b["pid"]))
    a = a[~a["pid"].isin(overlap)]
    b = b[~b["pid"].isin(overlap)]
    return subject_table(a), subject_table(b), overlap


def _stat(s, n, m, mask, weight):
    if weight == "pooled":
        return s[mask].sum() / n[mask].sum() - s[~mask].sum() / n[~mask].sum()
    return m[mask].mean() - m[~mask].mean()


def estimate(A, B, weight):
    both = pd.concat([A.assign(g=1), B.assign(g=0)]).sort_index()
    mask = both["g"].values == 1
    return float(_stat(both["s"].values, both["n"].values, both["m"].values, mask, weight))


def bootstrap_ci(A, B, weight, n_boot, seed):
    rng = np.random.default_rng(seed)
    iA = rng.integers(0, len(A), size=(n_boot, len(A)))
    iB = rng.integers(0, len(B), size=(n_boot, len(B)))
    if weight == "pooled":
        d = (A["s"].values[iA].sum(1) / A["n"].values[iA].sum(1)
             - B["s"].values[iB].sum(1) / B["n"].values[iB].sum(1))
    else:
        d = A["m"].values[iA].mean(1) - B["m"].values[iB].mean(1)
    lo, hi = np.percentile(d, [2.5, 97.5])
    return float(lo), float(hi)


def permutation_p(A, B, weight, n_perm, seed):
    rng = np.random.default_rng(seed + 1)
    both = pd.concat([A.assign(g=1), B.assign(g=0)]).sort_index()
    s, n, m = both["s"].values, both["n"].values, both["m"].values
    mask = both["g"].values == 1
    k, K = int(mask.sum()), len(both)
    obs = _stat(s, n, m, mask, weight)
    idx = rng.random((n_perm, K)).argsort(1)[:, :k]
    if weight == "pooled":
        sA, nA = s[idx].sum(1), n[idx].sum(1)
        null = sA / nA - (s.sum() - sA) / (n.sum() - nA)
    else:
        null = m[idx].mean(1) - (m.sum() - m[idx].sum(1)) / (K - k)
    return float((1 + np.sum(null >= obs)) / (1 + n_perm))


def gee(raw, tone_a, tone_b, exclude):
    import statsmodels.api as sm
    import statsmodels.formula.api as smf
    d = raw[raw["skin"].isin([tone_a, tone_b]) & ~raw["pid"].isin(exclude)].copy()
    d["is_b"] = (d["skin"] == tone_b).astype(int)
    d["y"] = d["covered"].astype(int)
    res = smf.gee("y ~ is_b", groups=d["pid"], data=d,
                  family=sm.families.Binomial(),
                  cov_struct=sm.cov_struct.Exchangeable()).fit()
    b0, b1 = res.params["Intercept"], res.params["is_b"]
    expit = lambda x: 1.0 / (1.0 + np.exp(-x))
    return {"coef": float(b1), "robust_se": float(res.bse["is_b"]),
            "p": float(res.pvalues["is_b"]),
            "implied_cov_diff": float(expit(b0) - expit(b0 + b1))}


def cell_coverage(trials, tone, state, n_boot, seed):
    rng = np.random.default_rng(seed + 2)
    st = subject_table(trials[(trials["skin"] == tone) & (trials["state"] == state)])
    s, n = st["s"].values, st["n"].values
    idx = rng.integers(0, len(st), size=(n_boot, len(st)))
    boot = s[idx].sum(1) / n[idx].sum(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return {"coverage": float(s.sum() / n.sum()), "ci": [float(lo), float(hi)],
            "n_subjects": int(len(st)), "n_trials": int(n.sum()),
            "per_subject": {k: round(float(v), 3) for k, v in st["m"].items()}}


def holm(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    adj = np.empty_like(p)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - rank) * p[i]))
        adj[i] = running
    return adj


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--csv", type=Path,
                    default=Path("results/day_r1_matched/mmpd_per_trial_lacp.csv"))
    ap.add_argument("--out_dir", type=Path,
                    default=Path("results/day_r1_matched/subject_level_stats"))
    ap.add_argument("--n_boot", type=int, default=20000)
    ap.add_argument("--n_perm", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    raw, trials = load_trials(a.csv)
    tones = trials.groupby("pid")["skin"].nunique()
    dual = sorted(tones[tones > 1].index)

    A, B, overlap = contrast_groups(trials, REFERENCE_TONE, PRIMARY_TONE)
    primary = {"n_subjects": [len(A), len(B)], "excluded_overlap": overlap}
    for w in ("pooled", "subject_mean"):
        primary[w] = {"estimate": estimate(A, B, w),
                      "ci": list(bootstrap_ci(A, B, w, a.n_boot, a.seed)),
                      "p_one_sided": permutation_p(A, B, w, a.n_perm, a.seed)}
    primary["gee"] = gee(raw, REFERENCE_TONE, PRIMARY_TONE, overlap)

    rows = []
    for t in COMPARISON_TONES:
        A, B, overlap = contrast_groups(trials, REFERENCE_TONE, t)
        rows.append({"contrast": f"{REFERENCE_TONE}-{t}",
                     "n_ref": len(A), "n_cmp": len(B),
                     "excluded_overlap": ";".join(overlap),
                     "estimate_pp": 100 * estimate(A, B, "subject_mean"),
                     "ci_low_pp": 100 * bootstrap_ci(A, B, "subject_mean", a.n_boot, a.seed)[0],
                     "ci_high_pp": 100 * bootstrap_ci(A, B, "subject_mean", a.n_boot, a.seed)[1],
                     "p_one_sided": permutation_p(A, B, "subject_mean", a.n_perm, a.seed)})
    pw = pd.DataFrame(rows)
    pw["p_holm"] = holm(pw["p_one_sided"].values)
    pw["reject_holm_0.05"] = pw["p_holm"] <= FAMILY_ALPHA
    pw.to_csv(a.out_dir / "pairwise_holm.csv", index=False)

    walking = cell_coverage(trials, PRIMARY_TONE, WALKING, a.n_boot, a.seed)

    summary = {"input": str(a.csv), "n_trials": int(len(trials)),
               "dual_annotated_subjects": dual,
               "n_boot": a.n_boot, "n_perm": a.n_perm, "seed": a.seed,
               "primary_3_minus_6": primary,
               "pairwise": pw.to_dict(orient="records"),
               "fitz6_walking": walking}
    (a.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    pp = lambda x: f"{100 * x:.2f}"
    lines = [f"Input: {a.csv}", f"Unique trials: {len(trials)}",
             f"Dual-annotated subjects: {dual}", "",
             f"Fitzpatrick 3 minus 6, subjects {primary['n_subjects'][0]} / {primary['n_subjects'][1]}"]
    for w in ("pooled", "subject_mean"):
        r = primary[w]
        lines.append(f"  {w:12s} {pp(r['estimate'])} pp  95% CI [{pp(r['ci'][0])}, {pp(r['ci'][1])}]"
                     f"  one-sided permutation p = {r['p_one_sided']:.4f}")
    g = primary["gee"]
    lines += [f"  GEE coef {g['coef']:.3f}  robust SE {g['robust_se']:.3f}  p = {g['p']:.4f}", "",
              "Pairwise contrasts (equal subject weight, Holm over 3):",
              pw.round(4).to_string(index=False), "",
              f"Fitzpatrick 6, walking: {walking['coverage']:.3f}  95% CI "
              f"[{walking['ci'][0]:.3f}, {walking['ci'][1]:.3f}]  subjects {walking['n_subjects']}",
              f"  per subject: {walking['per_subject']}"]
    report = "\n".join(lines)
    (a.out_dir / "report.txt").write_text(report + "\n", encoding="utf-8")
    print(report)
    print(f"\nSaved to {a.out_dir}")


if __name__ == "__main__":
    main()