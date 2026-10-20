#!/usr/bin/env python3
"""Release check for rppg-uq-audit.

Run from the repository root:  python scripts/verify_release.py

1. Every repository path cited in README.md exists.
2. Table VIII operating points are recomputed for DLCN and MMPD and compared
   with results/day_r1_matched/op_*/operating_point_summary.csv.
3. The subject-level statistics of Sec. IV-E are recomputed for PhysNet,
   TSCAN and PhysFormer and compared with
   results/day_r1_matched/subject_level_stats_*/summary.json.
Outputs go to a temporary directory; nothing in the repository is modified.
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results" / "day_r1_matched"
results = []


def record(name, ok, detail=""):
    results.append(ok)
    print(f"{'PASS' if ok else 'FAIL'}  {name}" + (f"  ({detail})" if detail else ""))


def run(args):
    p = subprocess.run([sys.executable, *map(str, args)], cwd=ROOT,
                       capture_output=True, text=True)
    if p.returncode != 0:
        print(p.stdout[-2000:], p.stderr[-2000:])
    return p.returncode == 0


def numbers(obj, prefix=""):
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k != "input":
                yield from numbers(v, f"{prefix}.{k}")
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            yield from numbers(v, f"{prefix}[{i}]")
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        yield prefix, float(obj)


readme = (ROOT / "README.md").read_text(encoding="utf-8")
pattern = r"`((?:scripts|tools|results|configs|splits|cells|figures|toolbox_patches|checkpoints)/[^`*\s]*)`"
cited = sorted(set(re.findall(pattern, readme)))
missing = [p for p in cited if not (ROOT / p).exists()]
record(f"README paths exist ({len(cited)} cited)", not missing, ", ".join(missing))

with tempfile.TemporaryDirectory() as tmp:
    tmp = Path(tmp)
    ops = [("DLCN", R / "dlcn_main.tidy.csv", 30, ["light", "state"], "light", R / "op_dlcn_ls"),
           ("MMPD", R / "mmpd_main" / "mmpd_full_n20.tidy.csv", 12, ["skin"], "skin", R / "op_mmpd_skin")]
    for name, csv, ncal, cells, group, ref in ops:
        out = tmp / f"op_{name}"
        ok = run(["scripts/operating_point.py", "--csv", csv, "--out_dir", out,
                  "--n_cal_subjects", ncal, "--cell_cols", *cells, "--group_col", group])
        if ok:
            a = pd.read_csv(out / "operating_point_summary.csv").sort_values(["criterion", "rho"]).reset_index(drop=True)
            b = pd.read_csv(ref / "operating_point_summary.csv").sort_values(["criterion", "rho"]).reset_index(drop=True)
            try:
                pd.testing.assert_frame_equal(a, b, check_dtype=False, check_like=True, rtol=0, atol=1e-10)
            except AssertionError as e:
                ok = False
                print(str(e)[:500])
        record(f"Table VIII operating points, {name}", ok)

    for bb, args in [("physnet", ["scripts/subject_level_stats.py", "--out_dir"]),
                     ("tscan", ["scripts/subject_level_stats_bb.py", R / "mmpd_tscan_per_trial_lacp.csv"]),
                     ("physformer", ["scripts/subject_level_stats_bb.py", R / "mmpd_physformer_per_trial_lacp.csv"])]:
        out = tmp / f"sls_{bb}"
        ok = run([*args, out])
        detail = ""
        if ok:
            got = dict(numbers(json.loads((out / "summary.json").read_text(encoding="utf-8"))))
            ref = dict(numbers(json.loads((R / f"subject_level_stats_{bb}" / "summary.json").read_text(encoding="utf-8"))))
            bad = [k for k in ref if k not in got or not np.isclose(got[k], ref[k], rtol=0, atol=1e-9)]
            ok = not bad and set(got) == set(ref)
            detail = f"{len(ref)} values" if ok else f"mismatch: {bad[:5]}"
        record(f"Sec. IV-E subject-level statistics, {bb}", ok, detail)

print(f"\n{sum(results)}/{len(results)} checks passed")
sys.exit(0 if all(results) else 1)