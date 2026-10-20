#!/usr/bin/env python3
"""Backbone variant of subject_level_stats.py.

Usage: python scripts/subject_level_stats_bb.py <per_trial_lacp.csv> [out_dir]

Runs exactly the same procedure as scripts/subject_level_stats.py on the
per-trial LACP records of another backbone (TSCAN, PhysFormer).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import subject_level_stats

if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    csv = Path(sys.argv[1])
    out = Path(sys.argv[2]) if len(sys.argv) > 2 else csv.parent / (csv.stem + "_subject_level_stats")
    sys.argv = [sys.argv[0], "--csv", str(csv), "--out_dir", str(out)]
    subject_level_stats.main()