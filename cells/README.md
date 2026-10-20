# Cell definitions used in the paper

All partitions are Cartesian products of the columns below, read from the tidy per-trial CSVs
(`results/day5_mmpd/mmpd_full_n20.tidy.csv`, `results/day_r1_matched/dlcn_main.tidy.csv`).
Calibration/test subject lists for the ten conformal splits are in `splits/*_seeds0-9.json`
(seed k: `np.random.default_rng(k).permutation(sorted(pid))`; the first n_cal subjects form the calibration set).
`mmpd_cells_full.csv` and `dlcn_cells.csv` give the number of rows and subjects per cell.

## MMPD (33 subjects, 647 trials; 12 calibration subjects)
| column | codes | partitions in the paper |
| --- | --- | --- |
| `light` | 1 LED-low, 2 LED-high, 3 incandescent, 4 natural | lighting-only; lighting x state |
| `state` | 1 stationary, 2 rotation, 3 talking, 4 walking | state-only; lighting x state; skin x state |
| `skin` | Fitzpatrick 3, 4, 5, 6 (annotated per video; one subject appears under 3 and 4) | skin-only; skin x state |
| `exercise` | 1 = ..., 2 = ... | not used as a cell axis |

## DLCN (98 subjects, 780 trials; 30 calibration subjects)
| column | codes | partitions in the paper |
| --- | --- | --- |
| `light` | 1 FI&FP, 2 VI&FP, 3 FI&VP, 4 VI&VP (F fixed, V variable; I illumination, P projector) | lighting-only; lighting x state |
| `state` | 1 rest, 2 exercise | state-only; lighting x state |

Binary skin grouping of Sec. V-c: {3,4} -> 0, {5,6} -> 1 (`scripts/label_sensitivity.py --coarse_map "3:0,4:0,5:1,6:1"`).
Sigma floor for the trial-consistent proxy: `scripts/conformal_prediction.py --sigma_col unc_consistent --sigma_floor 0.879`.