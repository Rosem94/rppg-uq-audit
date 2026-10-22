# rppg-uq-audit

Code, configurations, per-trial results and figures for

> Myrzashova *et al.*, "Cell-Aware Conformal Prediction for Demographic Reliability Assessment in Remote Photoplethysmography," *IEEE Journal of Biomedical and Health Informatics*, under review.

Every number in the paper is recomputed by the scripts in this repository from the per-trial result files in `results/`; no statistic is hard-coded.

## Quick check (CPU only, no datasets)

On Windows, enable long paths before cloning: `git config --global core.longpaths true` (some file lists in `splits/` have long rPPG-Toolbox names).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements-analysis.txt
python scripts/verify_release.py
```

`verify_release.py` checks that every path cited in this README exists, recomputes the operating points of Table VIII and the subject-level statistics of Sec. IV-E for all three backbones from the shipped per-trial files, and compares them with the stored results. It writes nothing into the repository. Tested with Python 3.10 on Windows.

## Repository layout

| Path | Contents |
| --- | --- |
| `scripts/` | Analysis, statistics and figure scripts |
| `tools/` | Monte Carlo dropout inference drivers for PhysNet, TSCAN, PhysFormer and the heteroscedastic PhysNet |
| `toolbox_patches/` | Modified rPPG-Toolbox files and the toolbox commit they apply to ([README](toolbox_patches/README.md)) |
| `configs/` | Training and inference configurations of the reported runs |
| `results/` | Per-trial (`*.tidy.csv`, `*_per_trial_lacp.csv`) and aggregated results, one directory per experiment |
| `splits/` | Subject-level calibration/test splits and seeds |
| `cells/` | Cell definitions and attribute codes for every partition ([README](cells/README.md)) |
| `figures/paper/` | Figures 1-3 of the paper |
| `figures/supplementary/` | The two diagnostic figures removed from the manuscript for length |
| `checkpoints/`, `checkpoints.md` | Trained PhysNet checkpoints and their SHA-256 checksums |

## Reproducing the paper

There are two levels. Level 1 needs only this repository: every table, statistic and figure is recomputed from the per-trial files in `results/`. Level 2 regenerates those per-trial files from the videos and needs the datasets, a GPU and the patched rPPG-Toolbox.

### Level 1: from per-trial results

| Paper item | Script | Inputs / outputs |
| --- | --- | --- |
| Table II, HR accuracy | `scripts/analyze_ubfc_intra.py`, `scripts/analyze_dlcn_full_v2.py`, `scripts/analyze_mmpd_full.py` | `results/day5_ubfc/`, `results/day3_dlcn_full/`, `results/day5_mmpd/` |
| Table III, cross-backbone | `scripts/day9_cross_backbone_summary.py`, `scripts/analyze_dlcn_tscan_full.py`, `scripts/analyze_mmpd_tscan_full.py` | `results/day8_tscan/`, `results/priority2/`, `results/day9_cross_backbone/` |
| Sec. IV-B4, cross-training-source and in-distribution controls | `scripts/analyze_mmpd_trained_inference.py`, `scripts/analyze_indist_mmpd.py`, `scripts/analyze_indist_dlcn.py`, `scripts/find_bad_mmpd_labels.py`, `scripts/exclude_bad_mmpd_labels.py` | `results/priority1/`, `results/r4_8_indist/`, `results/r1_2_matched/`, `results/r4_8_indist_dlcn/` |
| Sec. IV-B5, UBFC reference split | `scripts/analyze_ubfc_intra.py` | `results/day10_12subj_ubfc/` |
| Sec. IV-C, Table IV, coverage and permutation control | `scripts/conformal_prediction.py`, `scripts/miscal_counts.py`, `scripts/summarize_null_control.py` | `results/r2/`, `results/day6/` |
| Sec. IV-C4, trial-consistent uncertainty | `scripts/analyze_sigma_consistency.py`, `scripts/conformal_prediction.py` (`--sigma_col unc_consistent --sigma_floor 0.879`) | `results/day_r1_matched/` |
| Sec. IV-C5, CQR baseline | `scripts/cqr_baseline.py` | `results/day_r1_matched/` |
| Fig. 1, Sec. IV-D, selective prediction | `scripts/selective_prediction.py`, `scripts/plot_selective_combined.py` | `results/day_r1_matched/dlcn_main/selective_seeds.csv`, `results/day6/mmpd_skin_only/selective_seeds_mmpd_skin_only.csv` |
| Table V, Fig. 2, two-condition criterion | `scripts/check_all_benefits.py`, `scripts/phase_points.py`, `scripts/phase_points_bb.py`, `scripts/make_fig4_phase_24.py` | `results/day6/`, `results/day7/`, `results/day12_baseline/`, `results/day_r1_matched/` |
| Table VI, accuracy and calibration by skin tone | `scripts/analyze_mmpd_full.py`, `scripts/analyze_sigma_consistency.py`, `scripts/conformal_prediction.py` | `results/day5_mmpd/`, `results/day6/` |
| Sec. IV-E3-E5, subject-level statistics | `scripts/subject_level_stats.py`, `scripts/subject_level_stats_bb.py` | `results/day_r1_matched/subject_level_stats_physnet/`, `results/day_r1_matched/subject_level_stats_tscan/`, `results/day_r1_matched/subject_level_stats_physformer/` |
| Table VII, subject-cluster CIs | `scripts/all_cells_ci_table.py` | `results/day_r1_matched/mmpd_per_trial_lacp.csv` -> `results/day_r1_matched/all_cells_ci.csv` (all 20 skin x motion cells) |
| Sec. IV-F, Fig. 3, aggregation, ensemble and heteroscedastic controls | `scripts/matched_sigma_analysis.py`, `scripts/matched_perseed_vs_ens.py`, `scripts/subsample_5pass_control.py`, `scripts/day10_ensemble_uq.py`, `scripts/analyze_dlcn_hetero.py`, `scripts/analyze_mmpd_hetero.py`, `scripts/make_fig6_matched.py` | `results/day10_ensemble/`, `results/day5_dlcn/`, `results/day_r1_matched/`, `results/day11_figs/` |
| Secs. V-B and V-C, label corruption and missing labels | `scripts/label_sensitivity.py` | `results/day_r1_matched/` |
| Table VIII, operating points | `scripts/operating_point.py` | `results/day_r1_matched/op_dlcn_ls/`, `results/day_r1_matched/op_mmpd_skin/` |
| Supplementary figures | `scripts/make_fig2_calibration.py`, `scripts/make_fig3_cp_coverage.py` | `results/day3_dlcn_full/`, `results/day4_cp/` |
| Splits and seeds | `scripts/export_conformal_splits.py`, `scripts/generate_mmpd_csv_splits.py`, `scripts/generate_mmpd_csv_splits_matched.py` | `splits/` |

Examples, run from the repository root:

```bash
python scripts/operating_point.py --csv results/day_r1_matched/dlcn_main.tidy.csv \
    --out_dir out/op_dlcn --n_cal_subjects 30 --cell_cols light state --group_col light
python scripts/operating_point.py --csv results/day_r1_matched/mmpd_main/mmpd_full_n20.tidy.csv \
    --out_dir out/op_mmpd --n_cal_subjects 12 --cell_cols skin --group_col skin
python scripts/subject_level_stats.py --out_dir out/subject_level_stats_physnet
python scripts/subject_level_stats_bb.py results/day_r1_matched/mmpd_tscan_per_trial_lacp.csv out/subject_level_stats_tscan
python scripts/plot_selective_combined.py --dlcn_csv results/day_r1_matched/dlcn_main/selective_seeds.csv \
    --mmpd_csv results/day6/mmpd_skin_only/selective_seeds_mmpd_skin_only.csv --dlcn_worst "L4, S2" --mmpd_worst skin6 \
    --out_dir out --basename fig1_selective_combined
python scripts/make_fig4_phase_24.py
python scripts/make_fig6_matched.py
```

Conformal splits use `np.random.default_rng(seed).permutation(sorted(pid))` for seeds 0-9; the first 30 (DLCN) or 12 (MMPD) subjects form the calibration set. The subject-level statistics average each trial's coverage over the splits in which it was tested and resample subjects; a subject annotated under two skin tones contributes each trial under its own annotation and is excluded from any contrast in which it would fall in both groups.

### Level 2: from videos

1. Obtain the datasets (below) and install rPPG-Toolbox at the commit given in [toolbox_patches/README.md](toolbox_patches/README.md), with the patched files copied in.
2. Train with the configurations in `configs/train_configs/`, or use the checkpoints in `checkpoints/`.
3. Run MC-dropout inference with `tools/run_mc_inference.py`, `tools/run_tscan_mc_inference.py`, `tools/run_physformer_mc_inference.py` or `tools/run_hetero_mc_inference.py` and the configurations in `configs/infer_configs/`; then convert the pickles to per-trial CSVs with the `scripts/analyze_*` scripts of Table II and Table III.
4. Continue with Level 1.

Dataset paths in the configurations are local; edit `DATA_PATH` and `CACHED_PATH` before running.

## Data

The datasets are public and are not redistributed here:

- UBFC-rPPG, https://sites.google.com/view/ybenezeth/ubfcrppg
- MMPD, https://github.com/McJackTang/MMPD_rPPG_dataset
- DLCN, from its authors under their data-use agreement

## Checkpoints

`checkpoints/` holds the validation-best checkpoints of the ten PhysNet runs reported in the paper (UBFC-rPPG MC-dropout and its five ensemble seeds, the heteroscedastic head, MMPD at the original and at the UBFC-matched learning rate, DLCN in-distribution); `checkpoints.md` gives the training configuration, epoch and SHA-256 of each file. TSCAN and PhysFormer use the official rPPG-Toolbox weights `final_model_release/UBFC-rPPG_TSCAN.pth` and `final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth`, which are not redistributed.

## License

Original code: MIT License. Files under `toolbox_patches/` are modifications of rPPG-Toolbox and remain under its Responsible AI Source Code License, version 1.1.

## Citation

See [CITATION.cff](CITATION.cff); a journal reference will be added on publication.

## Contact

Raushan-Rose Myrzashova, Hefei University of Technology, rose.myrzashova@gmail.com