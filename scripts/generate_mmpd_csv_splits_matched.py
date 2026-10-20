"""Writes the label-audited MMPD file lists (suffix _CLEAN) for the model trained with the UBFC-matched
learning rate; the original file lists are left unchanged.
"""
import pandas as pd
from pathlib import Path

base = Path(r".\data\MMPD_preprocessed\DataFileLists")
exp_name = "MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"
out_exp_name = exp_name + "_CLEAN"

source = base / f"{exp_name}_0.0_1.0_CLEAN.csv"
print(f"Reading: {source.name}")
df = pd.read_csv(source, index_col=0)
print(f"  Total rows: {len(df)}")

df["subject_id"] = df["input_files"].str.extract(r"subject(\d+)_").astype(int)
subjects = sorted(df["subject_id"].unique())
n = len(subjects)
print(f"  Unique subjects: {n} ({subjects[0]} to {subjects[-1]})")

splits = [
    (0.0, 0.7, "0.0_0.7"),
    (0.7, 0.85, "0.7_0.85"),
    (0.85, 1.0, "0.85_1.0"),
]

for begin, end, label in splits:
    start_idx = int(begin * n)
    end_idx = int(end * n)
    subset_subjects = subjects[start_idx:end_idx]
    print(f"\nSplit {label}: subjects {subset_subjects[0]}-{subset_subjects[-1]} (n={len(subset_subjects)})")

    subset_df = df[df["subject_id"].isin(subset_subjects)].drop(columns=["subject_id"]).reset_index(drop=True)
    # Write under a DISTINCT exp_name (_CLEAN suffix) so the original priority-1
    # original splits (checkpoint Epoch28) are never touched or overwritten.
    out_path = base / f"{out_exp_name}_{label}.csv"
    subset_df.to_csv(out_path)
    print(f"  Wrote {len(subset_df)} rows -> {out_path.name}")

print("\nDone. These _CLEAN splits are separate files; the original "
      "splits used by MMPD_MMPD_PHYSNET_MCDROPOUT.yaml are untouched. "
      "MMPD_MMPD_PHYSNET_MCDROPOUT_MATCHED.yaml's EXP_DATA_NAME must be set to "
      "'_CLEAN' so the loader reads these files instead.")
