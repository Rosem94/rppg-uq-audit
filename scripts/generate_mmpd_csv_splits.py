"""Writes the subject-level train/validation/test file lists for MMPD from the preprocessed cache, in the
format read by rPPG-Toolbox.
"""
import pandas as pd
from pathlib import Path

base = Path(r".\data\MMPD_preprocessed\DataFileLists")
exp_name = "MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

source = base / f"{exp_name}_0.0_1.0.csv"
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
    out_path = base / f"{exp_name}_{label}.csv"
    subset_df.to_csv(out_path)
    print(f"  Wrote {len(subset_df)} rows -> {out_path.name}")

print("\nDone.")
