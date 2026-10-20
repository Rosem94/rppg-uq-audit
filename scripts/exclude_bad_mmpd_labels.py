"""Excludes MMPD chunks whose reference label has zero variance from the split file lists used for the MMPD
model trained with the UBFC-matched learning rate (Sec. IV-B4).
"""
import glob
import os
import re

import numpy as np
import pandas as pd

CACHE_DIR = (
    r".\data\MMPD_preprocessed"
    r"\MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_"
    r"LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_"
    r"Dyamic_DetFalse_det_len30_Median_face_boxFalse"
)
LIST_DIR = os.path.join(CACHE_DIR, "..", "DataFileLists")
EXP_NAME = (
    "MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_"
    "LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_"
    "Dyamic_DetFalse_det_len30_Median_face_boxFalse"
)
SOURCE_CSV = os.path.join(LIST_DIR, f"{EXP_NAME}_0.0_1.0.csv")
OUT_CSV = os.path.join(LIST_DIR, f"{EXP_NAME}_0.0_1.0_CLEAN.csv")


def find_bad_label_files():
    label_files = sorted(glob.glob(os.path.join(CACHE_DIR, "*_label*.npy")))
    bad = []
    for lbl_path in label_files:
        lbl = np.load(lbl_path)
        if np.isnan(lbl).any() or np.isinf(lbl).any():
            bad.append(lbl_path)
        elif float(lbl.std()) < 1e-6:
            bad.append(lbl_path)
    return bad


def input_stub_from_label(label_path):
    """subjectX_..._label7.npy -> subjectX_..._input7 (matches build_file_list column)."""
    base = os.path.basename(label_path)
    m = re.match(r"(.+)_label(\d+)\.npy$", base)
    if not m:
        return None
    stem, idx = m.group(1), m.group(2)
    return f"{stem}_input{idx}"


def main():
    bad_labels = find_bad_label_files()
    print(f"Found {len(bad_labels)} bad (zero-std/NaN) label files")
    bad_stubs = {input_stub_from_label(p) for p in bad_labels}
    bad_stubs.discard(None)
    print("Bad chunk stubs:", sorted(bad_stubs))

    print(f"Reading: {SOURCE_CSV}")
    df = pd.read_csv(SOURCE_CSV, index_col=0)
    print(f"  Total rows: {len(df)}")

    def row_stub(input_file):
        base = os.path.basename(str(input_file))
        base = re.sub(r"\.npy$", "", base)
        base = re.sub(r"_input(\d+).*$", r"_input\1", base)
        return base

    df["_stub"] = df["input_files"].map(row_stub)
    mask_bad = df["_stub"].isin(bad_stubs)
    print(f"  Rows matching bad stubs: {mask_bad.sum()} (expected {len(bad_stubs)})")
    if mask_bad.sum() != len(bad_stubs):
        print("  WARNING: matched count != bad label count; inspect _stub values below")
        print(df.loc[mask_bad, ["input_files", "_stub"]].to_string())

    clean = df[~mask_bad].drop(columns=["_stub"]).reset_index(drop=True)
    clean.to_csv(OUT_CSV)
    print(f"  Wrote {len(clean)} rows (dropped {len(df) - len(clean)}) -> {OUT_CSV}")
    print("\nNext: regenerate the 0.0-0.7 / 0.7-0.85 / 0.85-1.0 split CSVs from this "
          "CLEAN file using generate_mmpd_csv_splits.py (point its `source` at "
          f"{os.path.basename(OUT_CSV)} instead of the original).")


if __name__ == "__main__":
    main()
