import numpy as np
import glob
import os

cache_dir = r".\data\MMPD_preprocessed\MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

input_files = sorted(glob.glob(os.path.join(cache_dir, "*_input*.npy")))
print(f"Total input files: {len(input_files)}")
print(f"Checking first 20...")

nan_inp = 0
nan_lbl = 0
zero_std_lbl = 0
for inp_path in input_files[:20]:
    lbl_path = inp_path.replace("_input", "_label")
    inp = np.load(inp_path)
    
    has_nan_inp = bool(np.isnan(inp).any())
    has_inf_inp = bool(np.isinf(inp).any())
    if has_nan_inp or has_inf_inp:
        nan_inp += 1
        print(f"  BAD INPUT: {os.path.basename(inp_path)} nan={has_nan_inp} inf={has_inf_inp}")
    
    if os.path.exists(lbl_path):
        lbl = np.load(lbl_path)
        has_nan_lbl = bool(np.isnan(lbl).any())
        has_inf_lbl = bool(np.isinf(lbl).any())
        lbl_std = float(lbl.std())
        if has_nan_lbl or has_inf_lbl:
            nan_lbl += 1
            print(f"  BAD LABEL: {os.path.basename(lbl_path)} nan={has_nan_lbl} inf={has_inf_lbl} std={lbl_std:.4f}")
        elif lbl_std < 1e-6:
            zero_std_lbl += 1
            print(f"  ZERO-STD LABEL: {os.path.basename(lbl_path)} std={lbl_std:.6f}")

print(f"\nSummary: {nan_inp}/20 NaN inputs, {nan_lbl}/20 NaN labels, {zero_std_lbl}/20 zero-std labels")

print(f"\nSample input stats: shape={inp.shape}, mean={inp.mean():.4f}, std={inp.std():.4f}, min={inp.min():.4f}, max={inp.max():.4f}")
if os.path.exists(lbl_path):
    lbl = np.load(lbl_path)
    print(f"Sample label stats: shape={lbl.shape}, mean={lbl.mean():.4f}, std={lbl.std():.4f}, min={lbl.min():.4f}, max={lbl.max():.4f}")
