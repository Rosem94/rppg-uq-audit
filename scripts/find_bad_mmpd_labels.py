import numpy as np
import glob
import os

cache_dir = r".\data\MMPD_preprocessed\MMPD_SizeW72_SizeH72_ClipLength160_DataTypeDiffNormalized_DataAugNone_LabelTypeDiffNormalized_Crop_faceTrue_BackendHC_Large_boxTrue_Large_size1.5_Dyamic_DetFalse_det_len30_Median_face_boxFalse"

label_files = sorted(glob.glob(os.path.join(cache_dir, "*_label*.npy")))
print(f"Total label files: {len(label_files)}")

nan_inf = []
zero_std = []
near_zero_std = []

for lbl_path in label_files:
    lbl = np.load(lbl_path)
    if np.isnan(lbl).any() or np.isinf(lbl).any():
        nan_inf.append((lbl_path, "nan/inf"))
    else:
        std = float(lbl.std())
        if std < 1e-6:
            zero_std.append((lbl_path, std))
        elif std < 0.01:
            near_zero_std.append((lbl_path, std))

print(f"\nNaN/Inf labels: {len(nan_inf)}")
print(f"Zero-std labels (std<1e-6): {len(zero_std)}")
print(f"Near-zero-std labels (std<0.01): {len(near_zero_std)}")

if zero_std:
    print(f"\nFirst 10 zero-std labels:")
    for path, std in zero_std[:10]:
        print(f"  std={std:.2e}  {os.path.basename(path)}")

if near_zero_std:
    print(f"\nFirst 10 near-zero-std labels:")
    for path, std in near_zero_std[:10]:
        print(f"  std={std:.4f}  {os.path.basename(path)}")
