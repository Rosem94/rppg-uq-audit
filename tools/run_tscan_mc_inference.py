"""Run MC Dropout inference with a trained TSCAN checkpoint.

Parallel to tools/run_mc_inference.py but adapted for TSCAN architecture.

Key differences from PhysNet variant:
  * Model class: TSCAN (with built-in 4 dropout layers at p=0.25 and p=0.5).
  * Input convention: 6-channel concatenated (3 DiffNormalized + 3 Standardized).
  * Native dropout rates (0.25/0.5) are kept for MC sampling --- no override.
    Rationale: existing UBFC-TSCAN checkpoint weights compensate for the
    training dropout magnitude; overriding to a smaller rate at inference
    introduces input-dependent bias in mean predictions. The sigma-blindness
    pattern documented in the paper is qualitative across cells, robust to
    absolute sigma magnitude, so cross-backbone comparison remains valid.

Config requirements:
  * TEST.DATA.PREPROCESS.DATA_TYPE must be ['DiffNormalized', 'Standardized']
  * TEST.DATA.PREPROCESS.RESIZE.H/W must match img_size of trained checkpoint
    (typically 72 for UbiComp Lab UBFC-TSCAN release).
  * MODEL.TSCAN.FRAME_DEPTH must match training (typically 20 for UBFC-TSCAN).

Usage:
    python tools/run_tscan_mc_inference.py \
        --config configs/infer_configs/DLCN_FULL_TEST_UBFC_TSCAN_MC.yaml \
        --checkpoint final_model_release/UBFC-rPPG_TSCAN.pth \
        --n_samples 20 \
        --output mc_results/dlcn_full_tscan_n20.pkl

Output pickle structure (identical to PhysNet variant for downstream compat):
    {
        subj_index_str: {
            chunk_idx_int: {
                "mean":  Tensor (T,)   -- mean per-timestep BVP prediction
                "std":   Tensor (T,)   -- per-timestep predictive std
                "label": Tensor (T,)   -- ground truth BVP
            }, ...
        }, ...
    }
"""

import argparse
import os
import pickle
import random
import sys

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on PYTHONPATH so we can import sibling packages.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from dataset import data_loader
from neural_methods.model.TS_CAN import TSCAN


# Reproducible seed for the dataloader generator (matches PhysNet variant).
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def enable_dropout(model):
    """Reactivate Dropout layers AFTER model.eval(). BatchNorm stays in eval.

    TSCAN does not use BatchNorm, but Conv2d layers default to eval-correct
    behavior, so this is a no-op for everything except dropout. Note: TSCAN
    has 4 built-in dropout layers (3x p=0.25 post-pooling, 1x p=0.5 post-FC).
    """
    n_active = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n_active += 1
    return n_active


@torch.no_grad()
def mc_predict_tscan(model, data, n_samples, frame_depth):
    """Run N stochastic forward passes through TSCAN with dropout active.

    Args:
        model:       TSCAN instance, weights loaded, dropout enabled.
        data:        Tensor (N, T, C, H, W) with C=6 (concat diff+raw).
        n_samples:   Number of MC samples.
        frame_depth: TSM segment length, used for truncation guard.

    Returns:
        mean: (N, T_eff) mean BVP prediction
        std:  (N, T_eff) per-timestep std
        T_eff: effective length after truncation (typically T)
    """
    N, T, C, H, W = data.shape
    nt = N * T

    # TSM requires nt divisible by frame_depth. Truncate if not.
    base_len = frame_depth
    truncate = (nt // base_len) * base_len
    data_flat = data.view(nt, C, H, W)[:truncate]

    samples = np.empty((n_samples, truncate), dtype=np.float32)
    for k in range(n_samples):
        out = model(data_flat)              # (truncate, 1)
        samples[k] = out.view(-1).cpu().numpy()

    mean = samples.mean(axis=0)              # (truncate,)
    std = samples.std(axis=0)                # (truncate,)

    # Reshape back to (N, T_eff) per-chunk format
    T_eff = truncate // N
    mean = mean[:N * T_eff].reshape(N, T_eff)
    std = std[:N * T_eff].reshape(N, T_eff)
    return mean, std, T_eff


def build_test_loader(config):
    """Construct the test DataLoader (same dispatcher pattern as PhysNet variant)."""
    name = config.TEST.DATA.DATASET
    if name == "UBFC-rPPG":
        loader_cls = data_loader.UBFCrPPGLoader.UBFCrPPGLoader
    elif name == "MMPD":
        loader_cls = data_loader.MMPDLoader.MMPDLoader
    elif name == "DLCN":
        loader_cls = data_loader.DLCNLoader.DLCNLoader
    elif name == "PURE":
        loader_cls = data_loader.PURELoader.PURELoader
    else:
        raise ValueError(f"Unsupported test dataset for MC inference: {name}")

    test_data = loader_cls(
        name="test",
        data_path=config.TEST.DATA.DATA_PATH,
        config_data=config.TEST.DATA,
        device=config.DEVICE,
    )
    gen = torch.Generator()
    gen.manual_seed(RANDOM_SEED)
    return DataLoader(
        dataset=test_data,
        num_workers=4,
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="MC Dropout inference for TSCAN."
    )
    parser.add_argument("--config", required=True,
                        help="Path to a yaml config (TEST.DATA section is used).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth file containing TSCAN state_dict.")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of stochastic forward passes (Gal recommends >=10).")
    parser.add_argument("--output", required=True,
                        help="Path to output pickle file (parent dir created if missing).")
    parser.add_argument("--img_size", type=int, default=72,
                        help="Image size used during training (typically 72).")
    parser.add_argument("--frame_depth", type=int, default=20,
                        help="TSM segment length (typically 20 for TSCAN).")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    class _ArgsShim:
        def __init__(self, cfg, opts):
            self.config_file = cfg
            self.opts = opts or []
    cfg_args = _ArgsShim(args.config, args.opts)
    config = get_config(cfg_args)

    print(f"=> Using config: {args.config}")
    print(f"=> Test dataset:  {config.TEST.DATA.DATASET}")
    print(f"=> Checkpoint:    {args.checkpoint}")
    print(f"=> Img size:      {args.img_size}")
    print(f"=> Frame depth:   {args.frame_depth}")
    print(f"=> N samples:     {args.n_samples}")
    print(f"=> Output:        {args.output}")
    print(f"=> Dropout rates: native (0.25/0.5, kept active during MC sampling)")

    # Build model + load checkpoint.
    device = torch.device(config.DEVICE)
    model = TSCAN(
        in_channels=3,
        frame_depth=args.frame_depth,
        img_size=args.img_size,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state_dict = torch.load(args.checkpoint, map_location=device)
    # rPPG-Toolbox checkpoints sometimes have 'module.' prefix from DataParallel.
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    n_dropout = enable_dropout(model)
    print(f"=> Activated {n_dropout} dropout layers for MC sampling.")

    # Quick sanity check that dropout is producing variance
    test_input = torch.randn(args.frame_depth, 6, args.img_size, args.img_size).to(device)
    with torch.no_grad():
        out1 = model(test_input)
        out2 = model(test_input)
    variance_check = (out1 - out2).abs().mean().item()
    if variance_check < 1e-6:
        raise RuntimeError(
            f"Dropout sanity check FAILED: variance between two forward passes "
            f"is {variance_check:.3e}, expected > 0. Dropout may not be active."
        )
    print(f"=> Dropout sanity OK: variance between two passes = {variance_check:.4e}")

    # Build test loader.
    test_loader = build_test_loader(config)

    # Run MC inference.
    results = {}
    for batch in tqdm(test_loader, desc="MC Dropout inference (TSCAN)"):
        data = batch[0].to(torch.float32).to(device)
        labels = batch[1]
        subj_indices = batch[2]
        sort_indices = batch[3]

        mean_pred, std_pred, t_eff = mc_predict_tscan(
            model, data,
            n_samples=args.n_samples,
            frame_depth=args.frame_depth,
        )

        for i in range(data.shape[0]):
            subj = str(subj_indices[i])
            chunk_idx = int(sort_indices[i])
            if subj not in results:
                results[subj] = {}
            # labels[i] is (T, 1) or (T,); slice to t_eff to match prediction
            label_i = labels[i].view(-1)[:t_eff]
            results[subj][chunk_idx] = {
                "mean":  torch.from_numpy(mean_pred[i].copy()),
                "std":   torch.from_numpy(std_pred[i].copy()),
                "label": label_i.clone(),
            }

    # Persist.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    n_subjects = len(results)
    n_chunks = sum(len(v) for v in results.values())
    print(f"\n=> Saved {n_chunks} chunks across {n_subjects} subjects to {args.output}")
    mean_sigma = np.mean(
        [v["std"].mean().item() for s in results.values() for v in s.values()]
    )
    print(f"=> Mean per-chunk sigma (TSCAN MC dropout proxy): {mean_sigma:.6f}")


if __name__ == "__main__":
    main()