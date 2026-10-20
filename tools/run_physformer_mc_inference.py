"""Run MC Dropout inference with a trained PhysFormer checkpoint.

Parallel to tools/run_tscan_mc_inference.py but adapted for PhysFormer.

Key differences from TSCAN variant:
  * Model class: ViT_ST_ST_Compact3_TDC_gra_sharp from neural_methods.model.PhysFormer.
  * Input convention: 3-channel DiffNormalized in NCDHW format (B, 3, T, H, W).
    No 6-channel concatenation. No TSM frame-depth truncation.
  * Forward signature: model(x, gra_sharp) returns (rPPG, Score1, Score2, Score3).
    Only rPPG is used; attention scores are discarded.
  * Native dropout 0.1 across 36 sites (3 per block x 12 blocks). Kept
    active during MC sampling -- same rationale as TSCAN.

Config requirements:
  * TEST.DATA.PREPROCESS.DATA_TYPE must be ['DiffNormalized']
  * TEST.DATA.PREPROCESS.RESIZE.H/W must be 128 (PhysFormer fixed input size).
  * TEST.DATA.PREPROCESS.CHUNK_LENGTH must be 160 (PhysFormer fixed temporal length).
  * TEST.DATA.DATA_FORMAT must be NCDHW.
  * MODEL.PHYSFORMER.* must match the trained checkpoint.

Usage:
    python tools/run_physformer_mc_inference.py \
        --config configs/infer_configs/DLCN_FULL_TEST_UBFC_PHYSFORMER.yaml \
        --checkpoint final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth \
        --n_samples 20 \
        --output mc_results/dlcn_full_physformer_n20.pkl

Output pickle structure (identical to PhysNet/TSCAN variants):
    {
        subj_index_str: {
            chunk_idx_int: {
                "mean":  Tensor (T,)
                "std":   Tensor (T,)
                "label": Tensor (T,)
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

# Ensure project root is on PYTHONPATH.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from dataset import data_loader
from neural_methods.model.PhysFormer import ViT_ST_ST_Compact3_TDC_gra_sharp


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

    PhysFormer has 36 dropout sites (12 transformer blocks x 3 per block:
    1 in attention softmax, 1 in proj output, 1 in feed-forward output).
    """
    n_active = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d)):
            m.train()
            n_active += 1
    return n_active


@torch.no_grad()
def mc_predict_physformer(model, data, n_samples, gra_sharp=2.0):
    """Run N stochastic forward passes through PhysFormer with dropout active.

    Args:
        model:       PhysFormer instance, weights loaded, dropout enabled.
        data:        Tensor (N, C, T, H, W) with C=3 (DiffNormalized).
        n_samples:   Number of MC samples.
        gra_sharp:   PhysFormer gradient-sharpening hyperparameter (canonical 2.0).

    Returns:
        mean: (N, T) mean BVP prediction
        std:  (N, T) per-timestep std
    """
    N, C, T, H, W = data.shape

    samples = np.empty((n_samples, N, T), dtype=np.float32)
    for k in range(n_samples):
        # PhysFormer returns (rPPG, Score1, Score2, Score3); discard attention scores
        rPPG, *_ = model(data, gra_sharp)
        # rPPG shape: (N, T)
        samples[k] = rPPG.cpu().numpy()

    mean = samples.mean(axis=0)              # (N, T)
    std = samples.std(axis=0)                # (N, T)
    return mean, std


def build_test_loader(config):
    """Construct the test DataLoader (same dispatcher pattern as PhysNet/TSCAN variants)."""
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


def build_physformer(config, device):
    """Construct PhysFormer with parameters from config."""
    T = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
    H = config.TEST.DATA.PREPROCESS.RESIZE.H
    W = config.TEST.DATA.PREPROCESS.RESIZE.W

    patch_size = config.MODEL.PHYSFORMER.PATCH_SIZE
    dim = config.MODEL.PHYSFORMER.DIM
    ff_dim = config.MODEL.PHYSFORMER.FF_DIM
    num_heads = config.MODEL.PHYSFORMER.NUM_HEADS
    num_layers = config.MODEL.PHYSFORMER.NUM_LAYERS
    theta = config.MODEL.PHYSFORMER.THETA
    dropout_rate = config.MODEL.DROP_RATE

    model = ViT_ST_ST_Compact3_TDC_gra_sharp(
        image_size=(T, H, W),
        patches=(patch_size, patch_size, patch_size),
        dim=dim,
        ff_dim=ff_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout_rate=dropout_rate,
        theta=theta,
        in_channels=3,
        frame=T,
    ).to(device)
    return model


def parse_args():
    parser = argparse.ArgumentParser(description="MC Dropout inference for PhysFormer.")
    parser.add_argument("--config", required=True,
                        help="Path to a yaml config (TEST.DATA section is used).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth file containing PhysFormer state_dict.")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of stochastic forward passes (Gal recommends >=10).")
    parser.add_argument("--output", required=True,
                        help="Path to output pickle file (parent dir created if missing).")
    parser.add_argument("--gra_sharp", type=float, default=2.0,
                        help="PhysFormer gradient-sharpening hyperparameter (canonical 2.0).")
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

    print(f"=> Using config:  {args.config}")
    print(f"=> Test dataset:  {config.TEST.DATA.DATASET}")
    print(f"=> Checkpoint:    {args.checkpoint}")
    print(f"=> Image size:    {config.TEST.DATA.PREPROCESS.RESIZE.H}x{config.TEST.DATA.PREPROCESS.RESIZE.W}")
    print(f"=> Chunk length:  {config.TEST.DATA.PREPROCESS.CHUNK_LENGTH}")
    print(f"=> Dropout rate:  {config.MODEL.DROP_RATE} (native, kept active during MC sampling)")
    print(f"=> gra_sharp:     {args.gra_sharp}")
    print(f"=> N samples:     {args.n_samples}")
    print(f"=> Output:        {args.output}")

    # Build model + load checkpoint.
    device = torch.device(config.DEVICE)
    model = build_physformer(config, device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    state_dict = torch.load(args.checkpoint, map_location=device)
    if any(k.startswith("module.") for k in state_dict):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model.eval()
    n_dropout = enable_dropout(model)
    print(f"=> Activated {n_dropout} dropout layers for MC sampling.")

    # Sanity check that dropout is producing variance
    T = config.TEST.DATA.PREPROCESS.CHUNK_LENGTH
    H = config.TEST.DATA.PREPROCESS.RESIZE.H
    W = config.TEST.DATA.PREPROCESS.RESIZE.W
    test_input = torch.randn(1, 3, T, H, W).to(device)
    with torch.no_grad():
        out1, *_ = model(test_input, args.gra_sharp)
        out2, *_ = model(test_input, args.gra_sharp)
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
    for batch in tqdm(test_loader, desc="MC Dropout inference (PhysFormer)"):
        data = batch[0].to(torch.float32).to(device)
        labels = batch[1]
        subj_indices = batch[2]
        sort_indices = batch[3]

        mean_pred, std_pred = mc_predict_physformer(
            model, data,
            n_samples=args.n_samples,
            gra_sharp=args.gra_sharp,
        )

        for i in range(data.shape[0]):
            subj = str(subj_indices[i])
            chunk_idx = int(sort_indices[i])
            if subj not in results:
                results[subj] = {}
            label_i = labels[i].view(-1)
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
    print(f"=> Mean per-chunk sigma (PhysFormer MC dropout proxy): {mean_sigma:.6f}")


if __name__ == "__main__":
    main()