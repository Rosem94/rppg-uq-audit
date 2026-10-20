"""Run MC Dropout inference with a trained PhysNet_MCDropout checkpoint.

Loads a trained model, runs N stochastic forward passes per test sample,
saves per-chunk (mean, std, label) tuples to a pickle for downstream analysis
(uncertainty vs error scatter, calibration metrics, selective prediction curves).

Usage:
    python tools/run_mc_inference.py \
        --config configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml \
        --checkpoint runs/exp/.../UBFC_PhysNet_MCDropout_Epoch29.pth \
        --n_samples 20 \
        --output mc_results/ubfc_test_mc20.pkl

The output pickle structure:
    {
        subj_index_str: {
            chunk_idx_int: {
                "mean":  Tensor (T,)   -- mean rPPG prediction
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
from torch.utils.data import DataLoader
from tqdm import tqdm

# Ensure project root is on PYTHONPATH so we can import sibling packages.
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from dataset import data_loader
from neural_methods.model.PhysNet_MCDropout import (
    PhysNet_MCDropout,
    enable_dropout,
    mc_predict,
)


# Reproducible seed for the dataloader generator.
RANDOM_SEED = 100
torch.manual_seed(RANDOM_SEED)
torch.cuda.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def build_test_loader(config):
    """Construct the test DataLoader using the same dispatcher as main.py."""
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
        num_workers=4,                # lower than train (no contention with bg job)
        batch_size=config.INFERENCE.BATCH_SIZE,
        shuffle=False,
        worker_init_fn=seed_worker,
        generator=gen,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="MC Dropout inference for PhysNet_MCDropout."
    )
    parser.add_argument("--config", required=True,
                        help="Path to a yaml config (TEST.DATA section is used).")
    parser.add_argument("--checkpoint", required=True,
                        help="Path to .pth file containing model state_dict.")
    parser.add_argument("--n_samples", type=int, default=20,
                        help="Number of stochastic forward passes (Gal recommends >=10).")
    parser.add_argument("--output", required=True,
                        help="Path to output pickle file (parent dir created if missing).")
    parser.add_argument("--dropout_rate", type=float, default=None,
                        help="Override MODEL.PHYSNET.DROPOUT_RATE from config.")
    # Allow yacs-style key=value overrides for --config compatibility with main.py.
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER)
    return parser.parse_args()


def main():
    args = parse_args()

    # main.py defines RANDOM_SEED globally and sets cudnn flags;
    # add_data_loader_args / add_trainer_args are not strictly required for inference.
    # We construct the config exactly as main.py would.
    class _ArgsShim:
        def __init__(self, cfg, opts):
            self.config_file = cfg
            self.opts = opts or []
    cfg_args = _ArgsShim(args.config, args.opts)
    config = get_config(cfg_args)
    config.defrost()
    if args.dropout_rate is not None:
        config.MODEL.PHYSNET.DROPOUT_RATE = args.dropout_rate
    config.freeze()

    print(f"=> Using config: {args.config}")
    print(f"=> Test dataset:  {config.TEST.DATA.DATASET}")
    print(f"=> Checkpoint:    {args.checkpoint}")
    print(f"=> N samples:     {args.n_samples}")
    print(f"=> Output:        {args.output}")
    print(f"=> Dropout rate:  {config.MODEL.PHYSNET.DROPOUT_RATE}")

    # Build model + load checkpoint.
    device = torch.device(config.DEVICE)
    model = PhysNet_MCDropout(
        frames=config.MODEL.PHYSNET.FRAME_NUM,
        dropout_rate=config.MODEL.PHYSNET.DROPOUT_RATE,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    enable_dropout(model)        # critical: keeps dropout active during inference

    # Build test loader.
    test_loader = build_test_loader(config)

    # Run MC inference.
    results = {}
    for batch in tqdm(test_loader, desc="MC Dropout inference"):
        data = batch[0].to(torch.float32).to(device)
        labels = batch[1]
        subj_indices = batch[2]
        sort_indices = batch[3]

        mean_pred, std_pred, samples = mc_predict(model, data, n_samples=args.n_samples)
        mean_pred = mean_pred.cpu()
        std_pred = std_pred.cpu()
        samples = samples.cpu()          # (n_samples, B, T): raw per-pass rPPG

        for i in range(data.shape[0]):
            subj = str(subj_indices[i])
            chunk_idx = int(sort_indices[i])
            if subj not in results:
                results[subj] = {}
            results[subj][chunk_idx] = {
                "mean":    mean_pred[i].clone(),
                "std":     std_pred[i].clone(),
                "label":   labels[i].clone(),
                "samples": samples[:, i, :].clone(),   # (n_samples, T) for comment-3
            }

    # Persist.
    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    n_subjects = len(results)
    n_chunks = sum(len(v) for v in results.values())
    print(f"\n=> Saved {n_chunks} chunks across {n_subjects} subjects to {args.output}")
    print(f"=> Mean of per-chunk std (epistemic uncertainty proxy): "
          f"{np.mean([v['std'].mean().item() for s in results.values() for v in s.values()]):.6f}")


if __name__ == "__main__":
    main()