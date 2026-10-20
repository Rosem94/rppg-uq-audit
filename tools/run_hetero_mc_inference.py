"""Run MC Dropout inference with a trained PhysNetHetero checkpoint.

Like tools/run_mc_inference.py but for the heteroscedastic model: in addition to
the epistemic (MC-dropout) mean/std and raw per-pass samples, it also saves the
aleatoric per-timestep log-variance head output (averaged across passes).

Usage:
    python tools/run_hetero_mc_inference.py \
        --config configs/infer_configs/DLCN_FULL_MC.yaml \
        --checkpoint runs/exp/.../UBFC_PhysNet_Hetero_EpochNN.pth \
        --n_samples 20 \
        --output mc_results/dlcn_full_hetero_n20.pkl

Output pickle structure:
    {
        subj_index_str: {
            chunk_idx_int: {
                "mean":    Tensor (T,)      -- mean BVP prediction
                "std":     Tensor (T,)      -- per-ts epistemic std (sigma_MCD)
                "label":   Tensor (T,)      -- ground truth BVP
                "samples": Tensor (N, T)    -- raw per-pass BVP (trial-consistent sigma)
                "log_var": Tensor (T,)      -- aleatoric log-variance head output
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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from config import get_config
from dataset import data_loader
from neural_methods.model.PhysNetHetero import (
    PhysNetHetero,
    enable_dropout,
    mc_predict_hetero,
)

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
    parser = argparse.ArgumentParser(description="MC Dropout inference for PhysNetHetero.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--n_samples", type=int, default=20)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dropout_rate", type=float, default=None)
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
    config.defrost()
    if args.dropout_rate is not None:
        config.MODEL.PHYSNET.DROPOUT_RATE = args.dropout_rate
    config.freeze()

    print(f"=> Config:     {args.config}")
    print(f"=> Test set:   {config.TEST.DATA.DATASET}")
    print(f"=> Checkpoint: {args.checkpoint}")
    print(f"=> N samples:  {args.n_samples}")
    print(f"=> Output:     {args.output}")
    print(f"=> Dropout:    {config.MODEL.PHYSNET.DROPOUT_RATE}")

    device = torch.device(config.DEVICE)
    model = PhysNetHetero(
        frames=config.MODEL.PHYSNET.FRAME_NUM,
        dropout_rate=config.MODEL.PHYSNET.DROPOUT_RATE,
    ).to(device)

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    state_dict = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    enable_dropout(model)

    test_loader = build_test_loader(config)

    results = {}
    for batch in tqdm(test_loader, desc="Hetero MC inference"):
        data = batch[0].to(torch.float32).to(device)
        labels = batch[1]
        subj_indices = batch[2]
        sort_indices = batch[3]

        mean_pred, std_pred, samples, logvar_mean = mc_predict_hetero(
            model, data, n_samples=args.n_samples)
        mean_pred = mean_pred.cpu()
        std_pred = std_pred.cpu()
        samples = samples.cpu()          # (N, B, T)
        logvar_mean = logvar_mean.cpu()  # (B, T)

        for i in range(data.shape[0]):
            subj = str(subj_indices[i])
            chunk_idx = int(sort_indices[i])
            if subj not in results:
                results[subj] = {}
            results[subj][chunk_idx] = {
                "mean":    mean_pred[i].clone(),
                "std":     std_pred[i].clone(),
                "label":   labels[i].clone(),
                "samples": samples[:, i, :].clone(),   # (N, T)
                "log_var": logvar_mean[i].clone(),     # (T,)
            }

    out_dir = os.path.dirname(os.path.abspath(args.output))
    os.makedirs(out_dir, exist_ok=True)
    with open(args.output, "wb") as f:
        pickle.dump(results, f)
    n_subjects = len(results)
    n_chunks = sum(len(v) for v in results.values())
    mean_alea = np.mean([
        float(torch.exp(v["log_var"]).mean().sqrt())
        for s in results.values() for v in s.values()
    ])
    mean_epi = np.mean([
        v["std"].mean().item() for s in results.values() for v in s.values()
    ])
    print(f"\n=> Saved {n_chunks} chunks across {n_subjects} subjects to {args.output}")
    print(f"=> Mean per-chunk epistemic std (sigma_MCD):   {mean_epi:.6f}")
    print(f"=> Mean per-chunk aleatoric sigma (sqrt E[var]): {mean_alea:.6f}")


if __name__ == "__main__":
    main()
