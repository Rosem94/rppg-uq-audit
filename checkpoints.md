# Checkpoints

Validation-best checkpoints of the ten PhysNet runs reported in the paper (state dicts, about 3 MB each). Verify with `sha256sum <file>` or `Get-FileHash <file> -Algorithm SHA256`.

| File | Run | Training configuration | Epoch | SHA-256 |
| --- | --- | --- | --- | --- |
| `checkpoints/UBFC_PhysNet_MCDropout.pth` | UBFC-rPPG, MC dropout (main model) | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 17 | `71886bf41bd7ea55258f2880ec5153521d7367148403b43df989abdd9d37353f` |
| `checkpoints/UBFC_PhysNet_MCDropout_seed0.pth` | UBFC-rPPG, MC dropout (main model), ensemble seed 0 | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 17 | `4634e62c2bc0f54c2d2d5165a6a92360c8eee354dada3137277add9c204a93aa` |
| `checkpoints/UBFC_PhysNet_MCDropout_seed1.pth` | UBFC-rPPG, MC dropout (main model), ensemble seed 1 | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 15 | `dd453cf7e42dc723b749fe58d2cca48f65d9d9893f7d27dc6dc6b59b1b403321` |
| `checkpoints/UBFC_PhysNet_MCDropout_seed2.pth` | UBFC-rPPG, MC dropout (main model), ensemble seed 2 | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 25 | `a16c7cbaccc1c5f7580d677f9c1aa050c9c5554d224c36466578b1c06a926ab0` |
| `checkpoints/UBFC_PhysNet_MCDropout_seed3.pth` | UBFC-rPPG, MC dropout (main model), ensemble seed 3 | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 29 | `dc58a4c8d5350a02b232e51053d27a9d10f1139ab2867b03c836041111c6a71d` |
| `checkpoints/UBFC_PhysNet_MCDropout_seed4.pth` | UBFC-rPPG, MC dropout (main model), ensemble seed 4 | `configs/train_configs/UBFC_UBFC_PHYSNET_MCDROPOUT.yaml` | 13 | `fd6751e1aceff15360476fe8e26e46178f83199031a37fec2802695063c38d0e` |
| `checkpoints/UBFC_PhysNet_Hetero.pth` | UBFC-rPPG, heteroscedastic head | `configs/train_configs/UBFC_UBFC_PHYSNET_HETERO.yaml` | 18 | `3b9195670b2fa81474dfc0ad4c8eaae96c8d09a6f60617eb11ef9a9587be46b2` |
| `checkpoints/MMPD_PhysNet_MCDropout.pth` | MMPD, learning rate 1e-5 | `configs/train_configs/MMPD_MMPD_PHYSNET_MCDROPOUT.yaml` | 28 | `3f9af11a7c1a305e4f8c1a309a6c85b947de9ac54824bce8f23583f31fa679a4` |
| `checkpoints/MMPD_PhysNet_MCDropout_Matched.pth` | MMPD, UBFC-matched learning rate 5e-5 | `configs/train_configs/MMPD_MMPD_PHYSNET_MCDROPOUT_MATCHED.yaml` | 28 | `dcfbe8c61ac51fcdcd0b058df14185eeababb07e3347452e2473d2fc14469b37` |
| `checkpoints/DLCN_PhysNet_MCDropout_InDist.pth` | DLCN, in-distribution | `configs/train_configs/DLCN_DLCN_PHYSNET_MCDROPOUT_INDIST.yaml` | 14 | `dbb2d8e589b08d30fce0ed256f6c7b97208c37de334c27dd8a62a36d282f0da3` |

The ensemble seeds share the main training configuration and differ only in the random seed. TSCAN and PhysFormer use the official rPPG-Toolbox weights `final_model_release/UBFC-rPPG_TSCAN.pth` and `final_model_release/UBFC-rPPG_PhysFormer_DiffNormalized.pth`, which are not redistributed.
