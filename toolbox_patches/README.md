# rPPG-Toolbox modifications

The patches in this directory were validated against upstream
[rPPG-Toolbox](https://github.com/ubicomplab/rPPG-Toolbox) commit
`b7500b848f84ad7f86e277b4612563b69f4f88f9`.
Relative paths are preserved: `toolbox_patches/<path>` replaces `<path>` in the toolbox.

| Toolbox path | Change |
| --- | --- |
| `config.py` | Adds `INFO.STATE` and the MC-dropout / heteroscedastic configuration fields |
| `main.py` | Registers the DLCN loader and the two PhysNet variants |
| `dataset/data_loader/DLCNLoader.py` | New: DLCN loader with subject-level splits and per-trial metadata |
| `dataset/data_loader/__init__.py` | Registers `DLCNLoader` |
| `neural_methods/model/PhysNet_MCDropout.py` | New: PhysNet with `Dropout3d` after each block, kept active at inference |
| `neural_methods/model/PhysNetHetero.py` | New: PhysNet with a parallel log-variance head |
| `neural_methods/loss/HeteroscedasticLoss.py` | New: negative-Pearson plus Gaussian NLL loss |
| `neural_methods/trainer/PhysnetMCDropoutTrainer.py` | New: trainer with MC-dropout prediction |
| `neural_methods/trainer/PhysnetHeteroTrainer.py` | New: trainer for the heteroscedastic model |
| `neural_methods/trainer/__init__.py` | Registers the two trainers |

```bash
git clone https://github.com/ubicomplab/rPPG-Toolbox.git
cd rPPG-Toolbox
git checkout b7500b848f84ad7f86e277b4612563b69f4f88f9
cp -r /path/to/rppg-uq-audit/toolbox_patches/* .
cp -r /path/to/rppg-uq-audit/tools .
```

The inference drivers in `tools/` import the toolbox packages and must be run from the toolbox root.
These files are modifications of rPPG-Toolbox and remain under its Responsible AI Source Code License, version 1.1.