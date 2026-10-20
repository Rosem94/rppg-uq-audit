"""The dataloader for DLCN dataset.

DLCN: rPPG dataset captured under Dynamic Lighting Conditions at Night.
Reference: Li et al. 2025, arXiv:2507.04306.

File structure: flat directory of P{participant}_{trial}.h5 files.
Each .h5 contains pre-cropped 128x128 RGB face video at 30 fps,
synchronized BVP signal, plus HR and SpO2 (1 Hz).

Total: 98 participants x 8 trials = 784 files (one participant has only 4).

Trial -> (lighting_scenario, state) mapping (empirically verified on P10, P50,
and consistent with paper Section III.A protocol):
    trial 1 -> (FI&FP, rest)        trial 5 -> (FI&FP, post-exercise)
    trial 2 -> (VI&FP, rest)        trial 6 -> (VI&FP, post-exercise)
    trial 3 -> (FI&VP, rest)        trial 7 -> (FI&VP, post-exercise)
    trial 4 -> (VI&VP, rest)        trial 8 -> (VI&VP, post-exercise)

Lighting codes:
    1 = FI&FP (Fixed Intensity, Fixed Position)        -- control / easy
    2 = VI&FP (Varying Intensity, Fixed Position)       -- intensity flicker
    3 = FI&VP (Fixed Intensity, Varying Position)       -- moving shadows
    4 = VI&VP (Varying Intensity, Varying Position)     -- hardest

State codes:
    1 = rest          (lower HR, ~60-90 BPM)
    2 = post-exercise (higher HR, ~90-130 BPM)

Notes:
    - Frames are already face-cropped and resized to 128x128 by dataset authors.
      Set DO_CROP_FACE: False in config preprocessing block.
    - SpO2 available in .h5 but unused (rPPG/BVP task only).
    - P63 has only 4 trials (rest block, no exercise). Loader handles gracefully.
"""

import glob
import os
import re

import h5py
import numpy as np
from dataset.data_loader.BaseLoader import BaseLoader
import pandas as pd


class DLCNLoader(BaseLoader):
    """The data loader for the DLCN dataset."""

    def __init__(self, name, data_path, config_data, device=None):
        """Initializes a DLCN dataloader.

        Args:
            data_path(str): folder containing flat P{N}_{M}.h5 files.
            name(str): name of the dataloader (train/valid/test).
            config_data(CfgNode): data settings (ref: config.py).
        """
        self.info = config_data.INFO
        super().__init__(name, data_path, config_data, device)

    def get_raw_data(self, data_path):
        """Returns data dict list under the path (for DLCN dataset).

        Each entry: {"index": "P10_1", "path": "...P10_1.h5",
                     "participant": 10, "trial": 1}
        """
        h5_files = glob.glob(os.path.join(data_path, "P*_*.h5"))
        if not h5_files:
            raise ValueError(self.dataset_name + " data paths empty!")

        dirs = []
        pattern = re.compile(r"P(\d+)_(\d+)\.h5$")
        for h5_path in h5_files:
            filename = os.path.basename(h5_path)
            match = pattern.match(filename)
            if match is None:
                continue  # skip any stray files
            participant = int(match.group(1))
            trial = int(match.group(2))
            lighting, state = self.trial_to_metadata(trial)
            dirs.append({
                "index": f"P{participant}_T{trial}_L{lighting}_S{state}",
                "path": h5_path,
                "participant": participant,
                "trial": trial,
            })
        return dirs

    def split_raw_data(self, data_dirs, begin, end):
        """Returns a subset of data dirs, split by PARTICIPANT id.

        Splits along participant boundaries to prevent subject leakage between
        train/valid/test (all trials of a given participant stay together).
        """
        if begin == 0 and end == 1:
            return data_dirs

        by_participant = {}
        for entry in data_dirs:
            pid = entry["participant"]
            by_participant.setdefault(pid, []).append(entry)

        participant_ids = sorted(by_participant.keys())
        n_participants = len(participant_ids)
        chosen_range = range(int(begin * n_participants), int(end * n_participants))
        chosen_pids = [participant_ids[i] for i in chosen_range]
        print(f"used participant ids for split: {chosen_pids}")

        data_dirs_new = []
        for pid in chosen_pids:
            data_dirs_new.extend(by_participant[pid])
        return data_dirs_new

    def preprocess_dataset_subprocess(self, data_dirs, config_preprocess, i, file_list_dict):
        """Invoked by preprocess_dataset for multi-process workers."""
        entry = data_dirs[i]
        trial = entry["trial"]
        lighting, state = self.trial_to_metadata(trial)

        # Filename embeds metadata for INFO-based filtering at load time.
        # Pattern: P{participant}_T{trial}_L{lighting}_S{state}
        saved_filename = (
            f"P{entry['participant']}"
            f"_T{trial}"
            f"_L{lighting}"
            f"_S{state}"
        )

        frames, bvps = self.read_h5(entry["path"])

        if config_preprocess.USE_PSUEDO_PPG_LABEL:
            bvps = self.generate_pos_psuedo_labels(frames, fs=self.config_data.FS)

        # NOTE: BaseLoader.preprocess will run face crop + resize + chunking.
        # DLCN frames are already pre-cropped to 128x128, so config must have
        # DO_CROP_FACE: False. RESIZE in config still applies (e.g. to 72x72).
        frames_clips, bvps_clips = self.preprocess(frames, bvps, config_preprocess)

        input_name_list, label_name_list = self.save_multi_process(
            frames_clips, bvps_clips, saved_filename
        )
        file_list_dict[i] = input_name_list

    def load_preprocessed_data(self):
        """Loads the preprocessed data, applying INFO filters from config.

        Mirrors MMPDLoader.load_preprocessed_data: parses metadata from the
        saved filename and includes only chunks whose (lighting, state) match
        the INFO whitelists in config.
        """
        file_list_path = self.file_list_path
        file_list_df = pd.read_csv(file_list_path)
        inputs_temp = file_list_df['input_files'].tolist()

        inputs = []
        for each_input in inputs_temp:
            # filename pattern: P{p}_T{t}_L{l}_S{s}_input{chunk}.npy
            base = os.path.split(each_input)[-1].split('.')[0]
            parts = base.split('_')
            # parts: ['P10', 'T3', 'L3', 'S1', 'input0']
            try:
                lighting = int(parts[2][1:])
                state = int(parts[3][1:])
            except (IndexError, ValueError):
                # Defensive: skip any chunk whose name does not parse.
                continue
            if (lighting in self.info.LIGHT) and (state in self.info.STATE):
                inputs.append(each_input)

        if not inputs:
            raise ValueError(self.dataset_name + ' dataset loading data error! '
                             'Check INFO filters in config (LIGHT, STATE).')
        inputs = sorted(inputs)
        labels = [f.replace("input", "label") for f in inputs]
        self.inputs = inputs
        self.labels = labels
        self.preprocessed_data_len = len(inputs)

    @staticmethod
    def trial_to_metadata(trial):
        """Maps trial index 1-8 to (lighting, state) based on DLCN protocol.

        Mapping verified empirically on P10 and P50:
            trials 1-4 = rest, trials 5-8 = post-exercise
            lighting = ((trial - 1) % 4) + 1  -- cycles FI&FP, VI&FP, FI&VP, VI&VP

        Returns:
            lighting: 1=FI&FP, 2=VI&FP, 3=FI&VP, 4=VI&VP
            state:    1=rest, 2=post-exercise
        """
        if not 1 <= trial <= 8:
            raise ValueError(f"DLCN trial must be in [1, 8], got {trial}")
        lighting = ((trial - 1) % 4) + 1
        state = 1 if trial <= 4 else 2
        return lighting, state

    @staticmethod
    def read_h5(h5_file):
        """Reads a DLCN .h5 file. Returns (frames, bvp).

        frames: (T, 128, 128, 3) uint8, RGB.
        bvp:    (T,) float64, synchronized with frames at 30 fps.
        """
        with h5py.File(h5_file, "r") as f:
            frames = np.asarray(f["imgs"])
            bvps = np.asarray(f["bvp"])
        return frames, bvps