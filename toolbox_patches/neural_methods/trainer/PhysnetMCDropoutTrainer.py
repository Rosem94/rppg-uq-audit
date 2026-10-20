"""PhysNet MC Dropout Trainer.

Identical to PhysnetTrainer except for:
1. Imports PhysNet_MCDropout (with Dropout3d in feature extractor) instead of
   the baseline PhysNet model.
2. Reads MODEL.PHYSNET.DROPOUT_RATE from config (default 0.1).
3. Provides predict_with_uncertainty() method for MC Dropout inference at test time.

The training loop, optimizer, scheduler, loss, valid, save logic are unchanged.
This separation keeps the baseline PhysnetTrainer untouched while allowing direct
comparison "PhysNet vs PhysNet+MC Dropout" in the paper.
"""
import os
from collections import OrderedDict

import numpy as np
import torch
import torch.optim as optim
from evaluation.metrics import calculate_metrics
from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson
from neural_methods.model.PhysNet_MCDropout import (
    PhysNet_MCDropout,
    enable_dropout,
    mc_predict,
)
from neural_methods.trainer.BaseTrainer import BaseTrainer
from torch.autograd import Variable
from tqdm import tqdm


class PhysnetMCDropoutTrainer(BaseTrainer):

    def __init__(self, config, data_loader):
        """Inits parameters from args and the writer for TensorboardX."""
        super().__init__()
        self.device = torch.device(config.DEVICE)
        self.max_epoch_num = config.TRAIN.EPOCHS
        self.model_dir = config.MODEL.MODEL_DIR
        self.model_file_name = config.TRAIN.MODEL_FILE_NAME
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.num_of_gpu = config.NUM_OF_GPU_TRAIN
        self.base_len = self.num_of_gpu
        self.config = config
        self.min_valid_loss = None
        self.best_epoch = 0

        # MC Dropout-specific config (with sane defaults for backward compat).
        dropout_rate = getattr(config.MODEL.PHYSNET, "DROPOUT_RATE", 0.1)

        self.model = PhysNet_MCDropout(
            frames=config.MODEL.PHYSNET.FRAME_NUM,
            dropout_rate=dropout_rate,
        ).to(self.device)

        if config.TOOLBOX_MODE == "train_and_test":
            self.num_train_batches = len(data_loader["train"])
            self.loss_model = Neg_Pearson()
            self.optimizer = optim.Adam(
                self.model.parameters(), lr=config.TRAIN.LR)
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer, max_lr=config.TRAIN.LR,
                epochs=config.TRAIN.EPOCHS, steps_per_epoch=self.num_train_batches)
        elif config.TOOLBOX_MODE == "only_test":
            pass
        else:
            raise ValueError("PhysNet MC Dropout trainer initialized in incorrect toolbox mode!")

    def train(self, data_loader):
        """Training routine for model"""
        if data_loader["train"] is None:
            raise ValueError("No data for train")

        mean_training_losses = []
        mean_valid_losses = []
        lrs = []
        for epoch in range(self.max_epoch_num):
            print('')
            print(f"====Training Epoch: {epoch}====")
            running_loss = 0.0
            train_loss = []
            self.model.train()
            tbar = tqdm(data_loader["train"], ncols=80)
            for idx, batch in enumerate(tbar):
                tbar.set_description("Train epoch %s" % epoch)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    batch[0].to(torch.float32).to(self.device))
                BVP_label = batch[1].to(torch.float32).to(self.device)
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)
                loss = self.loss_model(rPPG, BVP_label)
                if torch.isnan(loss) or torch.isinf(loss):
                    self.optimizer.zero_grad()
                    tbar.set_postfix(loss=float('nan'))
                    continue
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                running_loss += loss.item()
                if idx % 100 == 99:
                    print(f'[{epoch}, {idx + 1:5d}] loss: {running_loss / 100:.3f}')
                    running_loss = 0.0
                train_loss.append(loss.item())
                lrs.append(self.scheduler.get_last_lr())
                self.optimizer.step()
                self.scheduler.step()
                self.optimizer.zero_grad()
                tbar.set_postfix(loss=loss.item())

            mean_training_losses.append(np.mean(train_loss))
            self.save_model(epoch)
            if not self.config.TEST.USE_LAST_EPOCH:
                valid_loss = self.valid(data_loader)
                mean_valid_losses.append(valid_loss)
                print('validation loss: ', valid_loss)
                if self.min_valid_loss is None:
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
                elif (valid_loss < self.min_valid_loss):
                    self.min_valid_loss = valid_loss
                    self.best_epoch = epoch
                    print("Update best model! Best epoch: {}".format(self.best_epoch))
        if not self.config.TEST.USE_LAST_EPOCH:
            print("best trained epoch: {}, min_val_loss: {}".format(
                self.best_epoch, self.min_valid_loss))
        if self.config.TRAIN.PLOT_LOSSES_AND_LR:
            self.plot_losses_and_lrs(mean_training_losses, mean_valid_losses, lrs, self.config)

    def valid(self, data_loader):
        """Runs the model on valid sets. Standard eval mode (no dropout)."""
        if data_loader["valid"] is None:
            raise ValueError("No data for valid")

        print('')
        print(" ====Validing===")
        valid_loss = []
        self.model.eval()
        with torch.no_grad():
            vbar = tqdm(data_loader["valid"], ncols=80)
            for valid_idx, valid_batch in enumerate(vbar):
                vbar.set_description("Validation")
                BVP_label = valid_batch[1].to(torch.float32).to(self.device)
                rPPG, x_visual, x_visual3232, x_visual1616 = self.model(
                    valid_batch[0].to(torch.float32).to(self.device))
                rPPG = (rPPG - torch.mean(rPPG)) / torch.std(rPPG)
                BVP_label = (BVP_label - torch.mean(BVP_label)) / torch.std(BVP_label)
                loss_ecg = self.loss_model(rPPG, BVP_label)
                valid_loss.append(loss_ecg.item())
                vbar.set_postfix(loss=loss_ecg.item())
            valid_loss = np.asarray(valid_loss)
        return np.mean(valid_loss)

    def test(self, data_loader):
        """Runs the model on test sets. Standard eval mode (no dropout, deterministic).

        For MC Dropout uncertainty estimation, use predict_with_uncertainty() instead.
        Standard test() is kept identical to PhysnetTrainer.test() for compatibility
        with the toolbox metrics pipeline.
        """
        if data_loader["test"] is None:
            raise ValueError("No data for test")

        print('')
        print("===Testing===")
        predictions = dict()
        labels = dict()

        if self.config.TOOLBOX_MODE == "only_test":
            if not os.path.exists(self.config.INFERENCE.MODEL_PATH):
                raise ValueError("Inference model path error! Please check INFERENCE.MODEL_PATH in your yaml.")
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
            print("Testing uses pretrained model!")
            print(self.config.INFERENCE.MODEL_PATH)
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_epoch_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                print("Testing uses last epoch as non-pretrained model!")
                print(last_epoch_model_path)
                self.model.load_state_dict(torch.load(last_epoch_model_path))
            else:
                best_model_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                print("Testing uses best epoch selected using model selection as non-pretrained model!")
                print(best_model_path)
                self.model.load_state_dict(torch.load(best_model_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        print("Running model evaluation on the testing dataset!")
        with torch.no_grad():
            for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
                batch_size = test_batch[0].shape[0]
                data, label = test_batch[0].to(
                    self.config.DEVICE), test_batch[1].to(self.config.DEVICE)
                pred_ppg_test, _, _, _ = self.model(data)

                if self.config.TEST.OUTPUT_SAVE_DIR:
                    label = label.cpu()
                    pred_ppg_test = pred_ppg_test.cpu()

                for idx in range(batch_size):
                    subj_index = test_batch[2][idx]
                    sort_index = int(test_batch[3][idx])
                    if subj_index not in predictions.keys():
                        predictions[subj_index] = dict()
                        labels[subj_index] = dict()
                    predictions[subj_index][sort_index] = pred_ppg_test[idx]
                    labels[subj_index][sort_index] = label[idx]

        print('')
        calculate_metrics(predictions, labels, self.config)
        if self.config.TEST.OUTPUT_SAVE_DIR:
            self.save_test_outputs(predictions, labels, self.config)

    def predict_with_uncertainty(self, data_loader, n_samples=20):
        """MC Dropout inference: N stochastic forward passes per batch.

        Returns a dict per subject with mean prediction, per-timestep std,
        and (optionally) raw samples for downstream uncertainty analysis.

        Args:
            data_loader: dict with key "test"
            n_samples: number of stochastic forward passes (default 20, Gal & Ghahramani 2016)

        Returns:
            dict[subj_index][sort_index] = {"mean": tensor, "std": tensor, "label": tensor}
        """
        if data_loader["test"] is None:
            raise ValueError("No data for MC Dropout inference")

        # Load checkpoint (same logic as test())
        if self.config.TOOLBOX_MODE == "only_test":
            self.model.load_state_dict(torch.load(self.config.INFERENCE.MODEL_PATH))
        else:
            if self.config.TEST.USE_LAST_EPOCH:
                last_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.max_epoch_num - 1) + '.pth')
                self.model.load_state_dict(torch.load(last_path))
            else:
                best_path = os.path.join(
                    self.model_dir, self.model_file_name + '_Epoch' + str(self.best_epoch) + '.pth')
                self.model.load_state_dict(torch.load(best_path))

        self.model = self.model.to(self.config.DEVICE)
        self.model.eval()
        enable_dropout(self.model)  # critical: dropout ON for MC sampling

        results = dict()
        print(f"Running MC Dropout inference (n_samples={n_samples}) ...")
        for _, test_batch in enumerate(tqdm(data_loader["test"], ncols=80)):
            batch_size = test_batch[0].shape[0]
            data = test_batch[0].to(torch.float32).to(self.config.DEVICE)
            label = test_batch[1].to(self.config.DEVICE).cpu()

            mean_pred, std_pred, _ = mc_predict(self.model, data, n_samples=n_samples)
            mean_pred = mean_pred.cpu()
            std_pred = std_pred.cpu()

            for idx in range(batch_size):
                subj_index = test_batch[2][idx]
                sort_index = int(test_batch[3][idx])
                if subj_index not in results:
                    results[subj_index] = dict()
                results[subj_index][sort_index] = {
                    "mean": mean_pred[idx],
                    "std": std_pred[idx],
                    "label": label[idx],
                }
        return results

    def save_model(self, index):
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        model_path = os.path.join(
            self.model_dir, self.model_file_name + '_Epoch' + str(index) + '.pth')
        torch.save(self.model.state_dict(), model_path)
        print('Saved Model Path: ', model_path)