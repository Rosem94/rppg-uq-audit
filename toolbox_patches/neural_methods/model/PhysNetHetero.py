""" PhysNetHetero

Baseline PhysNet (Yu et al., BMVC 2019) extended with:
  1. Dropout3d layers in the deep feature extractor (epistemic / MC dropout),
     matching the PhysNet_MCDropout convention used elsewhere in this repo.
  2. A second output head (ConvBlock10_var) that predicts a per-timestep
     log-variance s_t = log sigma^2_aleatoric,t (aleatoric / data uncertainty),
     trained jointly via HeteroscedasticLoss (Kendall & Gal 2017; Song 2023).

forward() returns (rPPG, log_var, x_visual, x_visual3232, x_visual1616) so the
first two outputs feed the heteroscedastic loss while the visual maps preserve
the baseline trainer signature.

Dropout placement note: Dropout3d(p) is applied after the four 64-channel deep
blocks (ConvBlock6-9). If your PhysNet_MCDropout uses a different placement,
align it for an exact epistemic match; the aleatoric/total-uncertainty
conclusion does not depend on the exact placement.

Only for research purpose. MIT License.
"""

import torch
import torch.nn as nn


class PhysNetHetero(nn.Module):
    def __init__(self, frames=128, dropout_rate=0.1):
        super(PhysNetHetero, self).__init__()

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[
                4, 1, 1], stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # Mean (BVP) head -- identical to baseline ConvBlock10.
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)
        # Aleatoric log-variance head -- parallel 64->1 conv.
        self.ConvBlock10_var = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

        # MC-dropout layers (kept active at inference via enable_dropout()).
        self.dropout = nn.Dropout3d(p=dropout_rate)

    def forward(self, x):  # Batch_size*[3, T, 128,128]
        x_visual = x
        [batch, channel, length, width, height] = x.shape

        x = self.ConvBlock1(x)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock2(x)
        x_visual6464 = self.ConvBlock3(x)
        x = self.MaxpoolSpaTem(x_visual6464)

        x = self.ConvBlock4(x)
        x_visual3232 = self.ConvBlock5(x)
        x = self.MaxpoolSpaTem(x_visual3232)

        x = self.ConvBlock6(x)
        x = self.dropout(x)
        x_visual1616 = self.ConvBlock7(x)
        x = self.dropout(x_visual1616)
        x = self.MaxpoolSpa(x)

        x = self.ConvBlock8(x)
        x = self.dropout(x)
        x = self.ConvBlock9(x)
        x = self.dropout(x)

        x = self.upsample(x)
        x = self.upsample2(x)

        x = self.poolspa(x)               # [B, 64, T, 1, 1]
        rPPG = self.ConvBlock10(x).view(-1, length)        # [B, T]
        log_var = self.ConvBlock10_var(x).view(-1, length)  # [B, T]

        return rPPG, log_var, x_visual, x_visual3232, x_visual1616


def enable_dropout(model):
    """Set all dropout layers to train mode while the rest stays in eval mode."""
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()


@torch.no_grad()
def mc_predict_hetero(model, x, n_samples=20):
    """N stochastic forward passes with dropout active.

    Returns:
        mean_pred:   (B, T)  mean BVP prediction across passes
        std_pred:    (B, T)  per-timestep epistemic std across passes (sigma_MCD)
        samples:     (N, B, T) raw per-pass BVP (for trial-consistent HR sigma)
        logvar_mean: (B, T)  mean aleatoric log-variance across passes
    """
    model.eval()
    enable_dropout(model)
    rppg_list, logvar_list = [], []
    for _ in range(n_samples):
        rppg, log_var, _, _, _ = model(x)
        rppg_list.append(rppg.unsqueeze(0))
        logvar_list.append(log_var.unsqueeze(0))
    rppg_stack = torch.cat(rppg_list, dim=0)        # (N, B, T)
    logvar_stack = torch.cat(logvar_list, dim=0)    # (N, B, T)
    mean_pred = rppg_stack.mean(dim=0)
    std_pred = rppg_stack.std(dim=0, unbiased=True)
    logvar_mean = logvar_stack.mean(dim=0)
    return mean_pred, std_pred, rppg_stack, logvar_mean
