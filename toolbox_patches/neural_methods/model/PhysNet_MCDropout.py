""" PhysNet with MC Dropout for uncertainty quantification.

Modification of PhysNet_padding_Encoder_Decoder_MAX (Yu et al. 2019, BMVC) that adds
spatial dropout (nn.Dropout3d) after each ConvBlock activation in the feature extractor.

Used for Monte Carlo Dropout-based uncertainty estimation
(Gal & Ghahramani 2016, "Dropout as a Bayesian Approximation", ICML).

Architecture is identical to baseline PhysNet except for 9 added Dropout3d layers,
one after each ConvBlock1-9 ReLU. ConvBlock10 (regression head), upsample blocks,
and pooling layers are left unchanged.

Use enable_dropout(model) at inference time to keep dropout active despite model.eval().
"""

import torch
import torch.nn as nn


class PhysNet_MCDropout(nn.Module):
    """PhysNet with spatial dropout in the feature extractor for MC Dropout."""

    def __init__(self, frames=128, dropout_rate=0.1):
        super().__init__()
        self.dropout_rate = dropout_rate

        self.ConvBlock1 = nn.Sequential(
            nn.Conv3d(3, 16, [1, 5, 5], stride=1, padding=[0, 2, 2]),
            nn.BatchNorm3d(16),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

        self.ConvBlock2 = nn.Sequential(
            nn.Conv3d(16, 32, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock3 = nn.Sequential(
            nn.Conv3d(32, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

        self.ConvBlock4 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock5 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock6 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock7 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock8 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )
        self.ConvBlock9 = nn.Sequential(
            nn.Conv3d(64, 64, [3, 3, 3], stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.Dropout3d(p=dropout_rate),
        )

        self.upsample = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1],
                               stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )
        self.upsample2 = nn.Sequential(
            nn.ConvTranspose3d(in_channels=64, out_channels=64, kernel_size=[4, 1, 1],
                               stride=[2, 1, 1], padding=[1, 0, 0]),
            nn.BatchNorm3d(64),
            nn.ELU(),
        )

        # Regression head: NO dropout here (Gal & Ghahramani 2016)
        self.ConvBlock10 = nn.Conv3d(64, 1, [1, 1, 1], stride=1, padding=0)

        self.MaxpoolSpa = nn.MaxPool3d((1, 2, 2), stride=(1, 2, 2))
        self.MaxpoolSpaTem = nn.MaxPool3d((2, 2, 2), stride=2)
        self.poolspa = nn.AdaptiveAvgPool3d((frames, 1, 1))

    def forward(self, x):  # x: Batch_size * [3, T, H, W]
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
        x_visual1616 = self.ConvBlock7(x)
        x = self.MaxpoolSpa(x_visual1616)

        x = self.ConvBlock8(x)
        x = self.ConvBlock9(x)
        x = self.upsample(x)
        x = self.upsample2(x)

        x = self.poolspa(x)
        x = self.ConvBlock10(x)

        rPPG = x.view(-1, length)
        return rPPG, x_visual, x_visual3232, x_visual1616


def enable_dropout(model: nn.Module) -> None:
    """Enable Dropout layers during inference for MC Dropout sampling.

    PyTorch's `model.eval()` disables both BatchNorm updates and Dropout.
    For MC Dropout we need BatchNorm in eval-mode but Dropout in train-mode.
    This walks the module tree and reactivates Dropout/Dropout3d only.
    """
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout2d, nn.Dropout3d)):
            module.train()


@torch.no_grad()
def mc_predict(model: nn.Module, x: torch.Tensor, n_samples: int = 20):
    """Run N stochastic forward passes for MC Dropout uncertainty estimation.

    Args:
        model: A model in eval mode with dropout reactivated via enable_dropout().
        x: Input tensor (B, C, T, H, W).
        n_samples: Number of forward passes (Gal recommends >= 10; 20 is standard).

    Returns:
        mean: (B, T) tensor of mean rPPG predictions.
        std:  (B, T) tensor of per-timestep predictive std (epistemic uncertainty).
        all_predictions: (n_samples, B, T) raw predictions for downstream analysis.
    """
    predictions = []
    for _ in range(n_samples):
        rppg, *_ = model(x)
        predictions.append(rppg)
    stacked = torch.stack(predictions, dim=0)  # (N, B, T)
    return stacked.mean(dim=0), stacked.std(dim=0), stacked