"""Heteroscedastic loss for PhysNet (Kendall & Gal 2017; Song et al. 2023).

Combines the standard Neg-Pearson waveform-shape term with a Gaussian
negative-log-likelihood (loss-attenuation) term that uses a learned
per-timestep log-variance head. The log-variance is the aleatoric
(data-noise) uncertainty; the NLL term down-weights high-noise samples
exactly as in Kendall & Gal, so the aleatoric estimate acts as a
regulariser on the regression --- this follows Song et al.'s formulation
where data uncertainty and the regression objective share one loss.

    L = NegPearson(b_hat, b)
        + beta * mean_t[ 0.5 * exp(-s_t) * (b_hat_t - b_t)^2 + 0.5 * s_t ]

where s_t = log sigma^2_aleatoric,t is the model's second output head.
All tensors are in the z-normalised signal space used by the trainer.

forward() returns (total_loss, shape_term, nll_term); the trainer should
backprop total_loss and may log the two components separately.
"""
import torch
import torch.nn as nn

from neural_methods.loss.PhysNetNegPearsonLoss import Neg_Pearson


class HeteroscedasticLoss(nn.Module):
    def __init__(self, beta=1.0, logvar_min=-7.0, logvar_max=7.0):
        super().__init__()
        self.beta = beta
        self.logvar_min = logvar_min
        self.logvar_max = logvar_max
        self.neg_pearson = Neg_Pearson()

    def forward(self, rPPG, log_var, BVP_label):
        # rPPG, log_var, BVP_label: (B, T), all in normalised signal space.
        shape_loss = self.neg_pearson(rPPG, BVP_label)

        # Clamp log-variance for numerical stability (precision = exp(-s)).
        log_var = torch.clamp(log_var, min=self.logvar_min, max=self.logvar_max)
        precision = torch.exp(-log_var)
        nll = 0.5 * precision * (rPPG - BVP_label) ** 2 + 0.5 * log_var
        nll = nll.mean()

        total = shape_loss + self.beta * nll
        return total, shape_loss.detach(), nll.detach()
