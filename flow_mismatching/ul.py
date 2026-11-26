"""Unified Latents (UL, arXiv:2602.17270) noise linking, ELBO bitrate, and loss utilities.

Equations follow Heek et al. (2026), "Unified Latents (UL): How to train your latents":
- Encoder produces a deterministic latent z_clean = E(x), then injects fixed Gaussian noise
  at maximum log-SNR lambda_0 (default lambda_0 = 5 => alpha_0 = sqrt(sigmoid(5)) ~= 0.9967,
  sigma_0 = sqrt(sigmoid(-5)) ~= 0.0817):
      p(z_0 | z_clean) = N(alpha_0 * z_clean, sigma_0^2 * I).
- Variance-preserving (VP) prior ELBO (Eq. 2 / Algorithm 1):
      z_t = alpha_t * z_clean + sigma_t * eps,   alpha_t^2 = sigmoid(lambda_t), sigma_t^2 = sigmoid(-lambda_t)
      KL[p(z_0 | x) || p_theta(z_0)]
          <= E_{t ~ U(0, 1), eps}[ -0.5 * (d lambda / dt) * exp(lambda_t) * ||z_clean - z_hat(z_t, t)||^2 ]
             + KL[p(z_1 | x) || N(0, I)].
- Equivalent Flow-Matching affine-path formulation (z_s = (1 - s) * eps + s * z_clean, s in [0, s_1]):
  Setting s_1 = alpha_0 / (alpha_0 + sigma_0) = sigmoid(lambda_0 / 2) matches the exact SNR
  lambda_fm(s) = 2 * log(s / (1 - s)) at s = s_1, with z_{s=0} = eps ~ N(0, I) (zero boundary KL!).
  With velocity v = z_clean - eps and z_hat = z_s + (1 - s) * v_theta(z_s, s), the exact ELBO simplifies to:
      KL[p(z_{s_1} | x) || p_theta(z_{s_1})]
          <= s_1 * E_{s ~ U(0, s_1), eps}[ (s / (1 - s)) * ||v_theta(z_s, s) - (z_clean - eps)||^2 ].
- Decoder reweighted ELBO (Section 3.2 & Fig. 3):
  On epsilon-MSE, w_eps(lambda) = c_lf * sigmoid(b - lambda), discounting low-noise high-frequency detail
  so the decoder models it while c_lf > 1 (typically 1.3 - 1.7) prevents posterior collapse.
"""

import math
from typing import Tuple

import torch

LN2 = math.log(2.0)


def sigma0_to_logsnr(sigma0: float) -> float:
    """Convert VP base noise std sigma_0 in (0, 1) to maximum log-SNR lambda_0 = log((1 - sigma_0^2) / sigma_0^2)."""
    if not 0.0 < sigma0 < 1.0:
        raise ValueError(f"sigma0 must lie strictly in (0, 1), got {sigma0}")
    s2 = sigma0 * sigma0
    return math.log((1.0 - s2) / s2)


def logsnr_to_sigma0(logsnr0: float) -> float:
    """Convert maximum log-SNR lambda_0 to VP base noise std sigma_0 = sqrt(sigmoid(-lambda_0))."""
    return math.sqrt(1.0 / (1.0 + math.exp(logsnr0)))


def sigma0_to_alpha_sigma(sigma0: float) -> Tuple[float, float]:
    """Return (alpha_0, sigma_0) on the variance-preserving circle alpha_0^2 + sigma_0^2 = 1."""
    if not 0.0 < sigma0 < 1.0:
        raise ValueError(f"sigma0 must lie strictly in (0, 1), got {sigma0}")
    return math.sqrt(1.0 - sigma0 * sigma0), float(sigma0)


def sigma0_to_fm_t1(sigma0: float) -> float:
    """Map UL base noise std sigma_0 to the Flow-Matching affine-path endpoint t_1 in (0, 1).

    On the affine path z_t = (1 - t) * eps + t * z_clean, SNR(t) = (t / (1 - t))^2.
    Equating SNR(t_1) = alpha_0^2 / sigma_0^2 yields t_1 = alpha_0 / (alpha_0 + sigma_0).
    """
    alpha0, s0 = sigma0_to_alpha_sigma(sigma0)
    return alpha0 / (alpha0 + s0)


def fm_t_to_logsnr(t: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Log-SNR along the affine path z_t = (1 - t) * eps + t * x: lambda(t) = 2 * log(t / (1 - t))."""
    t_clamped = t.clamp(eps, 1.0 - eps)
    return 2.0 * torch.log(t_clamped / (1.0 - t_clamped))


def vp_alpha_sigma_from_logsnr(logsnr: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return (alpha, sigma) with alpha^2 = sigmoid(logsnr), sigma^2 = sigmoid(-logsnr)."""
    alpha = torch.sqrt(torch.sigmoid(logsnr))
    sigma = torch.sqrt(torch.sigmoid(-logsnr))
    return alpha, sigma


def encode_with_fixed_noise(
    z_clean: torch.Tensor,
    sigma0: float,
    noise: torch.Tensor | None = None,
) -> torch.Tensor:
    """Sample UL noisy latent z_0 = alpha_0 * z_clean + sigma_0 * eps."""
    alpha0, s0 = sigma0_to_alpha_sigma(sigma0)
    if noise is None:
        noise = torch.randn_like(z_clean)
    return alpha0 * z_clean + s0 * noise


def vp_prior_boundary_kl(z_clean: torch.Tensor, logsnr_min: float) -> torch.Tensor:
    """Exact per-sample KL[N(alpha_1 * z_clean, sigma_1^2 I) || N(0, I)] in nats (summed over non-batch dims)."""
    logsnr_t = torch.tensor(logsnr_min, device=z_clean.device, dtype=torch.float32)
    alpha1_sq = torch.sigmoid(logsnr_t)
    sigma1_sq = torch.sigmoid(-logsnr_t)
    # KL(N(mu, sigma^2) || N(0, 1)) = 0.5 * (-log(sigma^2) + sigma^2 + mu^2 - 1)
    # Note: -log(sigma1^2) = softplus(logsnr_min) for numerical stability.
    neg_log_sigma1_sq = torch.nn.functional.softplus(logsnr_t)
    kl_elem = 0.5 * (neg_log_sigma1_sq + sigma1_sq + alpha1_sq * z_clean.float().pow(2) - 1.0)
    return kl_elem.flatten(1).sum(dim=1)


def vp_prior_elbo_nats(
    z_clean: torch.Tensor,
    z_hat: torch.Tensor,
    logsnr: torch.Tensor,
    neg_dlogsnr_dt: torch.Tensor | float,
    logsnr_min: float,
    include_boundary_kl: bool = True,
) -> torch.Tensor:
    """Per-sample UL diffusion prior ELBO upper bound on KL[p(z_0 | x) || p_theta(z_0)] in nats.

    Args:
        z_clean: clean encoder output [B, ...].
        z_hat: prior's prediction of z_clean from z_t [B, ...].
        logsnr: log-SNR lambda(t) of shape [B].
        neg_dlogsnr_dt: positive rate -d(lambda)/dt at t (scalar or shape [B]).
        logsnr_min: minimum log-SNR lambda(1) at the noisy end of the schedule.
        include_boundary_kl: whether to add KL[p(z_1 | x) || N(0, I)].

    Returns:
        Tensor of shape [B] containing the per-sample ELBO estimate in nats (summed over latent dims).
    """
    sq_err = (z_clean.float() - z_hat.float()).pow(2).flatten(1).sum(dim=1)
    weight = 0.5 * torch.as_tensor(neg_dlogsnr_dt, device=z_clean.device, dtype=torch.float32) * torch.exp(
        logsnr.float()
    )
    loss = weight * sq_err
    if include_boundary_kl:
        loss = loss + vp_prior_boundary_kl(z_clean, logsnr_min)
    return loss


def fm_prior_elbo_nats(
    v_pred: torch.Tensor,
    v_target: torch.Tensor,
    t: torch.Tensor,
    t1: float,
) -> torch.Tensor:
    """Per-sample UL prior ELBO upper bound in nats for a Flow-Matching velocity network on t ~ U(0, t1).

    Along z_t = (1 - t) * eps + t * z_clean with target v = z_clean - eps and t1 = alpha_0 / (alpha_0 + sigma_0):
        KL[p(z_{t1} | x) || p_theta(z_{t1})] <= t1 * E_{t ~ U(0, t1)}[ (t / (1 - t)) * ||v_pred - v_target||^2 ].
    Note that the boundary KL at t = 0 is identically zero because z_0 = eps ~ N(0, I).

    Args:
        v_pred: predicted velocity v_theta(z_t, t) of shape [B, ...].
        v_target: target velocity z_clean - eps of shape [B, ...].
        t: sampled timesteps in (0, t1), shape [B].
        t1: upper integration endpoint t_1 = sigma0_to_fm_t1(sigma0).

    Returns:
        Tensor of shape [B] in nats (summed over all latent dimensions).
    """
    if not 0.0 < t1 < 1.0:
        raise ValueError(f"t1 must lie strictly in (0, 1), got {t1}")
    sq_err = (v_pred.float() - v_target.float()).pow(2).flatten(1).sum(dim=1)
    t_f = t.float()
    weight = t1 * (t_f / (1.0 - t_f))
    return weight * sq_err


def nats_to_bpd(nats: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Convert total nats to bits per dimension (bpd)."""
    return nats / (LN2 * float(num_dims))


def decoder_sigmoid_weight_eps(
    logsnr: torch.Tensor,
    bias: float = 0.0,
    loss_factor: float = 1.5,
) -> torch.Tensor:
    """UL decoder reweighting on epsilon-MSE (arXiv:2602.17270 Fig. 3): c_lf * sigmoid(b - lambda)."""
    return float(loss_factor) * torch.sigmoid(float(bias) - logsnr.float())


def decoder_sigmoid_weight_fm_velocity(
    t: torch.Tensor,
    bias: float = 0.0,
    loss_factor: float = 1.5,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Equivalent UL sigmoid decoder weighting when training a Flow-Matching velocity decoder on x_t = (1-t)eps + t x.

    Combines the exact velocity-ELBO density (t / (1 - t)) with UL's sigmoid weighting c_lf * sigmoid(b - lambda(t)),
    where lambda(t) = 2 * log(t / (1 - t)). Because lambda(t) -> +inf as t -> 1, the product stays bounded as t -> 1
    (decays as (1 - t) / t), naturally discounting high-frequency noise levels.
    """
    t_c = t.float().clamp(eps, 1.0 - eps)
    logsnr = fm_t_to_logsnr(t_c, eps=eps)
    return (t_c / (1.0 - t_c)) * decoder_sigmoid_weight_eps(logsnr, bias=bias, loss_factor=loss_factor)


class LinearLogSNRSchedule:
    """Linear log-SNR VP schedule: lambda(t) = lambda_max + (lambda_min - lambda_max) * t, t in [0, 1]."""

    def __init__(self, sigma0: float = 0.08, logsnr_min: float = -10.0):
        self.sigma0 = float(sigma0)
        self.logsnr_max = sigma0_to_logsnr(sigma0)
        self.logsnr_min = float(logsnr_min)
        if self.logsnr_min >= self.logsnr_max:
            raise ValueError(f"logsnr_min ({logsnr_min}) must be < logsnr_max ({self.logsnr_max:.3f})")

    @property
    def neg_dlogsnr_dt(self) -> float:
        return self.logsnr_max - self.logsnr_min

    def logsnr(self, t: torch.Tensor) -> torch.Tensor:
        return self.logsnr_max + (self.logsnr_min - self.logsnr_max) * t.float()

    def alpha_sigma(self, t: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return vp_alpha_sigma_from_logsnr(self.logsnr(t))

    def forward_noise(
        self,
        z_clean: torch.Tensor,
        t: torch.Tensor,
        noise: torch.Tensor | None = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (z_t, eps, logsnr_t) for t in [0, 1]."""
        if noise is None:
            noise = torch.randn_like(z_clean)
        logsnr_t = self.logsnr(t)
        alpha, sigma = vp_alpha_sigma_from_logsnr(logsnr_t)
        view_shape = (-1,) + (1,) * (z_clean.ndim - 1)
        z_t = alpha.view(view_shape).to(z_clean.dtype) * z_clean + sigma.view(view_shape).to(z_clean.dtype) * noise
        return z_t, noise, logsnr_t


@torch.inference_mode()
def estimate_ul_bitrate_nats_per_dim(
    prior: torch.nn.Module,
    z_clean: torch.Tensor,
    sigma0: float,
    num_paths: int = 10,
    num_steps: int = 10,
    class_idx: int | None = None,
    amp_dtype: torch.dtype | None = None,
    generator: torch.Generator | None = None,
) -> float:
    """Estimate UL prior ELBO bitrate (nats per latent dimension) for a single latent [1, C, h, w].

    Uses midpoint quadrature over T steps in (0, t_1) and K noise draws:
        nats/dim = (t_1 / (T * D_z)) * sum_t (t / (1 - t)) * mean_k ||v_theta(z_t^k, t) - (z_clean - eps^k)||^2.
    Divide by math.log(2) to obtain bits per latent dimension (bpd).
    """
    if z_clean.ndim != 4 or z_clean.shape[0] != 1:
        raise ValueError(f"Expected z_clean of shape [1, C, h, w], got {tuple(z_clean.shape)}")
    t1 = sigma0_to_fm_t1(sigma0)
    device = z_clean.device
    _, c, h, w = z_clean.shape
    num_dims = float(c * h * w)
    use_amp = amp_dtype is not None and device.type == "cuda"

    # Midpoint grid strictly inside (0, t1) for unbiased Riemann integration.
    dt = t1 / float(num_steps)
    times = (torch.arange(num_steps, device=device, dtype=torch.float32) + 0.5) * dt

    eps = torch.randn((num_paths, c, h, w), device=device, dtype=z_clean.dtype, generator=generator)
    z1 = z_clean.expand(num_paths, -1, -1, -1)
    v_target = (z1 - eps).float()

    total_nats = 0.0
    for t in times:
        zt = (1.0 - t) * eps + t * z1
        t_batch = torch.full((num_paths,), float(t), device=device, dtype=torch.float32)
        extra = {}
        if class_idx is not None:
            extra["label"] = torch.full((num_paths,), int(class_idx), device=device, dtype=torch.long)
        with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
            v_pred = prior(zt, t_batch, extra=extra)
        mse_sum = (v_pred.float() - v_target).pow(2).flatten(1).sum(dim=1).mean()
        total_nats += float(t1 * (t / (1.0 - t)) * mse_sum) / float(num_steps)
    return total_nats / num_dims

