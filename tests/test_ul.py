"""Unit tests for Unified Latents (UL, arXiv:2602.17270) ELBO/bitrate equations and latent pipeline."""

import math

import pytest
import torch

from flow_mismatching.scoring import anomaly_heatmap, gaussian_smooth, image_score
from flow_mismatching.ul import (
    LN2,
    LinearLogSNRSchedule,
    decoder_sigmoid_weight_eps,
    decoder_sigmoid_weight_fm_velocity,
    encode_with_fixed_noise,
    fm_prior_elbo_nats,
    fm_t_to_logsnr,
    logsnr_to_sigma0,
    nats_to_bpd,
    sigma0_to_alpha_sigma,
    sigma0_to_fm_t1,
    sigma0_to_logsnr,
    vp_prior_elbo_nats,
)
from flow_mismatching.unet import UNetModel


def test_sigma0_logsnr_t1_conversions():
    # Paper default: lambda(0) = 5 => alpha_0 ~= 0.99665, sigma_0 ~= 0.08172
    s0 = logsnr_to_sigma0(5.0)
    assert math.isclose(s0, math.sqrt(1.0 / (1.0 + math.exp(5.0))), rel_tol=1e-7)
    assert abs(s0 - 0.08172) < 1e-4
    assert math.isclose(sigma0_to_logsnr(s0), 5.0, rel_tol=1e-6)

    for sigma0 in (0.04, 0.08, 0.16):
        alpha0, s = sigma0_to_alpha_sigma(sigma0)
        assert math.isclose(alpha0**2 + s**2, 1.0, rel_tol=1e-7)
        t1 = sigma0_to_fm_t1(sigma0)
        # SNR along affine path at t1 must equal alpha_0^2 / sigma_0^2
        logsnr_fm = float(fm_t_to_logsnr(torch.tensor(t1)))
        assert math.isclose(logsnr_fm, sigma0_to_logsnr(sigma0), rel_tol=1e-5)


@pytest.mark.parametrize("sigma0", [0.04, 0.08, 0.16])
@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_vp_prior_elbo_matches_analytical_gaussian_kl(sigma0: float, tau: float):
    """For z_clean ~ N(0, tau^2 I_d), Bayes-optimal denoiser achieves exact analytical KL = 0.5*d*log(1 + tau^2 * exp(lambda_0))."""
    torch.manual_seed(0)
    d = 8
    n = 65536
    schedule = LinearLogSNRSchedule(sigma0=sigma0, logsnr_min=-12.0)

    z_clean = tau * torch.randn(n, d)
    t = torch.rand(n)
    z_t, _, logsnr_t = schedule.forward_noise(z_clean, t)
    alpha_t, sigma_t = schedule.alpha_sigma(t)

    # Bayes-optimal posterior mean E[z_clean | z_t] for z_clean ~ N(0, tau^2 I):
    # z_hat* = (alpha_t * tau^2 / (alpha_t^2 * tau^2 + sigma_t^2)) * z_t
    denom = alpha_t.pow(2) * (tau**2) + sigma_t.pow(2)
    z_hat_opt = (alpha_t * (tau**2) / denom).unsqueeze(1) * z_t

    elbo_nats = vp_prior_elbo_nats(
        z_clean=z_clean,
        z_hat=z_hat_opt,
        logsnr=logsnr_t,
        neg_dlogsnr_dt=schedule.neg_dlogsnr_dt,
        logsnr_min=schedule.logsnr_min,
        include_boundary_kl=True,
    )
    est_nats_per_dim = float(elbo_nats.mean()) / d
    exact_nats_per_dim = 0.5 * math.log(1.0 + (tau**2) * math.exp(schedule.logsnr_max))

    # Tight upper bound (exact for tau=1, tiny O(exp(logsnr_min)) gap for tau!=1).
    assert abs(est_nats_per_dim - exact_nats_per_dim) < 0.03
    assert math.isclose(
        float(nats_to_bpd(elbo_nats.mean(), d)),
        est_nats_per_dim / LN2,
        rel_tol=1e-6,
    )


@pytest.mark.parametrize("sigma0", [0.04, 0.08, 0.16])
@pytest.mark.parametrize("tau", [0.5, 1.0, 2.0])
def test_fm_prior_elbo_matches_analytical_gaussian_kl(sigma0: float, tau: float):
    """Exact Flow-Matching affine-path UL ELBO matches analytical Gaussian KL for all tau with zero boundary KL."""
    torch.manual_seed(42)
    d = 8
    n = 65536
    t1 = sigma0_to_fm_t1(sigma0)
    logsnr0 = sigma0_to_logsnr(sigma0)

    z_clean = tau * torch.randn(n, d)
    eps = torch.randn(n, d)
    t = torch.rand(n) * t1
    t_col = t.unsqueeze(1)
    z_t = (1.0 - t_col) * eps + t_col * z_clean
    v_target = z_clean - eps

    # Bayes-optimal velocity E[z_clean - eps | z_t] for z_clean ~ N(0, tau^2 I), eps ~ N(0, I):
    # Cov(z_clean - eps, z_t) = t * tau^2 - (1 - t)
    # Var(z_t) = t^2 * tau^2 + (1 - t)^2
    v_opt = ((t_col * (tau**2) - (1.0 - t_col)) / (t_col.pow(2) * (tau**2) + (1.0 - t_col).pow(2))) * z_t

    elbo_nats = fm_prior_elbo_nats(v_pred=v_opt, v_target=v_target, t=t, t1=t1)
    est_nats_per_dim = float(elbo_nats.mean()) / d
    exact_nats_per_dim = 0.5 * math.log(1.0 + (tau**2) * math.exp(logsnr0))

    assert abs(est_nats_per_dim - exact_nats_per_dim) < 0.03


def test_bitrate_monotonicity_in_sigma0():
    """Smaller sigma_0 (less encoder noise) must correspond to strictly higher Gaussian bitrate."""
    rates = [0.5 * math.log(1.0 + math.exp(sigma0_to_logsnr(s))) for s in (0.16, 0.08, 0.04)]
    assert rates[0] < rates[1] < rates[2]


def test_decoder_sigmoid_weights():
    logsnrs = torch.tensor([-6.0, 0.0, 6.0])
    w_eps = decoder_sigmoid_weight_eps(logsnrs, bias=0.0, loss_factor=1.5)
    # High noise (lambda=-6) approaches c_lf=1.5; low noise (lambda=+6) is discounted toward 0.
    assert w_eps[0] > w_eps[1] > w_eps[2]
    assert abs(float(w_eps[0]) - 1.5) < 0.01
    assert float(w_eps[2]) < 0.01

    # Flow-matching decoder weight stays finite and positive across (0, 1).
    t = torch.linspace(0.01, 0.99, 50)
    w_fm = decoder_sigmoid_weight_fm_velocity(t, bias=0.0, loss_factor=1.5)
    assert torch.isfinite(w_fm).all() and (w_fm > 0).all()


def test_encode_with_fixed_noise_statistics():
    torch.manual_seed(0)
    z = torch.zeros(4096, 4, 8, 8)
    z_noisy = encode_with_fixed_noise(z, sigma0=0.08)
    assert abs(float(z_noisy.std()) - 0.08) < 1e-3


def test_latent_unet_and_scoring_upsample():
    model = UNetModel(
        in_channels=4,
        out_channels=4,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2, 4),
        attention_resolutions=(2, 4),
        num_head_channels=16,
        num_classes=None,
    )
    z = torch.randn(1, 4, 32, 32)
    hmap = anomaly_heatmap(
        model,
        z,
        num_paths=2,
        num_steps=3,
        aggregation="min",
        t_range=(0.0, sigma0_to_fm_t1(0.08)),
        output_size=(128, 128),
    )
    assert hmap.shape == (128, 128)
    smooth = gaussian_smooth(hmap, ksize=9, sigma=3.0)
    assert smooth.shape == (128, 128)
    score = image_score(smooth)
    assert math.isfinite(score) and score >= 0.0
