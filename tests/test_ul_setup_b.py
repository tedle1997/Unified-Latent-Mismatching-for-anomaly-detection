"""Unit tests for Phase 2 (Setup B): UL encoder, diffusion decoder, joint step, and decoder-free loading."""

from pathlib import Path

import pytest
import torch

from flow_mismatching.ul import estimate_ul_bitrate_nats_per_dim, sigma0_to_alpha_sigma, sigma0_to_fm_t1
from flow_mismatching.ul_models import ULDiffusionDecoder, ULEncoder
from flow_mismatching.unet import UNetModel
from flow_mismatching.utils import get_model_config, load_ul_components
from train import _ULSystem


@pytest.mark.parametrize("downsample", [4, 8])
def test_ul_encoder_decoder_shapes_and_gradients(downsample: int):
    torch.manual_seed(0)
    c_lat = 8
    img_sz = 64
    lat_sz = img_sz // downsample
    sigma0 = 0.08
    t1 = sigma0_to_fm_t1(sigma0)

    enc = ULEncoder(in_channels=3, latent_channels=c_lat, base_channels=32, downsample=downsample, num_res_blocks=1)
    prior = UNetModel(
        in_channels=c_lat,
        out_channels=c_lat,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_head_channels=16,
        num_classes=None,
    )
    dec = ULDiffusionDecoder(
        img_channels=3,
        latent_channels=c_lat,
        downsample=downsample,
        cond_channels=8,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        num_head_channels=16,
        num_classes=None,
    )
    system = _ULSystem(enc, prior, dec)

    x = torch.randn(2, 3, img_sz, img_sz)
    t_z = torch.tensor([0.2, 0.7])
    t_x = torch.tensor([0.3, 0.8])
    eps_z = torch.randn(2, c_lat, lat_sz, lat_sz)
    eps_z0 = torch.randn(2, c_lat, lat_sz, lat_sz)
    eps_x = torch.randn_like(x)

    loss, l_prior, l_dec, bpd, z_std = system(
        x, x, t_z, eps_z, eps_z0, t_x, eps_x, sigma0=sigma0, t1=t1, decoder_bias=4.0, loss_factor=2.5, extra={}
    )
    assert torch.isfinite(loss) and loss.item() > 0 and z_std.item() > 0
    assert torch.isfinite(l_prior) and torch.isfinite(l_dec) and bpd.item() > 0

    loss.backward()
    # Gradients must reach encoder, prior, and decoder simultaneously.
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in enc.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prior.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in dec.parameters())

    # Decoder sampling check.
    z_clean = enc(x).detach()
    rec = dec.sample(z_clean, img_shape=(img_sz, img_sz), steps=2)
    assert rec.shape == x.shape and torch.isfinite(rec).all()


def test_ul_checkpoint_roundtrip_drops_decoder_by_default(tmp_path: Path):
    enc = ULEncoder(in_channels=3, latent_channels=8, base_channels=32, downsample=4, num_res_blocks=1)
    prior = UNetModel(
        in_channels=8,
        out_channels=8,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_head_channels=16,
        num_classes=None,
    )
    dec = ULDiffusionDecoder(
        img_channels=3,
        latent_channels=8,
        downsample=4,
        cond_channels=8,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        num_head_channels=16,
        num_classes=None,
    )
    alpha0, s0 = sigma0_to_alpha_sigma(0.08)
    ckpt = tmp_path / "ul_ckpt.pt"
    torch.save(
        {
            "space": "ul",
            "model": prior.state_dict(),
            "encoder": enc.state_dict(),
            "decoder": dec.state_dict(),
            "model_config": get_model_config(prior),
            "encoder_config": enc.config_dict(),
            "decoder_config": dec.config_dict(),
            "ul_config": {"sigma0": s0, "alpha0": alpha0, "t1": sigma0_to_fm_t1(s0)},
            "epoch": 5,
        },
        ckpt,
    )

    p_loaded, e_loaded, d_loaded, meta = load_ul_components(str(ckpt), torch.device("cpu"), load_decoder=False)
    assert d_loaded is None
    assert meta["space"] == "ul" and meta["epoch"] == 5

    # Bitrate diagnostic runs cleanly with decoder discarded.
    z = e_loaded(torch.randn(1, 3, 64, 64))
    nats_d = estimate_ul_bitrate_nats_per_dim(p_loaded, z, sigma0=0.08, num_paths=2, num_steps=3)
    assert nats_d > 0 and torch.isfinite(torch.tensor(nats_d))

    # Optional diagnostic reload includes decoder.
    _, _, d_diag, _ = load_ul_components(str(ckpt), torch.device("cpu"), load_decoder=True)
    assert d_diag is not None
