"""CPU unit tests for Phase 3 (Setup C: Frozen DINOv2 + UL Adapter + MVTec-LOCO masks)."""

import json
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from flow_mismatching.data import NORM_STATS, PairedLatentImageDataset, TestDataset, find_mask
from flow_mismatching.encoders import Standardizer
from flow_mismatching.scoring import anomaly_heatmap
from flow_mismatching.ul import estimate_ul_bitrate_nats_per_dim, sigma0_to_alpha_sigma, sigma0_to_fm_t1
from flow_mismatching.ul_models import DINOULAdapter, ULDiffusionDecoder
from flow_mismatching.unet import UNetModel
from flow_mismatching.utils import get_model_config, load_ul_components
from train import _ULSystem


@pytest.mark.parametrize("adapter_type", ["mlp", "conv"])
@pytest.mark.parametrize("c_lat", [16, 32])
def test_dino_ul_adapter_joint_step_and_gradients(adapter_type: str, c_lat: int):
    torch.manual_seed(0)
    feat_dim, lat_hw, img_sz = 384, 16, 128
    sigma0 = 0.08
    t1 = sigma0_to_fm_t1(sigma0)

    adapter = DINOULAdapter(
        in_channels=feat_dim, latent_channels=c_lat, hidden_channels=64, adapter_type=adapter_type
    )
    prior = UNetModel(
        in_channels=c_lat,
        out_channels=c_lat,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(1, 2),
        num_head_channels=16,
        num_classes=None,
    )
    decoder = ULDiffusionDecoder(
        img_channels=3,
        latent_channels=c_lat,
        downsample=img_sz // lat_hw,  # 8x upsampling: 16x16 -> 128x128
        cond_channels=8,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        num_head_channels=16,
        num_classes=None,
    )
    system = _ULSystem(adapter, prior, decoder)

    feat = torch.randn(2, feat_dim, lat_hw, lat_hw)
    x_img = torch.randn(2, 3, img_sz, img_sz)
    t_z = torch.tensor([0.2, 0.7])
    t_x = torch.tensor([0.3, 0.8])
    eps_z = torch.randn(2, c_lat, lat_hw, lat_hw)
    eps_z0 = torch.randn(2, c_lat, lat_hw, lat_hw)
    eps_x = torch.randn_like(x_img)

    loss, l_prior, l_dec, bpd, z_std = system(
        feat, x_img, t_z, eps_z, eps_z0, t_x, eps_x, sigma0=sigma0, t1=t1, decoder_bias=4.0, loss_factor=2.5, extra={}
    )
    assert torch.isfinite(loss) and loss.item() > 0 and z_std.item() > 0
    assert torch.isfinite(l_prior) and torch.isfinite(l_dec) and bpd.item() > 0

    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in prior.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in decoder.parameters())


def test_ul_dino_checkpoint_roundtrip_decoder_free(tmp_path: Path):
    adapter = DINOULAdapter(in_channels=384, latent_channels=16, hidden_channels=64, adapter_type="mlp")
    prior = UNetModel(
        in_channels=16,
        out_channels=16,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(2,),
        num_head_channels=16,
        num_classes=None,
    )
    decoder = ULDiffusionDecoder(
        img_channels=3,
        latent_channels=16,
        downsample=8,
        cond_channels=8,
        model_channels=32,
        num_res_blocks=1,
        channel_mult=(1, 2),
        attention_resolutions=(),
        num_head_channels=16,
        num_classes=None,
    )
    alpha0, s0 = sigma0_to_alpha_sigma(0.08)
    ckpt = tmp_path / "ul_dino_ckpt.pt"
    torch.save(
        {
            "space": "ul_dino",
            "model": prior.state_dict(),
            "encoder": adapter.state_dict(),
            "decoder": decoder.state_dict(),
            "model_config": get_model_config(prior),
            "encoder_config": adapter.config_dict(),
            "decoder_config": decoder.config_dict(),
            "ul_config": {"sigma0": s0, "alpha0": alpha0, "t1": sigma0_to_fm_t1(s0)},
            "epoch": 10,
        },
        ckpt,
    )
    p_loaded, a_loaded, d_loaded, meta = load_ul_components(str(ckpt), torch.device("cpu"), load_decoder=False)
    assert d_loaded is None
    assert meta["space"] == "ul_dino"
    z = a_loaded(torch.randn(1, 384, 16, 16))
    assert z.shape == (1, 16, 16, 16)
    hmap = anomaly_heatmap(p_loaded, z, num_paths=2, num_steps=3, t_range=(0.0, sigma0_to_fm_t1(s0)), output_size=(128, 128))
    assert hmap.shape == (128, 128) and torch.isfinite(hmap).all()
    nats_d = estimate_ul_bitrate_nats_per_dim(p_loaded, z, sigma0=s0, num_paths=2, num_steps=3)
    assert nats_d > 0


def test_paired_latent_image_dataset(tmp_path: Path):
    img_path = tmp_path / "img0.png"
    Image.fromarray(np.full((64, 64, 3), 128, dtype=np.uint8)).save(img_path)
    feats = torch.randn(3, 384, 16, 16).half()
    stdz = Standardizer.fit(feats)
    torch.save(
        {"latents": feats, "labels": torch.tensor([0, 0, 0]), "paths": [str(img_path)] * 3, "class_names": ["bottle"]},
        tmp_path / "train_latents.pt",
    )
    ds = PairedLatentImageDataset(str(tmp_path), img_size=128, mean=NORM_STATS["mvtec"][0], std=NORM_STATS["mvtec"][1], standardizer=stdz)
    f0, x0, l0 = ds[0]
    assert f0.shape == (384, 16, 16) and x0.shape == (3, 128, 128) and l0 == 0


def test_mvtec_loco_directory_mask_union(tmp_path: Path):
    """MVTec-LOCO stores multiple binary mask PNGs inside ground_truth/<defect>/<stem>/."""
    cls_dir = tmp_path / "breakfast_box"
    (cls_dir / "test" / "good").mkdir(parents=True)
    (cls_dir / "test" / "logical_anomalies").mkdir(parents=True)
    (cls_dir / "ground_truth" / "logical_anomalies" / "000").mkdir(parents=True)

    img = Image.fromarray(np.zeros((32, 32, 3), dtype=np.uint8))
    img.save(cls_dir / "test" / "good" / "000.png")
    img.save(cls_dir / "test" / "logical_anomalies" / "000.png")

    # Two disjoint masks under ground_truth/logical_anomalies/000/ -> must be unioned!
    m1 = np.zeros((32, 32), dtype=np.uint8)
    m1[:8, :8] = 255
    m2 = np.zeros((32, 32), dtype=np.uint8)
    m2[24:, 24:] = 255
    Image.fromarray(m1).save(cls_dir / "ground_truth" / "logical_anomalies" / "000" / "000.png")
    Image.fromarray(m2).save(cls_dir / "ground_truth" / "logical_anomalies" / "000" / "001.png")

    found = find_mask(cls_dir / "ground_truth" / "logical_anomalies", "000")
    assert found is not None and found.is_dir()

    ds = TestDataset(str(tmp_path), "breakfast_box", img_size=32, mean=NORM_STATS["mvtec_loco"][0], std=NORM_STATS["mvtec_loco"][1])
    assert len(ds) == 2
    items_by_name = {name: (mask, lbl) for _, mask, lbl, name in [ds[0], ds[1]]}
    good_mask, good_lbl = items_by_name["good_000"]
    log_mask, log_lbl = items_by_name["logical_anomalies_000"]
    assert good_lbl == 0 and float(good_mask.sum()) == 0.0
    assert log_lbl == 1 and float(log_mask.sum()) == 128.0  # 64 + 64 unioned pixels
