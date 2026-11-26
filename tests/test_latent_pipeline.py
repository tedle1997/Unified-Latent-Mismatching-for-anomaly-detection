"""Unit tests for Phase 1 latent infrastructure: checkpoint config, Standardizer, and latent datasets."""

import json
from pathlib import Path

import torch

from flow_mismatching.data import LatentTestDataset, LatentTrainDataset
from flow_mismatching.encoders import Standardizer
from flow_mismatching.unet import UNetModel
from flow_mismatching.utils import get_model_config, load_model


def test_standardizer_roundtrip():
    torch.manual_seed(0)
    z = torch.randn(32, 4, 16, 16) * torch.tensor([0.5, 2.0, 3.0, 0.25]).view(1, 4, 1, 1) + torch.tensor(
        [-1.0, 2.0, 0.5, 4.0]
    ).view(1, 4, 1, 1)
    stdz = Standardizer.fit(z)
    z_norm = stdz.normalize(z)
    assert torch.allclose(z_norm.mean(dim=(0, 2, 3)), torch.zeros(4), atol=1e-5)
    assert torch.allclose(z_norm.std(dim=(0, 2, 3), unbiased=False), torch.ones(4), atol=1e-5)
    assert torch.allclose(stdz.denormalize(z_norm), z, atol=1e-4)

    # Dict serialization roundtrip + 3D single-sample normalization
    stdz2 = Standardizer.from_dict(stdz.to_dict())
    assert torch.allclose(stdz2.normalize(z[0]), z_norm[0], atol=1e-6)


def test_checkpoint_model_config_roundtrip(tmp_path: Path):
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
    ckpt_path = tmp_path / "latent_ckpt.pt"
    torch.save(
        {
            "model": model.state_dict(),
            "space": "vae",
            "model_config": get_model_config(model),
            "latent_stats": {"mean": [0.0, 0.0, 0.0, 0.0], "std": [1.0, 1.0, 1.0, 1.0], "eps": 1e-6},
            "epoch": 10,
        },
        ckpt_path,
    )
    loaded, meta = load_model(str(ckpt_path), torch.device("cpu"))
    assert meta["space"] == "vae"
    assert meta["num_classes"] is None
    assert loaded.in_channels == 4 and loaded.out_channels == 4
    assert loaded.channel_mult == (1, 2, 4)
    x = torch.randn(2, 4, 16, 16)
    t = torch.rand(2)
    with torch.no_grad():
        assert torch.allclose(model.eval()(x, t), loaded(x, t), atol=1e-6)


def test_latent_datasets(tmp_path: Path):
    (tmp_path / "test").mkdir()
    train_z = torch.randn(10, 4, 8, 8).half()
    stdz = Standardizer.fit(train_z)
    torch.save(
        {"latents": train_z, "labels": torch.zeros(10, dtype=torch.long), "paths": ["a.png"] * 10, "class_names": ["bottle"]},
        tmp_path / "train_latents.pt",
    )
    test_z = torch.randn(4, 4, 8, 8).half()
    torch.save(
        {
            "latents": test_z,
            "labels": torch.tensor([0, 0, 1, 1]),
            "names": ["good_0", "good_1", "bad_0", "bad_1"],
            "img_paths": ["x.png"] * 4,
            "mask_paths": ["", "", "", ""],
        },
        tmp_path / "test" / "bottle.pt",
    )
    with open(tmp_path / "meta.json", "w") as f:
        json.dump({"class_names": ["bottle"], "latent_shape": [4, 8, 8], "latent_stats": stdz.to_dict()}, f)

    ds_tr = LatentTrainDataset(str(tmp_path), standardizer=stdz)
    assert len(ds_tr) == 10
    z0, lbl0 = ds_tr[0]
    assert z0.shape == (4, 8, 8) and z0.dtype == torch.float32 and lbl0 == 0

    ds_te = LatentTestDataset(str(tmp_path), "bottle", mask_size=64, standardizer=stdz)
    assert len(ds_te) == 4
    zt, mask, lbl, name, img_p = ds_te[2]
    assert zt.shape == (4, 8, 8) and mask.shape == (1, 64, 64) and lbl == 1 and name == "bad_0" and img_p == "x.png"
