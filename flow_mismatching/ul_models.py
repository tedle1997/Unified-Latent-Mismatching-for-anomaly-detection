"""Neural architectures for Unified Latents (Setup B: learned encoder + train-only diffusion decoder).

At test time for Flow Mismatching anomaly detection, ONLY the deterministic encoder E_psi
and the jointly trained flow prior v_theta (UNetModel) are used — ULDiffusionDecoder is
discarded after training (kept only for optional reconstruction PSNR / OOD diagnostics).
"""

import math
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .unet import GroupNorm32, UNetModel


def _gn(channels: int) -> nn.GroupNorm:
    groups = 32
    while channels % groups != 0 and groups > 1:
        groups //= 2
    return GroupNorm32(groups, channels)


class _EncResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, dropout: float = 0.0):
        super().__init__()
        self.block1 = nn.Sequential(_gn(in_ch), nn.SiLU(), nn.Conv2d(in_ch, out_ch, 3, padding=1))
        self.block2 = nn.Sequential(
            _gn(out_ch),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
        )
        self.skip = nn.Identity() if in_ch == out_ch else nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.skip(x) + self.block2(self.block1(x))


class ULEncoder(nn.Module):
    """Deterministic convolutional encoder mapping [B, 3, H, W] -> [B, latent_channels, H/ds, W/ds].

    Supports power-of-2 downsampling factors (4 -> 32x32 from 128x128; 8 -> 16x16 from 128x128).
    """

    def __init__(
        self,
        in_channels: int = 3,
        latent_channels: int = 8,
        base_channels: int = 64,
        downsample: int = 4,
        num_res_blocks: int = 2,
        dropout: float = 0.0,
    ):
        super().__init__()
        if downsample not in (2, 4, 8, 16):
            raise ValueError(f"downsample must be one of (2, 4, 8, 16), got {downsample}")
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.base_channels = int(base_channels)
        self.downsample = int(downsample)
        self.num_res_blocks = int(num_res_blocks)
        self.dropout = float(dropout)

        num_stages = int(round(math.log2(self.downsample)))
        self.stem = nn.Conv2d(self.in_channels, self.base_channels, 3, padding=1)

        stages = []
        ch = self.base_channels
        for stage_idx in range(num_stages):
            out_ch = min(self.base_channels * (2 ** (stage_idx + 1)), 256)
            layers = [_EncResBlock(ch, ch, dropout=self.dropout) for _ in range(self.num_res_blocks)]
            # Strided 3x3 conv downsampling.
            layers.append(nn.Conv2d(ch, out_ch, 3, stride=2, padding=1))
            stages.append(nn.Sequential(*layers))
            ch = out_ch

        self.stages = nn.ModuleList(stages)
        self.mid = nn.Sequential(*[_EncResBlock(ch, ch, dropout=self.dropout) for _ in range(self.num_res_blocks)])
        self.head = nn.Sequential(
            _gn(ch),
            nn.SiLU(),
            nn.Conv2d(ch, self.latent_channels, 3, padding=1),
        )
        # Keep O(1) initial latent variance so z_0 = alpha_0*z + sigma_0*eps starts informative.
        nn.init.zeros_(self.head[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.stem(x)
        for stage in self.stages:
            h = stage(h)
        h = self.mid(h)
        return self.head(h)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "ul_cnn",
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "base_channels": self.base_channels,
            "downsample": self.downsample,
            "num_res_blocks": self.num_res_blocks,
            "dropout": self.dropout,
        }


class ULDiffusionDecoder(nn.Module):
    """Lightweight train-only pixel Flow-Matching diffusion decoder conditioned on noisy latent z_0.

    Upsamples z_0 [B, C_z, H/ds, W/ds] -> [B, cond_channels, H, W] via learned 2x residual
    upsampling stages and concatenates with noisy image x_t [B, 3, H, W] into a compact UNetModel.
    """

    def __init__(
        self,
        img_channels: int = 3,
        latent_channels: int = 8,
        downsample: int = 4,
        cond_channels: int = 16,
        model_channels: int = 64,
        num_res_blocks: int = 2,
        channel_mult: Tuple[int, ...] = (1, 2, 2, 4),
        attention_resolutions: Tuple[int, ...] = (8,),
        dropout: float = 0.05,
        num_head_channels: int = 32,
        num_classes: int | None = None,
    ):
        super().__init__()
        if downsample not in (2, 4, 8, 16):
            raise ValueError(f"downsample must be one of (2, 4, 8, 16), got {downsample}")
        self.img_channels = int(img_channels)
        self.latent_channels = int(latent_channels)
        self.downsample = int(downsample)
        self.cond_channels = int(cond_channels)
        self.model_channels = int(model_channels)
        self.num_res_blocks = int(num_res_blocks)
        self.channel_mult = tuple(int(c) for c in channel_mult)
        self.attention_resolutions = tuple(int(r) for r in attention_resolutions)
        self.dropout = float(dropout)
        self.num_head_channels = int(num_head_channels)
        self.num_classes = num_classes

        # Learned upsampler mapping z_0 [B, C_z, H/ds, W/ds] -> [B, cond_channels, H, W].
        num_up = int(round(math.log2(self.downsample)))
        up_layers = [nn.Conv2d(self.latent_channels, max(32, self.cond_channels * 2), 3, padding=1)]
        ch = max(32, self.cond_channels * 2)
        for i in range(num_up):
            next_ch = self.cond_channels if i == num_up - 1 else ch
            up_layers.extend(
                [
                    _gn(ch),
                    nn.SiLU(),
                    nn.Upsample(scale_factor=2, mode="nearest"),
                    nn.Conv2d(ch, next_ch, 3, padding=1),
                ]
            )
            ch = next_ch
        up_layers.extend([_gn(ch), nn.SiLU()])
        self.cond_upsampler = nn.Sequential(*up_layers)

        self.unet = UNetModel(
            in_channels=self.img_channels + self.cond_channels,
            out_channels=self.img_channels,
            model_channels=self.model_channels,
            num_res_blocks=self.num_res_blocks,
            channel_mult=self.channel_mult,
            attention_resolutions=self.attention_resolutions,
            dropout=self.dropout,
            num_head_channels=self.num_head_channels,
            num_classes=self.num_classes,
        )

    def forward(self, x_t: torch.Tensor, t: torch.Tensor, z_0: torch.Tensor, extra: Dict | None = None) -> torch.Tensor:
        cond = self.cond_upsampler(z_0)
        if cond.shape[-2:] != x_t.shape[-2:]:
            cond = F.interpolate(cond, size=x_t.shape[-2:], mode="bilinear", align_corners=False)
        extra_in = dict(extra) if extra else {}
        extra_in["concat_conditioning"] = cond
        return self.unet(x_t, t, extra=extra_in)

    @torch.inference_mode()
    def sample(
        self,
        z_0: torch.Tensor,
        img_shape: Tuple[int, int] | None = None,
        steps: int = 25,
        extra: Dict | None = None,
        amp_dtype: torch.dtype | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Reconstruct normalized images from z_0 using midpoint/Heun ODE integration (t: 0 -> 1)."""
        device = z_0.device
        b, _, h, w = z_0.shape
        out_h, out_w = img_shape if img_shape is not None else (h * self.downsample, w * self.downsample)
        use_amp = amp_dtype is not None and device.type == "cuda"

        cond = self.cond_upsampler(z_0)
        if cond.shape[-2:] != (out_h, out_w):
            cond = F.interpolate(cond, size=(out_h, out_w), mode="bilinear", align_corners=False)
        extra_in = dict(extra) if extra else {}
        extra_in["concat_conditioning"] = cond

        x = torch.randn((b, self.img_channels, out_h, out_w), device=device, dtype=torch.float32, generator=generator)
        dt = 1.0 / float(steps)
        for i in range(steps):
            t0 = torch.full((b,), i * dt, device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                v0 = self.unet(x, t0, extra=extra_in).float()
            x_mid = x + 0.5 * dt * v0
            t_mid = torch.full((b,), (i + 0.5) * dt, device=device, dtype=torch.float32)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                v_mid = self.unet(x_mid, t_mid, extra=extra_in).float()
            x = x + dt * v_mid
        return x

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "ul_diff_decoder",
            "img_channels": self.img_channels,
            "latent_channels": self.latent_channels,
            "downsample": self.downsample,
            "cond_channels": self.cond_channels,
            "model_channels": self.model_channels,
            "num_res_blocks": self.num_res_blocks,
            "channel_mult": list(self.channel_mult),
            "attention_resolutions": list(self.attention_resolutions),
            "dropout": self.dropout,
            "num_head_channels": self.num_head_channels,
            "num_classes": self.num_classes,
        }


def build_ul_encoder_from_config(cfg: Dict[str, Any]) -> nn.Module:
    enc_type = cfg.get("type", "ul_cnn")
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    if enc_type == "ul_cnn":
        return ULEncoder(**kwargs)
    if enc_type == "ul_dino_adapter":
        return DINOULAdapter(**kwargs)
    raise ValueError(f"Unsupported UL encoder type: {enc_type!r}")


class DINOULAdapter(nn.Module):
    """Trainable UL adapter projecting frozen DINOv2 patch grids [B, in_dim, h, w] -> [B, latent_channels, h, w].

    Supports:
        - adapter_type='mlp': 2-layer per-patch MLP (1x1 convs) preserving strict spatial locality.
        - adapter_type='conv': 2-layer 3x3 ConvNet with local spatial mixing.
    """

    def __init__(
        self,
        in_channels: int = 384,
        latent_channels: int = 16,
        hidden_channels: int = 256,
        adapter_type: str = "mlp",
        dropout: float = 0.0,
    ):
        super().__init__()
        if adapter_type not in ("mlp", "conv"):
            raise ValueError(f"adapter_type must be 'mlp' or 'conv', got {adapter_type!r}")
        self.in_channels = int(in_channels)
        self.latent_channels = int(latent_channels)
        self.hidden_channels = int(hidden_channels)
        self.adapter_type = adapter_type
        self.dropout = float(dropout)

        k, p = (1, 0) if adapter_type == "mlp" else (3, 1)
        self.net = nn.Sequential(
            _gn(self.in_channels),
            nn.Conv2d(self.in_channels, self.hidden_channels, kernel_size=k, padding=p),
            _gn(self.hidden_channels),
            nn.SiLU(),
            nn.Dropout(self.dropout) if self.dropout > 0 else nn.Identity(),
            nn.Conv2d(self.hidden_channels, self.latent_channels, kernel_size=k, padding=p),
        )
        # Keep O(1) initial latent variance so z_0 starts with high SNR.
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return self.net(feat)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "ul_dino_adapter",
            "in_channels": self.in_channels,
            "latent_channels": self.latent_channels,
            "hidden_channels": self.hidden_channels,
            "adapter_type": self.adapter_type,
            "dropout": self.dropout,
        }


def build_ul_decoder_from_config(cfg: Dict[str, Any]) -> ULDiffusionDecoder:
    kwargs = {k: v for k, v in cfg.items() if k != "type"}
    for key in ("channel_mult", "attention_resolutions"):
        if key in kwargs and isinstance(kwargs[key], list):
            kwargs[key] = tuple(kwargs[key])
    return ULDiffusionDecoder(**kwargs)
