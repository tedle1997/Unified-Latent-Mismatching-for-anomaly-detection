"""Frozen pretrained encoders (SD-VAE, DINOv2) and per-channel latent standardization."""

from typing import Any, Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn
from torchvision import transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class Standardizer(nn.Module):
    """Per-channel zero-mean unit-variance normalizer for [B, C, H, W] or [C, H, W] latents."""

    def __init__(self, mean: Sequence[float], std: Sequence[float], eps: float = 1e-6):
        super().__init__()
        mean_t = torch.as_tensor(mean, dtype=torch.float32).flatten()
        std_t = torch.as_tensor(std, dtype=torch.float32).flatten().clamp(min=eps)
        if mean_t.shape != std_t.shape:
            raise ValueError(f"mean shape {tuple(mean_t.shape)} != std shape {tuple(std_t.shape)}")
        self.register_buffer("mean", mean_t)
        self.register_buffer("std", std_t)
        self.eps = float(eps)

    @classmethod
    def fit(cls, latents: torch.Tensor, eps: float = 1e-6) -> "Standardizer":
        """Fit per-channel mean and std from [N, C, H, W] normal training latents."""
        if latents.ndim != 4:
            raise ValueError(f"Expected [N, C, H, W] tensor, got {tuple(latents.shape)}")
        x = latents.float()
        mean = x.mean(dim=(0, 2, 3))
        std = x.std(dim=(0, 2, 3), unbiased=False).clamp(min=eps)
        return cls(mean, std, eps=eps)

    @classmethod
    def from_dict(cls, stats: Dict[str, Any]) -> "Standardizer":
        return cls(stats["mean"], stats["std"], eps=stats.get("eps", 1e-6))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean": [float(v) for v in self.mean.detach().cpu().tolist()],
            "std": [float(v) for v in self.std.detach().cpu().tolist()],
            "eps": self.eps,
        }

    def _view_shape(self, x: torch.Tensor) -> Tuple[int, ...]:
        if x.ndim == 4:
            return (1, -1, 1, 1)
        if x.ndim == 3:
            return (-1, 1, 1)
        raise ValueError(f"Expected 3D or 4D tensor, got shape {tuple(x.shape)}")

    def normalize(self, z: torch.Tensor) -> torch.Tensor:
        shape = self._view_shape(z)
        mean = self.mean.to(device=z.device, dtype=z.dtype).view(shape)
        std = self.std.to(device=z.device, dtype=z.dtype).view(shape)
        return (z - mean) / std

    def denormalize(self, z: torch.Tensor) -> torch.Tensor:
        shape = self._view_shape(z)
        mean = self.mean.to(device=z.device, dtype=z.dtype).view(shape)
        std = self.std.to(device=z.device, dtype=z.dtype).view(shape)
        return z * std + mean


class SDVAEEncoder(nn.Module):
    """Frozen Stable Diffusion VAE encoder (256x256 RGB -> 32x32x4 latents by default)."""

    def __init__(
        self,
        model_id: str = "stabilityai/sd-vae-ft-mse",
        input_size: int = 256,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from diffusers import AutoencoderKL

        self.model_id = model_id
        self.input_size = int(input_size)
        self.latent_channels = 4
        self.downsample_factor = 8
        self.latent_size = self.input_size // self.downsample_factor
        self.vae = AutoencoderKL.from_pretrained(model_id)
        self.vae.eval()
        self.vae.requires_grad_(False)
        if device is not None:
            self.vae.to(device=device, dtype=dtype)

    def preprocess_transform(self) -> transforms.Compose:
        """Transform PIL RGB image -> tensor in [-1, 1] of size [3, input_size, input_size]."""
        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ]
        )

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode [-1, 1] RGB tensor [B, 3, H, W] to deterministic posterior mean [B, 4, H/8, W/8]."""
        x = x.to(device=next(self.vae.parameters()).device, dtype=next(self.vae.parameters()).dtype)
        return self.vae.encode(x).latent_dist.mean

    @torch.inference_mode()
    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Decode raw (unstandardized) VAE latents [B, 4, h, w] to RGB images in [0, 1]."""
        z = z.to(device=next(self.vae.parameters()).device, dtype=next(self.vae.parameters()).dtype)
        imgs = self.vae.decode(z).sample
        return ((imgs.float() + 1.0) * 0.5).clamp(0.0, 1.0)

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "sdvae",
            "model_id": self.model_id,
            "input_size": self.input_size,
            "latent_channels": self.latent_channels,
            "latent_size": self.latent_size,
        }


class DINOv2Encoder(nn.Module):
    """Frozen DINOv2 patch-feature encoder (224x224 RGB -> 16x16x384 for ViT-S/14)."""

    def __init__(
        self,
        model_id: str = "facebook/dinov2-small",
        input_size: int = 224,
        device: Optional[torch.device] = None,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        from transformers import AutoModel

        self.model_id = model_id
        self.input_size = int(input_size)
        self.model = AutoModel.from_pretrained(model_id)
        self.model.eval()
        self.model.requires_grad_(False)
        self.patch_size = int(self.model.config.patch_size)
        if self.input_size % self.patch_size != 0:
            raise ValueError(f"input_size ({self.input_size}) must be divisible by patch_size ({self.patch_size})")
        self.latent_channels = int(self.model.config.hidden_size)
        self.latent_size = self.input_size // self.patch_size
        self.num_register_tokens = int(getattr(self.model.config, "num_register_tokens", 0))
        if device is not None:
            self.model.to(device=device, dtype=dtype)

    def preprocess_transform(self) -> transforms.Compose:
        """Transform PIL RGB image -> ImageNet-normalized tensor [3, input_size, input_size]."""
        return transforms.Compose(
            [
                transforms.Resize((self.input_size, self.input_size), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode ImageNet-normalized [B, 3, H, W] -> spatial patch grid [B, C, H/p, W/p]."""
        param = next(self.model.parameters())
        x = x.to(device=param.device, dtype=param.dtype)
        out = self.model(pixel_values=x).last_hidden_state
        # Drop CLS token (index 0) and any register tokens that follow it.
        patch_tokens = out[:, 1 + self.num_register_tokens :, :]
        b, n, c = patch_tokens.shape
        g = self.latent_size
        if n != g * g:
            raise RuntimeError(f"Expected {g * g} patch tokens for input_size={self.input_size}, got {n}")
        return patch_tokens.transpose(1, 2).reshape(b, c, g, g).contiguous()

    def config_dict(self) -> Dict[str, Any]:
        return {
            "type": "dinov2",
            "model_id": self.model_id,
            "input_size": self.input_size,
            "patch_size": self.patch_size,
            "latent_channels": self.latent_channels,
            "latent_size": self.latent_size,
        }


def build_encoder(
    encoder_type: str,
    device: torch.device,
    model_id: Optional[str] = None,
    input_size: Optional[int] = None,
    dtype: torch.dtype = torch.float32,
):
    """Instantiate a frozen encoder by short name ('sdvae' or 'dinov2')."""
    if encoder_type == "sdvae":
        return SDVAEEncoder(
            model_id=model_id or "stabilityai/sd-vae-ft-mse",
            input_size=input_size or 256,
            device=device,
            dtype=dtype,
        )
    if encoder_type == "dinov2":
        return DINOv2Encoder(
            model_id=model_id or "facebook/dinov2-small",
            input_size=input_size or 224,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported encoder_type={encoder_type!r}; expected 'sdvae' or 'dinov2'")
