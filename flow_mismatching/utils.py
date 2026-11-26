"""Checkpoint, mixed-precision and visualization helpers."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import torch

from .unet import UNetModel

AMP_DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "off": None}


def resolve_amp_dtype(name: str, device: torch.device) -> Optional[torch.dtype]:
    """Autocast dtype for ``name`` in {"bf16", "fp16", "off"}; None on CPU or when off.

    Falls back from bf16 to fp16 on GPUs without bf16 support.
    """
    dtype = AMP_DTYPES[name]
    if device.type != "cuda" or dtype is None:
        return None
    if dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        print("bf16 is not supported on this GPU, using fp16 instead.")
        return torch.float16
    return dtype


def enable_tf32() -> None:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision("high")


def latest_checkpoint(run_dir: Path) -> Optional[Path]:
    checkpoints = sorted(Path(run_dir).glob("ckpt_epoch*.pt"))
    return checkpoints[-1] if checkpoints else None


MODEL_CONFIG_KEYS = (
    "in_channels",
    "model_channels",
    "out_channels",
    "num_res_blocks",
    "attention_resolutions",
    "dropout",
    "channel_mult",
    "conv_resample",
    "dims",
    "num_classes",
    "num_heads",
    "num_head_channels",
    "num_heads_upsample",
    "use_scale_shift_norm",
    "resblock_updown",
    "use_new_attention_order",
    "with_fourier_features",
    "ignore_time",
    "input_projection",
)


def unwrap_model(model: torch.nn.Module) -> torch.nn.Module:
    """Unwrap DDP / torch.compile wrappers to access the underlying module."""
    while hasattr(model, "module") or hasattr(model, "_orig_mod"):
        if hasattr(model, "module"):
            model = model.module
        elif hasattr(model, "_orig_mod"):
            model = model._orig_mod
    return model


def get_model_config(model: UNetModel) -> Dict[str, Any]:
    """Serialize UNetModel architecture kwargs into a JSON/checkpoint-friendly dict."""
    base = unwrap_model(model)
    cfg: Dict[str, Any] = {}
    for k in MODEL_CONFIG_KEYS:
        val = getattr(base, k)
        if isinstance(val, tuple):
            val = list(val)
        cfg[k] = val
    return cfg


def build_unet_from_config(model_config: Optional[Dict[str, Any]], num_classes: Optional[int] = None) -> UNetModel:
    """Instantiate UNetModel from a saved model_config dict (falls back to default pixel config)."""
    if not model_config:
        return UNetModel(num_classes=num_classes)
    kwargs = dict(model_config)
    for tuple_key in ("attention_resolutions", "channel_mult"):
        if tuple_key in kwargs and isinstance(kwargs[tuple_key], list):
            kwargs[tuple_key] = tuple(kwargs[tuple_key])
    if num_classes is not None and kwargs.get("num_classes") is None:
        kwargs["num_classes"] = num_classes
    return UNetModel(**kwargs)


def load_model(ckpt_path: str, device: torch.device) -> Tuple[UNetModel, Dict[str, Any]]:
    """Build the U-Net from a training checkpoint and load its weights.

    Supports both legacy pixel-space checkpoints (no ``model_config``) and latent-space
    checkpoints with saved ``model_config``, ``encoder_config``, and ``latent_stats``.
    Strips ``module.`` (DDP) and ``_orig_mod.`` (torch.compile) state-dict prefixes.

    Returns:
        The model in eval mode and the checkpoint metadata (everything except the weights
        and optimizer states).
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state = ckpt["model"] if "model" in ckpt else ckpt
    for prefix in ("module.", "_orig_mod."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}

    num_classes = state["label_emb.weight"].shape[0] - 1 if "label_emb.weight" in state else None
    model_config = ckpt.get("model_config") if isinstance(ckpt, dict) else None
    model = build_unet_from_config(model_config, num_classes=num_classes)
    model.load_state_dict(state)
    model.to(device).eval()

    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "scaler", "encoder", "decoder")} if "model" in ckpt else {}
    meta["num_classes"] = num_classes
    meta.setdefault("space", "pixel")
    return model, meta


def _strip_state_prefixes(state: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    for prefix in ("module.", "_orig_mod."):
        if state and all(k.startswith(prefix) for k in state):
            state = {k[len(prefix) :]: v for k, v in state.items()}
    return state


def load_ul_components(
    ckpt_path: str,
    device: torch.device,
    load_decoder: bool = False,
) -> Tuple[UNetModel, torch.nn.Module, Optional[torch.nn.Module], Dict[str, Any]]:
    """Load UL flow prior v_theta, learned encoder E_psi, and (optionally) decoder D_omega.

    By default ``load_decoder=False`` so the diffusion decoder is discarded at test time.
    """
    from .ul_models import build_ul_decoder_from_config, build_ul_encoder_from_config

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    if ckpt.get("space") not in ("ul", "ul_dino"):
        raise ValueError(f"Expected a space in ('ul', 'ul_dino'), got space={ckpt.get('space')!r}")

    prior_state = _strip_state_prefixes(ckpt["model"])
    num_classes = prior_state["label_emb.weight"].shape[0] - 1 if "label_emb.weight" in prior_state else None
    prior = build_unet_from_config(ckpt.get("model_config"), num_classes=num_classes)
    prior.load_state_dict(prior_state)
    prior.to(device).eval()

    encoder = build_ul_encoder_from_config(ckpt["encoder_config"])
    encoder.load_state_dict(_strip_state_prefixes(ckpt["encoder"]))
    encoder.to(device).eval()

    decoder = None
    if load_decoder and "decoder" in ckpt and ckpt.get("decoder_config"):
        decoder = build_ul_decoder_from_config(ckpt["decoder_config"])
        decoder.load_state_dict(_strip_state_prefixes(ckpt["decoder"]))
        decoder.to(device).eval()

    meta = {k: v for k, v in ckpt.items() if k not in ("model", "optimizer", "scaler", "encoder", "decoder")}
    meta["num_classes"] = num_classes
    return prior, encoder, decoder, meta


def denormalize(x: torch.Tensor, mean: Sequence[float], std: Sequence[float]) -> torch.Tensor:
    """Undo ``transforms.Normalize`` for an image batch [B, 3, H, W] or image [3, H, W]; clamps to [0, 1]."""
    shape = (-1, 1, 1)
    mean = torch.as_tensor(mean, device=x.device, dtype=x.dtype).view(shape)
    std = torch.as_tensor(std, device=x.device, dtype=x.dtype).view(shape)
    return (x * std + mean).clamp(0, 1)


def save_visualization(image: torch.Tensor, heatmap: torch.Tensor, mask: torch.Tensor, path: Path) -> None:
    """Save a row of (image | heatmap | ground truth).

    Args:
        image: [3, H, W] in [0, 1]; heatmap: [H, W]; mask: [H, W] or [1, H, W].
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12, 4), dpi=150)
    axes[0].imshow(image.permute(1, 2, 0).cpu().numpy())
    axes[1].imshow(heatmap.cpu().numpy(), cmap="jet")
    axes[2].imshow(mask.squeeze().cpu().numpy(), cmap="gray")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
