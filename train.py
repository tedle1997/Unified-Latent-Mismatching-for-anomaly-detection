"""Train the flow matching velocity network on normal training images (pixel, frozen VAE, or Unified Latents).

Examples:
    # Pixel space (Setup A0)
    python train.py --dataset mvtec --data_root ../data/mvtec --no-class_cond

    # Frozen SD-VAE latent space (Setup A1)
    python train.py --dataset mvtec --space vae --latent_cache ../cache/mvtec_sdvae --no-class_cond

    # Unified Latents from scratch (Setup B: joint encoder + flow prior + train-only diffusion decoder)
    python train.py --dataset mvtec --space ul --data_root ../data/mvtec --no-class_cond --ul_sigma0 0.08

    # Several GPUs (DistributedDataParallel)
    torchrun --nproc_per_node 4 train.py --dataset mvtec --data_root ../data/mvtec
"""

import argparse
import contextlib
import importlib.util
import math
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler
from tqdm import tqdm

from flow_mismatching.data import (
    NORM_STATS,
    LatentTrainDataset,
    PairedLatentImageDataset,
    TrainDataset,
    discover_classes,
    load_latent_cache_meta,
)
from flow_mismatching.encoders import Standardizer, build_encoder
from flow_mismatching.optim import StableAdamW
from flow_mismatching.ul import (
    LN2,
    decoder_sigmoid_weight_fm_velocity,
    encode_with_fixed_noise,
    fm_prior_elbo_nats,
    sigma0_to_alpha_sigma,
    sigma0_to_fm_t1,
)
from flow_mismatching.ul_models import DINOULAdapter, ULDiffusionDecoder, ULEncoder
from flow_mismatching.unet import UNetModel
from flow_mismatching.utils import (
    denormalize,
    enable_tf32,
    get_model_config,
    latest_checkpoint,
    resolve_amp_dtype,
    unwrap_model,
)


class _ULSystem(nn.Module):
    """Joint container for Unified Latents (encoder/adapter + prior + train-only diffusion decoder)."""

    def __init__(self, encoder: nn.Module, prior: UNetModel, decoder: ULDiffusionDecoder):
        super().__init__()
        self.encoder = encoder
        self.prior = prior
        self.decoder = decoder

    def forward(
        self,
        enc_in: torch.Tensor,
        x_target: torch.Tensor,
        t_z: torch.Tensor,
        eps_z: torch.Tensor,
        eps_z0: torch.Tensor,
        t_x: torch.Tensor,
        eps_x: torch.Tensor,
        sigma0: float,
        t1: float,
        decoder_bias: float,
        loss_factor: float,
        extra: dict,
        prior_weight: float = 1.0,
    ):
        z_clean = self.encoder(enc_in)

        # 1. Exact UL Flow-Matching prior ELBO on t_z in (0, t1).
        t_z_view = t_z.view(-1, 1, 1, 1)
        z_t = (1.0 - t_z_view) * eps_z + t_z_view * z_clean
        v_z_pred = self.prior(z_t, t_z, extra=extra)
        prior_nats = fm_prior_elbo_nats(v_z_pred, z_clean - eps_z, t_z, t1)

        # 2. UL train-only diffusion decoder conditioned on fixed-noise latent z_0.
        z_0 = encode_with_fixed_noise(z_clean, sigma0=sigma0, noise=eps_z0)
        t_x_view = t_x.view(-1, 1, 1, 1)
        x_t = (1.0 - t_x_view) * eps_x + t_x_view * x_target
        v_x_pred = self.decoder(x_t, t_x, z_0, extra=extra)
        dec_sq_err = (v_x_pred.float() - (x_target.float() - eps_x.float())).pow(2).flatten(1).sum(dim=1)
        dec_w = decoder_sigmoid_weight_fm_velocity(t_x, bias=decoder_bias, loss_factor=loss_factor)
        dec_nats = dec_w * dec_sq_err

        # Put both terms on the exact same per-image-dimension ELBO scale (nats per pixel dim).
        img_dims = float(x_target[0].numel())
        lat_dims = float(z_clean[0].numel())
        loss_prior = (prior_nats / img_dims).mean()
        loss_dec = (dec_nats / img_dims).mean()
        loss_total = float(prior_weight) * loss_prior + loss_dec

        prior_bpd = (prior_nats / (LN2 * lat_dims)).mean()
        z_std = z_clean.detach().float().std()
        return loss_total, loss_prior.detach(), loss_dec.detach(), prior_bpd.detach(), z_std


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NORM_STATS), help="dataset (sets the normalization)")
    parser.add_argument("--space", choices=["pixel", "vae", "ul", "ul_dino", "ul_stage2"], default="pixel", help="modeling space")
    parser.add_argument("--data_root", default=None, help="dataset root in the MVTec-AD folder layout (pixel / ul)")
    parser.add_argument("--latent_cache", default=None, help="cached latent directory from scripts/precompute_latents.py")
    parser.add_argument("--t_max", type=float, default=1.0,
                        help="upper timestep bound for non-ELBO Flow Matching training (e.g. 1.0 or UL t1)")
    parser.add_argument("--stage2_noise_std", type=float, default=0.0,
                        help="optional fixed Gaussian noise std added to cached latents in --space ul_stage2")
    parser.add_argument("--output_dir", default=None, help="checkpoints and samples (default: runs/<dataset>_<space>)")
    parser.add_argument("--img_size", type=int, default=128)
    parser.add_argument("--class_cond", action=argparse.BooleanOptionalAction, default=True,
                        help="condition the model on the class label")

    # U-Net overrides (None uses space-appropriate defaults).
    arch = parser.add_argument_group("Architecture")
    arch.add_argument("--model_channels", type=int, default=None)
    arch.add_argument("--num_res_blocks", type=int, default=None)
    arch.add_argument("--channel_mult", type=int, nargs="+", default=None)
    arch.add_argument("--attention_resolutions", type=int, nargs="+", default=None)
    arch.add_argument("--num_head_channels", type=int, default=32)
    arch.add_argument("--dropout", type=float, default=0.05)

    # Unified Latents (Setup B: ul, Setup C: ul_dino) hyperparameters.
    ul = parser.add_argument_group("Unified Latents (--space ul / ul_dino)")
    ul.add_argument("--ul_sigma0", type=float, default=0.08, help="fixed encoder noise std sigma_0 (default 0.08)")
    ul.add_argument("--ul_channels", type=int, default=None, help="latent channel count C (default: 8 for ul, 16 for ul_dino)")
    ul.add_argument("--ul_downsample", type=int, choices=[2, 4, 8], default=4,
                    help="spatial downsampling factor for --space ul (4 -> 32x32, 8 -> 16x16 at img_size=128)")
    ul.add_argument("--ul_enc_channels", type=int, default=64, help="base channels of UL encoder E_psi")
    ul.add_argument("--ul_adapter_type", choices=["mlp", "conv"], default="mlp", help="UL adapter type for --space ul_dino")
    ul.add_argument("--ul_adapter_hidden", type=int, default=256, help="hidden channels of DINOv2 UL adapter")
    ul.add_argument("--ul_dec_channels", type=int, default=32, help="base channels of train-only UL decoder D_omega (multiple of 32)")
    ul.add_argument("--ul_decoder_bias", type=float, default=4.0, help="sigmoid weighting bias b in decoder loss")
    ul.add_argument("--ul_loss_factor", type=float, default=2.5, help="decoder loss factor c_lf")
    ul.add_argument("--ul_prior_warmup_epochs", type=int, default=5,
                    help="linearly ramp prior weight 0.1->1.0 over first N epochs to prevent early posterior collapse")
    ul.add_argument("--ul_recon_steps", type=int, default=25, help="ODE steps for diagnostic UL reconstruction PSNR")

    parser.add_argument("--epochs", type=int, default=2000)
    parser.add_argument("--batch_size", type=int, default=32, help="batch size per GPU")
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--betas", type=float, nargs=2, default=(0.9, 0.99))
    parser.add_argument("--eps", type=float, default=1e-6)
    parser.add_argument("--update_clip", type=float, default=1.0, help="StableAdamW update-RMS clip; <= 0 disables")
    parser.add_argument("--grad_clip", type=float, default=0.0, help="gradient-norm clip; <= 0 disables")
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16", help="mixed precision")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 on Ampere+")
    parser.add_argument("--compile", action="store_true", help="compile model with torch.compile")
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--save_every", type=int, default=500, help="save a checkpoint every N epochs")
    parser.add_argument("--resume", action="store_true", help="resume from the latest checkpoint in output_dir")
    parser.add_argument("--num_samples", type=int, default=36, help="images generated at each checkpoint; 0 disables")
    parser.add_argument("--ode_method", default="heun3", help="ODE solver for the samples (a torchdiffeq method)")
    parser.add_argument("--ode_steps", type=int, default=100, help="ODE steps for the samples")
    return parser.parse_args()


def setup_device():
    """Return (device, rank, world_size); initializes the process group when launched with torchrun."""
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        local_rank = int(os.environ["LOCAL_RANK"])
        torch.cuda.set_device(local_rank)
        return torch.device("cuda", local_rank), dist.get_rank(), dist.get_world_size()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu"), 0, 1


def _stratified_uniform(batch_size: int, low: float, high: float, device: torch.device) -> torch.Tensor:
    """Low-variance stratified uniform samples in (low, high)."""
    u = (torch.arange(batch_size, device=device, dtype=torch.float32) + torch.rand(batch_size, device=device)) / float(
        max(batch_size, 1)
    )
    return low + (high - low) * u


@torch.inference_mode()
def save_samples(
    model,
    path,
    num_samples,
    sample_shape,
    num_classes,
    mean,
    std,
    ode_method,
    ode_steps,
    device,
    amp_dtype,
    latent_decode_fn=None,
    t_end: float = 1.0,
):
    """Generate samples by integrating the learned ODE from noise (t = 0) to t = t_end."""
    from flow_matching.solver import ODESolver
    from torchvision.utils import save_image

    model.eval()
    labels = torch.arange(num_samples, device=device) % num_classes if num_classes else None

    def velocity(x, t):
        t = torch.as_tensor(t, device=x.device, dtype=x.dtype).reshape(-1).expand(x.shape[0])
        return model(x, t, extra={"label": labels} if labels is not None else {})

    c, h, w = sample_shape
    x0 = torch.randn(num_samples, c, h, w, device=device)
    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        x1 = ODESolver(velocity_model=velocity).sample(
            x_init=x0,
            step_size=float(t_end) / float(ode_steps),
            method=ode_method,
            time_grid=torch.tensor([0.0, float(t_end)], device=device),
        )
    if latent_decode_fn is not None:
        imgs = latent_decode_fn(x1.float())
    else:
        imgs = denormalize(x1.float(), mean, std)
    save_image(imgs, path, nrow=int(math.sqrt(num_samples)))


@torch.inference_mode()
def save_ul_reconstructions(
    encoder: nn.Module,
    decoder: ULDiffusionDecoder,
    dataset,
    path: Path,
    num_images: int,
    sigma0: float,
    mean,
    std,
    recon_steps: int,
    class_cond: bool,
    device: torch.device,
    amp_dtype,
    is_paired_latent: bool = False,
) -> float:
    """Save (original | UL reconstruction) diagnostic grid and return mean reconstruction PSNR (dB)."""
    from torchvision.utils import save_image

    encoder.eval()
    decoder.eval()
    n = min(num_images, len(dataset))
    idxs = torch.linspace(0, len(dataset) - 1, steps=n).long().tolist()
    if is_paired_latent:
        feats, xs, lbls = zip(*[dataset[i] for i in idxs])
        enc_in = torch.stack(feats, dim=0).to(device)
        x = torch.stack(xs, dim=0).to(device)
    else:
        xs, lbls = zip(*[dataset[i] for i in idxs])
        x = torch.stack(xs, dim=0).to(device)
        enc_in = x
    extra = {"label": torch.tensor(lbls, device=device, dtype=torch.long)} if class_cond else {}

    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
        z_clean = encoder(enc_in)
    z_0 = encode_with_fixed_noise(z_clean.float(), sigma0=sigma0)
    x_rec = decoder.sample(z_0, img_shape=x.shape[-2:], steps=recon_steps, extra=extra, amp_dtype=amp_dtype)

    orig_rgb = denormalize(x.float(), mean, std)
    rec_rgb = denormalize(x_rec.float(), mean, std)
    mse = (orig_rgb - rec_rgb).pow(2).flatten(1).mean(dim=1).clamp(min=1e-10)
    psnr = float((-10.0 * torch.log10(mse)).mean())

    interleaved = torch.stack([orig_rgb, rec_rgb], dim=1).flatten(0, 1)
    save_image(interleaved, path, nrow=min(8, 2 * int(math.sqrt(n))))
    return psnr


def build_unet_for_space(args, in_channels: int, num_classes, latent_spatial: int | None = None):
    if args.space == "pixel":
        model_channels = args.model_channels if args.model_channels is not None else 64
        num_res_blocks = args.num_res_blocks if args.num_res_blocks is not None else 3
        channel_mult = tuple(args.channel_mult) if args.channel_mult is not None else (1, 1, 2, 2, 4, 4)
        attention_resolutions = (
            tuple(args.attention_resolutions) if args.attention_resolutions is not None else (4, 8, 16, 32)
        )
    else:
        # Compact 3-stage latent U-Net (~20M params at model_channels=96).
        model_channels = args.model_channels if args.model_channels is not None else 96
        num_res_blocks = args.num_res_blocks if args.num_res_blocks is not None else 2
        channel_mult = tuple(args.channel_mult) if args.channel_mult is not None else (1, 2, 4)
        default_attn = (1, 2, 4) if (latent_spatial is not None and latent_spatial <= 16) else (2, 4)
        attention_resolutions = (
            tuple(args.attention_resolutions) if args.attention_resolutions is not None else default_attn
        )
    return UNetModel(
        in_channels=in_channels,
        out_channels=in_channels,
        model_channels=model_channels,
        num_res_blocks=num_res_blocks,
        channel_mult=channel_mult,
        attention_resolutions=attention_resolutions,
        dropout=args.dropout,
        num_head_channels=args.num_head_channels,
        num_classes=num_classes,
    )


def main():
    args = parse_args()
    device, rank, world_size = setup_device()
    distributed = dist.is_initialized()
    is_main = rank == 0
    torch.manual_seed(args.seed + rank)

    if args.num_samples > 0 and importlib.util.find_spec("flow_matching") is None:
        raise SystemExit("Generating samples needs `pip install flow_matching` (or pass --num_samples 0).")

    if args.tf32 and device.type == "cuda":
        enable_tf32()
    amp_dtype = resolve_amp_dtype(args.amp, device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp_dtype == torch.float16)

    # Data setup.
    mean, std = NORM_STATS[args.dataset]
    encoder_config = None
    dino_encoder_config = None
    decoder_config = None
    latent_stats = None
    standardizer = None
    ul_config = None
    is_ul = args.space in ("ul", "ul_dino")
    ul_channels = args.ul_channels if args.ul_channels is not None else (16 if args.space == "ul_dino" else 8)

    if args.space in ("pixel", "ul"):
        if not args.data_root:
            raise SystemExit(f"--data_root is required when --space {args.space}")
        class_names = discover_classes(args.data_root, "train")
        dataset = TrainDataset(args.data_root, class_names, args.img_size, mean, std)
        if args.space == "pixel":
            sample_shape = (3, args.img_size, args.img_size)
        else:
            if args.img_size % args.ul_downsample != 0:
                raise SystemExit(f"--img_size ({args.img_size}) must be divisible by --ul_downsample ({args.ul_downsample})")
            lat_hw = args.img_size // args.ul_downsample
            sample_shape = (ul_channels, lat_hw, lat_hw)
            ul_downsample = args.ul_downsample
    elif args.space == "ul_dino":
        if not args.latent_cache:
            raise SystemExit("--latent_cache (precomputed DINOv2 cache) is required when --space ul_dino")
        cache_meta = load_latent_cache_meta(args.latent_cache)
        class_names = list(cache_meta["class_names"])
        dino_encoder_config = cache_meta["encoder_config"]
        latent_stats = cache_meta["latent_stats"]
        standardizer = Standardizer.from_dict(latent_stats)
        feat_c, feat_h, feat_w = [int(v) for v in cache_meta["latent_shape"]]
        if feat_h != feat_w or args.img_size % feat_h != 0:
            raise SystemExit(f"--img_size ({args.img_size}) must be divisible by DINOv2 grid size ({feat_h}x{feat_w})")
        ul_downsample = args.img_size // feat_h
        dataset = PairedLatentImageDataset(args.latent_cache, args.img_size, mean, std, standardizer=standardizer)
        sample_shape = (ul_channels, feat_h, feat_w)
    else:
        # Both vae (frozen SD-VAE) and ul_stage2 (frozen Stage-1 UL encoder) train from cached latents.
        if not args.latent_cache:
            raise SystemExit(f"--latent_cache is required when --space {args.space}")
        cache_meta = load_latent_cache_meta(args.latent_cache)
        class_names = list(cache_meta["class_names"])
        encoder_config = cache_meta["encoder_config"]
        latent_stats = cache_meta["latent_stats"]
        standardizer = Standardizer.from_dict(latent_stats)
        dataset = LatentTrainDataset(args.latent_cache, standardizer=standardizer)
        sample_shape = tuple(int(v) for v in cache_meta["latent_shape"])
        if args.space == "ul_stage2":
            ul_config = encoder_config.get("ul_config")

    num_classes = len(class_names) if args.class_cond else None
    sampler = DistributedSampler(dataset, shuffle=True, seed=args.seed) if distributed else None
    use_workers = args.num_workers if args.space not in ("vae", "ul_stage2") else 0
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=use_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=use_workers > 0,
    )

    # Model construction.
    if is_ul:
        ul_t1 = sigma0_to_fm_t1(args.ul_sigma0)
        ul_alpha0, ul_sigma0 = sigma0_to_alpha_sigma(args.ul_sigma0)
        if args.space == "ul":
            ul_encoder = ULEncoder(
                in_channels=3,
                latent_channels=ul_channels,
                base_channels=args.ul_enc_channels,
                downsample=ul_downsample,
                num_res_blocks=2,
                dropout=0.0,
            )
        else:
            ul_encoder = DINOULAdapter(
                in_channels=int(cache_meta["latent_shape"][0]),
                latent_channels=ul_channels,
                hidden_channels=args.ul_adapter_hidden,
                adapter_type=args.ul_adapter_type,
                dropout=0.0,
            )
        ul_prior = build_unet_for_space(
            args, in_channels=ul_channels, num_classes=num_classes, latent_spatial=sample_shape[1]
        )
        ul_decoder = ULDiffusionDecoder(
            img_channels=3,
            latent_channels=ul_channels,
            downsample=ul_downsample,
            cond_channels=16,
            model_channels=args.ul_dec_channels,
            num_res_blocks=2,
            channel_mult=(1, 2, 2, 4),
            attention_resolutions=(8,),
            dropout=args.dropout,
            num_head_channels=args.num_head_channels,
            num_classes=num_classes,
        )
        model_config = get_model_config(ul_prior)
        encoder_config = ul_encoder.config_dict()
        decoder_config = ul_decoder.config_dict()
        ul_config = {
            "sigma0": float(ul_sigma0),
            "alpha0": float(ul_alpha0),
            "t1": float(ul_t1),
            "channels": int(ul_channels),
            "downsample": int(ul_downsample),
            "decoder_bias": float(args.ul_decoder_bias),
            "loss_factor": float(args.ul_loss_factor),
        }
        net = _ULSystem(ul_encoder, ul_prior, ul_decoder).to(device)
    else:
        net = build_unet_for_space(
            args, in_channels=sample_shape[0], num_classes=num_classes, latent_spatial=sample_shape[1]
        ).to(device)
        model_config = get_model_config(net)

    if args.compile:
        net = torch.compile(net)
    model = DDP(net, device_ids=[device.index]) if distributed else net
    optimizer = StableAdamW(
        net.parameters(),
        lr=args.lr,
        betas=tuple(args.betas),
        eps=args.eps,
        weight_decay=args.weight_decay,
        amsgrad=True,
        update_clip=args.update_clip,
    )

    default_out = Path("runs") / (args.dataset if args.space == "pixel" else f"{args.dataset}_{args.space}")
    out_dir = Path(args.output_dir or default_out)
    out_dir.mkdir(parents=True, exist_ok=True)
    start_epoch = 0
    ckpt_path = latest_checkpoint(out_dir) if args.resume else None
    if ckpt_path is not None:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
        if args.class_cond and ckpt.get("class_names") != class_names:
            raise ValueError(f"Classes in {ckpt_path} ({ckpt.get('class_names')}) differ from {class_names}")
        raw_net = unwrap_model(net)
        if is_ul:
            raw_net.prior.load_state_dict(ckpt["model"])
            raw_net.encoder.load_state_dict(ckpt["encoder"])
            raw_net.decoder.load_state_dict(ckpt["decoder"])
        else:
            raw_net.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        if scaler.is_enabled() and ckpt.get("scaler"):
            scaler.load_state_dict(ckpt["scaler"])
        start_epoch = int(ckpt["epoch"])

    # Lazy VAE decoder for checkpoint sample visualization (--space vae).
    vae_decoder = None

    def vae_decode_fn(z_std: torch.Tensor) -> torch.Tensor:
        nonlocal vae_decoder
        if vae_decoder is None:
            vae_decoder = build_encoder(
                encoder_config["type"],
                device=device,
                model_id=encoder_config.get("model_id"),
                input_size=encoder_config.get("input_size"),
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
            )
        return vae_decoder.decode(standardizer.denormalize(z_std))

    if is_main:
        raw_net = unwrap_model(net)
        print(f"Space: {args.space} | sample shape: {sample_shape}")
        print(f"{len(dataset)} training items from {len(class_names)} classes: {', '.join(class_names)}")
        if is_ul:
            p_enc = sum(p.numel() for p in raw_net.encoder.parameters()) / 1e6
            p_pri = sum(p.numel() for p in raw_net.prior.parameters()) / 1e6
            p_dec = sum(p.numel() for p in raw_net.decoder.parameters()) / 1e6
            enc_label = "adapter" if args.space == "ul_dino" else "encoder"
            print(
                f"UL models: {enc_label} {p_enc:.2f}M + prior {p_pri:.1f}M + train-only decoder {p_dec:.1f}M "
                f"(sigma0={ul_config['sigma0']:.4f}, t1={ul_config['t1']:.4f}, "
                f"b={args.ul_decoder_bias}, c_lf={args.ul_loss_factor})"
            )
        else:
            num_params = sum(p.numel() for p in raw_net.parameters()) / 1e6
            print(f"U-Net: {num_params:.1f}M parameters, {'class-conditional' if num_classes else 'unconditional'}")
        print(f"{world_size} process(es), batch size {args.batch_size} per process, AMP: {amp_dtype or 'off'}")
        if ckpt_path is not None:
            print(f"Resuming from {ckpt_path} at epoch {start_epoch}")

    t_train_start = time.time()
    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        if sampler is not None:
            sampler.set_epoch(epoch)
        loss_sum = torch.zeros((), device=device)
        prior_sum = torch.zeros((), device=device)
        dec_sum = torch.zeros((), device=device)
        bpd_sum = torch.zeros((), device=device)
        zstd_sum = torch.zeros((), device=device)
        if is_ul and args.ul_prior_warmup_epochs > 0 and epoch < args.ul_prior_warmup_epochs:
            prior_weight = 0.1 + 0.9 * (float(epoch) / float(args.ul_prior_warmup_epochs))
        else:
            prior_weight = 1.0
        progress = tqdm(loader, desc=f"Epoch {epoch + 1}/{args.epochs}", leave=False, disable=not is_main)
        for step, batch in enumerate(progress):
            if args.space == "ul_dino":
                feat, x1, labels = batch
                enc_in = feat.to(device, non_blocking=True)
                x1 = x1.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
            else:
                x1, labels = batch
                x1 = x1.to(device, non_blocking=True)
                enc_in = x1
                labels = labels.to(device, non_blocking=True)
            extra = {"label": labels} if args.class_cond else {}

            update = (step + 1) % args.accumulation_steps == 0 or step + 1 == len(loader)
            with contextlib.nullcontext() if update or not distributed else model.no_sync():
                if is_ul:
                    bs = x1.shape[0]
                    t_z = _stratified_uniform(bs, 1e-4, ul_config["t1"] - 1e-4, device)
                    t_x = _stratified_uniform(bs, 1e-3, 1.0 - 1e-3, device)
                    eps_z = torch.randn((bs, *sample_shape), device=device, dtype=x1.dtype)
                    eps_z0 = torch.randn((bs, *sample_shape), device=device, dtype=x1.dtype)
                    eps_x = torch.randn_like(x1)
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        loss, l_prior, l_dec, bpd, z_std = model(
                            enc_in,
                            x1,
                            t_z,
                            eps_z,
                            eps_z0,
                            t_x,
                            eps_x,
                            sigma0=ul_config["sigma0"],
                            t1=ul_config["t1"],
                            decoder_bias=args.ul_decoder_bias,
                            loss_factor=args.ul_loss_factor,
                            extra=extra,
                            prior_weight=prior_weight,
                        )
                    prior_sum += l_prior.float()
                    dec_sum += l_dec.float()
                    bpd_sum += bpd.float()
                    zstd_sum += z_std.float()
                else:
                    # Conditional flow matching on the affine path x_t = (1 - t) x_0 + t x_1.
                    if args.space == "ul_stage2" and args.stage2_noise_std > 0:
                        x1 = x1 + args.stage2_noise_std * torch.randn_like(x1)
                    t = torch.rand(x1.shape[0], device=device) * float(args.t_max)
                    x0 = torch.randn_like(x1)
                    t_ = t.view(-1, 1, 1, 1)
                    xt = (1.0 - t_) * x0 + t_ * x1
                    with torch.autocast(device.type, dtype=amp_dtype, enabled=amp_dtype is not None):
                        loss = F.mse_loss(model(xt, t, extra=extra), x1 - x0)
                scaler.scale(loss / args.accumulation_steps).backward()
            loss_sum += loss.detach().float()

            if update:
                scaler.unscale_(optimizer)
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(net.parameters(), args.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)

        n_steps = max(len(loader), 1)
        epoch_loss = loss_sum / n_steps
        if distributed:
            dist.all_reduce(epoch_loss)
            epoch_loss /= world_size
        if is_ul:
            epoch_prior = prior_sum / n_steps
            epoch_dec = dec_sum / n_steps
            epoch_bpd = bpd_sum / n_steps
            epoch_zstd = zstd_sum / n_steps
            if distributed:
                for t_stat in (epoch_prior, epoch_dec, epoch_bpd, epoch_zstd):
                    dist.all_reduce(t_stat)
                    t_stat /= world_size
            bits_per_px = epoch_prior.item() * 3.0 / LN2
            if is_main:
                print(
                    f"Epoch {epoch + 1}/{args.epochs} | loss {epoch_loss.item():.4f} "
                    f"(prior {epoch_prior.item():.4f}, dec {epoch_dec.item():.4f}) | "
                    f"latent bpd {epoch_bpd.item():.3f} ({bits_per_px:.3f} bpp) | z_std {epoch_zstd.item():.3f}"
                )
        else:
            if is_main:
                print(f"Epoch {epoch + 1}/{args.epochs} | loss {epoch_loss.item():.4f}")

        if is_main and ((epoch + 1) % args.save_every == 0 or epoch + 1 == args.epochs):
            ckpt_path = out_dir / f"ckpt_epoch{epoch + 1:04d}.pt"
            elapsed_s = time.time() - t_train_start
            raw_net = unwrap_model(net)
            payload = {
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict() if scaler.is_enabled() else None,
                "epoch": epoch + 1,
                "elapsed_sec": elapsed_s,
                "dataset": args.dataset,
                "space": args.space,
                "img_size": args.img_size,
                "sample_shape": list(sample_shape),
                "model_config": model_config,
                "encoder_config": encoder_config,
                "dino_encoder_config": dino_encoder_config,
                "decoder_config": decoder_config,
                "ul_config": ul_config,
                "latent_stats": latent_stats,
                "norm_mean": list(mean),
                "norm_std": list(std),
                "class_cond": args.class_cond,
                "num_classes": num_classes,
                "class_names": class_names if args.class_cond else None,
                "args": vars(args),
            }
            if is_ul:
                payload["model"] = raw_net.prior.state_dict()
                payload["encoder"] = raw_net.encoder.state_dict()
                payload["decoder"] = raw_net.decoder.state_dict()
                payload["train_latent_bpd"] = float(epoch_bpd.item())
            else:
                payload["model"] = raw_net.state_dict()

            torch.save(payload, ckpt_path)
            print(f"Saved {ckpt_path} ({elapsed_s / 60:.1f} min elapsed)")

            if args.num_samples > 0:
                sample_path = out_dir / f"samples_epoch{epoch + 1:04d}.png"
                if is_ul:
                    recon_path = out_dir / f"recon_epoch{epoch + 1:04d}.png"
                    psnr = save_ul_reconstructions(
                        raw_net.encoder,
                        raw_net.decoder,
                        dataset,
                        recon_path,
                        num_images=min(16, args.num_samples),
                        sigma0=ul_config["sigma0"],
                        mean=mean,
                        std=std,
                        recon_steps=args.ul_recon_steps,
                        class_cond=args.class_cond,
                        device=device,
                        amp_dtype=amp_dtype,
                        is_paired_latent=(args.space == "ul_dino"),
                    )
                    print(f"Saved {recon_path} (train recon PSNR: {psnr:.2f} dB)")

                    # Scale prior ODE endpoint z_{t1} by (alpha_0 + sigma_0) to match VP z_0 = alpha_0*z + sigma_0*eps.
                    z0_scale = ul_config["alpha0"] + ul_config["sigma0"]
                    def ul_decode_fn(z_t1: torch.Tensor) -> torch.Tensor:
                        z0 = z_t1 * z0_scale
                        rec = raw_net.decoder.sample(
                            z0, img_shape=(args.img_size, args.img_size), steps=args.ul_recon_steps, amp_dtype=amp_dtype
                        )
                        return denormalize(rec.float(), mean, std)

                    save_samples(
                        raw_net.prior,
                        sample_path,
                        args.num_samples,
                        sample_shape,
                        num_classes,
                        mean,
                        std,
                        args.ode_method,
                        args.ode_steps,
                        device,
                        amp_dtype,
                        latent_decode_fn=ul_decode_fn,
                        t_end=ul_config["t1"],
                    )
                elif args.space != "ul_stage2":
                    save_samples(
                        raw_net,
                        sample_path,
                        args.num_samples,
                        sample_shape,
                        num_classes,
                        mean,
                        std,
                        args.ode_method,
                        args.ode_steps,
                        device,
                        amp_dtype,
                        latent_decode_fn=vae_decode_fn if args.space == "vae" else None,
                    )
                    print(f"Saved {sample_path}")
                if is_ul:
                    print(f"Saved {sample_path}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
