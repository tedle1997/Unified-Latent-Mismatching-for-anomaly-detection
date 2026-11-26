"""Precompute and cache fp16 latents (SD-VAE or DINOv2) for an MVTec-AD-layout dataset.

Examples:
    python scripts/precompute_latents.py --encoder sdvae --data_root ../data/mvtec --output_dir ../cache/mvtec_sdvae
    python scripts/precompute_latents.py --encoder dinov2 --data_root ../data/mvtec --output_dir ../cache/mvtec_dinov2
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

# Allow running from repo root without installing the package.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from flow_mismatching.data import NORM_STATS, discover_classes, find_mask, image_transform, list_images
from flow_mismatching.encoders import Standardizer, build_encoder
from flow_mismatching.utils import enable_tf32, load_ul_components


class _FrozenULCheckpointEncoder:
    """Adapter wrapping a Stage-1 UL checkpoint's frozen encoder E_psi for offline Stage-2 latent caching."""

    def __init__(self, ckpt_path: str, dataset: str, device: torch.device, dtype: torch.dtype):
        _, encoder, _, meta = load_ul_components(ckpt_path, device=device, load_decoder=False)
        self.encoder = encoder.eval().to(device=device)
        self.encoder.requires_grad_(False)
        self.ckpt_path = str(Path(ckpt_path).resolve())
        self.model_id = self.ckpt_path
        self.input_size = int(meta.get("img_size", 128))
        self.mean = meta.get("norm_mean") or NORM_STATS[dataset][0]
        self.std = meta.get("norm_std") or NORM_STATS[dataset][1]
        sample_shape = meta["sample_shape"]
        self.latent_channels = int(sample_shape[0])
        self.latent_size = int(sample_shape[1])
        self.ul_config = meta.get("ul_config")
        self.stage1_encoder_config = meta.get("encoder_config")
        self.dtype = dtype
        self.device = device

    def preprocess_transform(self):
        return image_transform(self.input_size, self.mean, self.std)

    @torch.inference_mode()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        use_amp = self.device.type == "cuda" and self.dtype in (torch.float16, torch.bfloat16)
        with torch.autocast(self.device.type, dtype=self.dtype, enabled=use_amp):
            return self.encoder(x.to(self.device))

    def config_dict(self):
        return {
            "type": "ul_ckpt",
            "ckpt_path": self.ckpt_path,
            "input_size": self.input_size,
            "latent_channels": self.latent_channels,
            "latent_size": self.latent_size,
            "stage1_encoder_config": self.stage1_encoder_config,
            "ul_config": self.ul_config,
        }


class _ImageListDataset(Dataset):
    def __init__(self, paths: List[Path], transform):
        self.paths = paths
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.transform(Image.open(self.paths[idx]).convert("RGB"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--encoder", required=True, choices=["sdvae", "dinov2", "ul_ckpt"])
    parser.add_argument("--ul_ckpt", default=None, help="Stage-1 UL checkpoint path (required when --encoder ul_ckpt)")
    parser.add_argument("--dataset", default="mvtec", choices=sorted(NORM_STATS), help="dataset normalization fallback")
    parser.add_argument("--model_id", default=None, help="override default HF model id")
    parser.add_argument("--input_size", type=int, default=None, help="override default encoder input resolution")
    parser.add_argument("--data_root", required=True, help="dataset root in MVTec-AD folder layout")
    parser.add_argument("--output_dir", required=True, help="directory to store fp16 cached latents + meta.json")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--classes", nargs="+", default=None, help="optional subset of classes")
    return parser.parse_args()


@torch.inference_mode()
def encode_paths(encoder, paths: List[Path], batch_size: int, num_workers: int, device: torch.device, desc: str) -> torch.Tensor:
    ds = _ImageListDataset(paths, encoder.preprocess_transform())
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=device.type == "cuda")
    chunks = []
    for batch in tqdm(loader, desc=desc, leave=False):
        z = encoder.encode(batch.to(device, non_blocking=True))
        chunks.append(z.to(dtype=torch.float16).cpu())
    return torch.cat(chunks, dim=0)


def collect_test_items(data_root: Path, class_name: str) -> List[Tuple[Path, str, int, str]]:
    """Return list of (img_path, mask_path_or_empty, label, item_name)."""
    class_dir = data_root / class_name
    test_dir = class_dir / "test"
    items = []
    for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
        defect = defect_dir.name
        for img_path in list_images(defect_dir):
            if defect == "good":
                mask_str, label = "", 0
            else:
                mask_path = find_mask(class_dir / "ground_truth" / defect, img_path.stem)
                if mask_path is None:
                    raise FileNotFoundError(f"No ground-truth mask for {img_path}")
                mask_str, label = str(mask_path), 1
            items.append((img_path, mask_str, label, f"{defect}_{img_path.stem}"))
    return items


def main():
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda":
        enable_tf32()

    t_start = time.time()
    data_root = Path(args.data_root).resolve()
    out_dir = Path(args.output_dir).resolve()
    (out_dir / "test").mkdir(parents=True, exist_ok=True)

    class_names = discover_classes(str(data_root), "train")
    if args.classes:
        class_names = [c for c in class_names if c in set(args.classes)]
        if not class_names:
            raise ValueError(f"None of {args.classes} found under {data_root}")

    if args.encoder == "ul_ckpt":
        if not args.ul_ckpt:
            raise SystemExit("--ul_ckpt is required when --encoder ul_ckpt")
        dtype = torch.bfloat16 if (device.type == "cuda" and torch.cuda.is_bf16_supported()) else torch.float32
        encoder = _FrozenULCheckpointEncoder(args.ul_ckpt, dataset=args.dataset, device=device, dtype=dtype)
    else:
        dtype = torch.float16 if device.type == "cuda" else torch.float32
        encoder = build_encoder(args.encoder, device=device, model_id=args.model_id, input_size=args.input_size, dtype=dtype)
    print(f"Loaded {args.encoder} ({encoder.model_id}), input={encoder.input_size}, latent={encoder.latent_channels}x{encoder.latent_size}x{encoder.latent_size}")

    # 1. Encode all normal training images across all classes.
    train_paths: List[Path] = []
    train_labels: List[int] = []
    for idx, cname in enumerate(class_names):
        files = list_images(data_root / cname / "train" / "good")
        train_paths.extend(files)
        train_labels.extend([idx] * len(files))

    train_latents = encode_paths(encoder, train_paths, args.batch_size, args.num_workers, device, desc="Encoding train")
    standardizer = Standardizer.fit(train_latents)
    torch.save(
        {
            "latents": train_latents,  # fp16 [N, C, H, W]
            "labels": torch.tensor(train_labels, dtype=torch.long),
            "paths": [str(p) for p in train_paths],
            "class_names": class_names,
        },
        out_dir / "train_latents.pt",
    )

    # 2. Encode test images per class.
    total_test = 0
    for cname in tqdm(class_names, desc="Test classes"):
        items = collect_test_items(data_root, cname)
        img_paths = [it[0] for it in items]
        test_latents = encode_paths(encoder, img_paths, args.batch_size, args.num_workers, device, desc=f"test/{cname}")
        total_test += len(items)
        torch.save(
            {
                "latents": test_latents,  # fp16 [M, C, H, W]
                "labels": torch.tensor([it[2] for it in items], dtype=torch.long),
                "names": [it[3] for it in items],
                "img_paths": [str(it[0]) for it in items],
                "mask_paths": [it[1] for it in items],
            },
            out_dir / "test" / f"{cname}.pt",
        )

    meta = {
        "data_root": str(data_root),
        "class_names": class_names,
        "num_train": len(train_paths),
        "num_test": total_test,
        "latent_shape": list(train_latents.shape[1:]),
        "encoder_config": encoder.config_dict(),
        "latent_stats": standardizer.to_dict(),
    }
    with open(out_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_start
    total_mb = sum(p.stat().st_size for p in out_dir.rglob("*.pt")) / (1024 * 1024)
    print(f"Cached {len(train_paths)} train + {total_test} test latents ({total_mb:.1f} MB) in {elapsed:.1f}s -> {out_dir}")


if __name__ == "__main__":
    main()
