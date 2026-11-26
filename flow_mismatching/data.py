"""Datasets in the MVTec-AD folder layout (VisA is converted to the same layout).

    <data_root>/<class>/train/good/*.png
    <data_root>/<class>/test/<good | defect type>/*.png
    <data_root>/<class>/ground_truth/<defect type>/<image stem>_mask.png   (MVTec-AD)
    <data_root>/<class>/ground_truth/<defect type>/<image stem>.png        (VisA)
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import functional as TF

IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

# Per-channel mean and std of the normal training images.
NORM_STATS = {
    "mvtec": ((0.431892, 0.401233, 0.391279), (0.254873, 0.251537, 0.248335)),
    "visa": ((0.320139, 0.389447, 0.311646), (0.278132, 0.229037, 0.216387)),
    "mvtec_loco": ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def image_transform(img_size: int, mean: Sequence[float], std: Sequence[float]) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )


def list_images(folder: Path) -> List[Path]:
    return [p for p in sorted(folder.rglob("*")) if p.suffix.lower() in IMG_EXTENSIONS]


def discover_classes(data_root: str, split: str) -> List[str]:
    """Sorted names of the class folders under data_root that contain a ``split`` folder."""
    root = Path(data_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset root not found: {root}")
    classes = sorted(p.name for p in root.iterdir() if (p / split).is_dir())
    if not classes:
        raise RuntimeError(f"No class folders with a '{split}' sub-folder found under {root}")
    return classes


class TrainDataset(Dataset):
    """Normal training images of all classes. Returns (image, class index)."""

    def __init__(
        self,
        data_root: str,
        class_names: Sequence[str],
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        self.class_names = list(class_names)
        self.files: List[Path] = []
        self.labels: List[int] = []
        for idx, name in enumerate(self.class_names):
            files = list_images(Path(data_root) / name / "train" / "good")
            self.files += files
            self.labels += [idx] * len(files)
        if not self.files:
            raise RuntimeError(f"No training images found under {data_root}/<class>/train/good")
        self.transform = image_transform(img_size, mean, std)

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        image = Image.open(self.files[idx]).convert("RGB")
        return self.transform(image), self.labels[idx]


def find_mask(mask_dir: Path, stem: str) -> Optional[Path]:
    """Ground-truth mask of an image:
    - <stem>_mask.* (MVTec-AD)
    - <stem>.* file (VisA)
    - <stem>/ directory containing one or more binary mask PNGs (MVTec-LOCO)
    """
    if not mask_dir.is_dir():
        return None
    for pattern in (f"{stem}_mask.*", f"{stem}.*"):
        candidates = [p for p in sorted(mask_dir.glob(pattern)) if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS]
        if candidates:
            return candidates[0]
    stem_dir = mask_dir / stem
    if stem_dir.is_dir() and list_images(stem_dir):
        return stem_dir
    for p in sorted(mask_dir.glob(f"{stem}*")):
        if p.is_file() and p.suffix.lower() in IMG_EXTENSIONS:
            return p
    return None


def load_binary_mask(mask_path: Optional[Path], mask_size: int, mask_resize) -> torch.Tensor:
    """Load binary mask tensor [1, mask_size, mask_size]; unions all PNGs if mask_path is a directory (MVTec-LOCO)."""
    if mask_path is None:
        return torch.zeros(1, mask_size, mask_size)
    if mask_path.is_dir():
        mask = torch.zeros(1, mask_size, mask_size)
        for p in list_images(mask_path):
            m = TF.to_tensor(mask_resize(Image.open(p).convert("L")))
            mask = torch.maximum(mask, (m > 0.5).float())
        return mask
    m = TF.to_tensor(mask_resize(Image.open(mask_path).convert("L")))
    return (m > 0.5).float()


class TestDataset(Dataset):
    """Test images of one class. Returns (image, mask [1, H, W], label, name).

    label is 0 for images in test/good and 1 otherwise; name is "<defect type>_<image stem>".
    """

    def __init__(
        self,
        data_root: str,
        class_name: str,
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
    ):
        class_dir = Path(data_root) / class_name
        test_dir = class_dir / "test"
        if not test_dir.is_dir():
            raise FileNotFoundError(f"Missing test folder: {test_dir}")

        self.items: List[Tuple[Path, Optional[Path], int, str]] = []
        for defect_dir in sorted(p for p in test_dir.iterdir() if p.is_dir()):
            defect = defect_dir.name
            for img_path in list_images(defect_dir):
                if defect == "good":
                    mask_path, label = None, 0
                else:
                    mask_path, label = find_mask(class_dir / "ground_truth" / defect, img_path.stem), 1
                    if mask_path is None:
                        raise FileNotFoundError(f"No ground-truth mask for {img_path}")
                self.items.append((img_path, mask_path, label, f"{defect}_{img_path.stem}"))
        if not self.items:
            raise RuntimeError(f"No test images found under {test_dir}")

        self.img_size = img_size
        self.transform = image_transform(img_size, mean, std)
        self.mask_resize = transforms.Resize(
            (img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        img_path, mask_path, label, name = self.items[idx]
        image = self.transform(Image.open(img_path).convert("RGB"))
        mask = load_binary_mask(mask_path, self.img_size, self.mask_resize)
        return image, mask, label, name


def load_latent_cache_meta(cache_dir: str) -> Dict:
    """Load meta.json produced by scripts/precompute_latents.py."""
    meta_path = Path(cache_dir) / "meta.json"
    if not meta_path.is_file():
        raise FileNotFoundError(f"Missing latent cache metadata: {meta_path}")
    with open(meta_path) as f:
        return json.load(f)


class LatentTrainDataset(Dataset):
    """Cached normal training latents from scripts/precompute_latents.py.

    Returns standardized float32 (latent [C, H, W], class_idx).
    """

    def __init__(self, cache_dir: str, standardizer=None):
        cache_path = Path(cache_dir)
        data = torch.load(cache_path / "train_latents.pt", map_location="cpu", weights_only=True)
        self.latents: torch.Tensor = data["latents"]  # fp16 [N, C, H, W]
        self.labels: torch.Tensor = data["labels"]
        self.class_names: List[str] = list(data["class_names"])
        self.standardizer = standardizer

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        z = self.latents[idx].float()
        if self.standardizer is not None:
            z = self.standardizer.normalize(z)
        return z, int(self.labels[idx])


class LatentTestDataset(Dataset):
    """Cached test latents of one class. Returns (latent [C, h, w], mask [1, M, M], label, name, img_path)."""

    def __init__(self, cache_dir: str, class_name: str, mask_size: int, standardizer=None):
        test_path = Path(cache_dir) / "test" / f"{class_name}.pt"
        if not test_path.is_file():
            raise FileNotFoundError(f"Missing cached test latents: {test_path}")
        data = torch.load(test_path, map_location="cpu", weights_only=True)
        self.latents: torch.Tensor = data["latents"]  # fp16 [M, C, h, w]
        self.labels: torch.Tensor = data["labels"]
        self.names: List[str] = list(data["names"])
        self.img_paths: List[str] = list(data["img_paths"])
        self.mask_paths: List[str] = list(data["mask_paths"])
        self.mask_size = int(mask_size)
        self.standardizer = standardizer
        self.mask_resize = transforms.Resize(
            (self.mask_size, self.mask_size), interpolation=transforms.InterpolationMode.NEAREST
        )

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int):
        z = self.latents[idx].float()
        if self.standardizer is not None:
            z = self.standardizer.normalize(z)
        mask_path = self.mask_paths[idx]
        mask = load_binary_mask(Path(mask_path) if mask_path else None, self.mask_size, self.mask_resize)
        return z, mask, int(self.labels[idx]), self.names[idx], self.img_paths[idx]


class PairedLatentImageDataset(Dataset):
    """Cached frozen features (e.g., DINOv2) paired with normalized RGB targets for UL adapter training.

    Returns (feat_std [C_in, h, w], image [3, img_size, img_size], class_idx).
    """

    def __init__(
        self,
        cache_dir: str,
        img_size: int,
        mean: Sequence[float],
        std: Sequence[float],
        standardizer=None,
    ):
        cache_path = Path(cache_dir)
        data = torch.load(cache_path / "train_latents.pt", map_location="cpu", weights_only=True)
        self.latents: torch.Tensor = data["latents"]  # fp16 [N, C_in, h, w]
        self.labels: torch.Tensor = data["labels"]
        self.paths: List[str] = list(data["paths"])
        self.class_names: List[str] = list(data["class_names"])
        self.standardizer = standardizer
        self.transform = image_transform(img_size, mean, std)

    def __len__(self) -> int:
        return self.latents.shape[0]

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, int]:
        feat = self.latents[idx].float()
        if self.standardizer is not None:
            feat = self.standardizer.normalize(feat)
        image = self.transform(Image.open(self.paths[idx]).convert("RGB"))
        return feat, image, int(self.labels[idx])
