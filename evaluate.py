"""Evaluate Flow Mismatching on MVTec-AD or VisA (pixel, frozen VAE, or Unified Latents).

For Unified Latents (--space ul checkpoints), the train-only diffusion decoder is DISCARDED
at test time by default; anomaly scoring runs directly on the UL flow prior v_theta(E_psi(y)).

Examples:
    python evaluate.py --dataset mvtec --data_root ../data/mvtec --ckpt runs/mvtec/ckpt_epoch2000.pt
    python evaluate.py --dataset mvtec --latent_cache ../cache/mvtec_sdvae --ckpt runs/mvtec_vae/ckpt_epoch1000.pt
    python evaluate.py --dataset mvtec --data_root ../data/mvtec --ckpt runs/B1_ul/ckpt_epoch1000.pt --aggregation min
    python evaluate.py --dataset mvtec --data_root ../data/mvtec --ckpt runs/B1_ul/ckpt_epoch1000.pt --aggregation mean

Writes to --output_dir:
    per_image.csv           image-level score and pixel metrics of every test image
    metrics_pooled.csv      per-class metrics, pixels pooled over each class, and their mean
    metrics_per_image.csv   per-class metrics, pixel metrics averaged over anomalous images
    config.json             the arguments of this run
    latency.json            wall-clock scoring latency diagnostics (ms/image)
    ul_diagnostics.json     normal vs. anomalous UL bitrate (and optional OOD recon PSNR diagnostics)
    vis/<class>/*.png       image | heatmap | ground truth (with --save_vis)
"""

import argparse
import csv
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from flow_mismatching.data import (
    NORM_STATS,
    LatentTestDataset,
    TestDataset,
    discover_classes,
    image_transform,
    load_latent_cache_meta,
)
from flow_mismatching.encoders import Standardizer, build_encoder
from flow_mismatching.metrics import PIXEL_METRICS, class_summary, pixel_metrics
from flow_mismatching.scoring import anomaly_heatmap, gaussian_smooth, image_score
from flow_mismatching.ul import LN2, encode_with_fixed_noise, estimate_ul_bitrate_nats_per_dim
from flow_mismatching.utils import (
    denormalize,
    enable_tf32,
    load_model,
    load_ul_components,
    resolve_amp_dtype,
    save_visualization,
)

METRIC_COLUMNS = [
    "image_auroc", "image_ap", "image_f1_max", "pixel_auroc", "pixel_ap", "pixel_f1_max", "pixel_aupro",
]


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(NORM_STATS))
    parser.add_argument("--data_root", default=None, help="dataset root in the MVTec-AD folder layout")
    parser.add_argument("--latent_cache", default=None, help="optional precomputed latent cache dir (for vae ckpts)")
    parser.add_argument("--ckpt", required=True, help="checkpoint saved by train.py")
    parser.add_argument("--output_dir", default=None, help="default: results/<dataset>")
    parser.add_argument("--classes", nargs="+", default=None, help="evaluate only these classes")
    parser.add_argument("--img_size", type=int, default=128, help="heatmap & ground-truth evaluation resolution")

    fm = parser.add_argument_group("Flow Mismatching")
    fm.add_argument("--num_paths", type=int, default=10, help="K, noise paths per image")
    fm.add_argument("--num_steps", type=int, default=10, help="T, time steps per path")
    fm.add_argument("--t0", type=float, default=0.0, help="time steps lie strictly inside (t0, t1)")
    fm.add_argument("--t1", type=float, default=None,
                    help="upper time bound (default: 1.0 for pixel/vae, or UL sigma0-aligned t1 for ul)")
    fm.add_argument("--power", type=float, default=2.0, help="r in the time weight w(t) = t^r")
    fm.add_argument("--aggregation", choices=["min", "mean"], default="min", help="aggregation over the paths")
    fm.add_argument("--gauss_ksize", type=int, default=9, help="Gaussian smoothing kernel size; 1 disables it")
    fm.add_argument("--gauss_sigma", type=float, default=3.0, help="Gaussian smoothing sigma")
    fm.add_argument("--topk_ratio", type=float, default=0.01, help="fraction of top pixels in the image score")
    fm.add_argument("--lam", type=float, default=0.5, help="weight of the max pixel in the image score")

    ul = parser.add_argument_group("UL Diagnostics")
    ul.add_argument("--diag_bitrate", action=argparse.BooleanOptionalAction, default=True,
                    help="log UL prior ELBO bitrate (normal vs. anomalous) for UL checkpoints")
    ul.add_argument("--diag_recon", action="store_true",
                    help="load train-only UL decoder for OOD encoder-collapse PSNR check (normal vs. anomalous patches)")
    ul.add_argument("--diag_recon_steps", type=int, default=20, help="ODE steps when --diag_recon is enabled")
    ul.add_argument("--diag_recon_max_per_class", type=int, default=20,
                    help="max images per class to reconstruct when --diag_recon is enabled")

    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--amp", choices=["bf16", "fp16", "off"], default="bf16", help="mixed precision")
    parser.add_argument("--tf32", action=argparse.BooleanOptionalAction, default=True, help="allow TF32 on Ampere+")
    parser.add_argument("--seed", type=int, default=0, help="seed of the noise paths, reset for every class")
    parser.add_argument("--num_workers", type=int, default=2)
    parser.add_argument("--save_vis", action="store_true", help="save image | heatmap | ground truth figures")
    return parser.parse_args()


def _sync_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def evaluate_class(
    model,
    space,
    class_name,
    class_idx,
    args,
    t1_eff,
    mean,
    std,
    device,
    amp_dtype,
    out_dir,
    online_encoder=None,
    ul_encoder=None,
    ul_decoder=None,
    ul_sigma0=None,
    standardizer=None,
):
    """Score every test image of one class. Returns (per-image rows, class summary, latencies_ms, ul_diag, loco_split)."""
    is_ul = space in ("ul", "ul_dino")
    use_cache = space in ("vae", "ul_dino", "ul_stage2") and args.latent_cache is not None
    if use_cache:
        dataset = LatentTestDataset(args.latent_cache, class_name, mask_size=args.img_size, standardizer=standardizer)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)
    elif space in ("vae", "ul_dino"):
        dataset = TestDataset(args.data_root, class_name, args.img_size, mean, std)
        dataset.transform = online_encoder.preprocess_transform()
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    elif space == "ul_stage2":
        dataset = TestDataset(args.data_root, class_name, args.img_size, mean, std)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)
    else:
        # Both pixel and UL-from-scratch consume NORM_STATS-normalized images at args.img_size.
        dataset = TestDataset(args.data_root, class_name, args.img_size, mean, std)
        loader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=args.num_workers)

    vis_transform = transforms.Compose(
        [
            transforms.Resize((args.img_size, args.img_size), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )
    norm_img_transform = image_transform(args.img_size, mean, std)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    use_amp = amp_dtype is not None and device.type == "cuda"

    rows, labels, scores, masks, heatmaps, per_image, latencies_ms, names_list = [], [], [], [], [], [], [], []
    ul_diag = {
        "normal_nats_per_dim": [],
        "anomalous_nats_per_dim": [],
        "recon_psnr_normal_img": [],
        "recon_psnr_anomalous_img": [],
        "recon_mse_normal_px": [],
        "recon_mse_anomalous_px": [],
    }
    recon_count = 0

    for idx, batch in enumerate(tqdm(loader, desc=class_name, leave=False)):
        if use_cache:
            x_in, mask, label, name, img_path = batch
            x_in = x_in.to(device)
        else:
            x_in, mask, label, name = batch
            x_in = x_in.to(device)
            img_path = None

        _sync_cuda(device)
        t0 = time.perf_counter()

        if space == "vae" and not use_cache:
            z = online_encoder.encode(x_in).float()
            if standardizer is not None:
                z = standardizer.normalize(z)
            score_input = z
        elif space == "ul_dino":
            if use_cache:
                feat_std = x_in
            else:
                feat_std = online_encoder.encode(x_in).float()
                if standardizer is not None:
                    feat_std = standardizer.normalize(feat_std)
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                score_input = ul_encoder(feat_std).float()
        elif space == "ul_stage2" and not use_cache:
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                z = ul_encoder(x_in).float()
            if standardizer is not None:
                z = standardizer.normalize(z)
            score_input = z
        elif space == "ul":
            with torch.inference_mode(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                score_input = ul_encoder(x_in).float()
        else:
            score_input = x_in

        heatmap = anomaly_heatmap(
            model,
            score_input,
            num_paths=args.num_paths,
            num_steps=args.num_steps,
            power=args.power,
            aggregation=args.aggregation,
            class_idx=class_idx,
            t_range=(args.t0, t1_eff),
            amp_dtype=amp_dtype,
            generator=generator,
            output_size=(args.img_size, args.img_size),
        )
        heatmap = gaussian_smooth(heatmap, args.gauss_ksize, args.gauss_sigma)
        score = image_score(heatmap, args.topk_ratio, args.lam)

        _sync_cuda(device)
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if idx > 0:  # skip first image per class as warmup
            latencies_ms.append(dt_ms)

        lbl_int = int(label)
        if is_ul and args.diag_bitrate and ul_sigma0 is not None:
            nats_d = estimate_ul_bitrate_nats_per_dim(
                model,
                score_input,
                sigma0=ul_sigma0,
                num_paths=min(args.num_paths, 4),
                num_steps=min(args.num_steps, 8),
                class_idx=class_idx,
                amp_dtype=amp_dtype,
                generator=generator,
            )
            if lbl_int == 0:
                ul_diag["normal_nats_per_dim"].append(nats_d)
            else:
                ul_diag["anomalous_nats_per_dim"].append(nats_d)

        if is_ul and ul_decoder is not None and recon_count < args.diag_recon_max_per_class:
            recon_count += 1
            z0 = encode_with_fixed_noise(score_input, sigma0=ul_sigma0)
            extra = {"label": torch.tensor([class_idx], device=device, dtype=torch.long)} if class_idx is not None else {}
            if space == "ul":
                orig_norm = x_in[0].float()
            else:
                src_path = img_path[0] if use_cache else dataset.items[idx][0]
                orig_norm = norm_img_transform(Image.open(src_path).convert("RGB")).to(device)
            x_rec = ul_decoder.sample(
                z0, img_shape=(args.img_size, args.img_size), steps=args.diag_recon_steps, extra=extra, amp_dtype=amp_dtype
            )
            orig_rgb = denormalize(orig_norm, mean, std)
            rec_rgb = denormalize(x_rec[0].float(), mean, std)
            px_mse = (orig_rgb - rec_rgb).pow(2).mean(dim=0)  # [H, W]
            img_mse = float(px_mse.mean().clamp(min=1e-10))
            psnr = -10.0 * float(np.log10(img_mse))
            if lbl_int == 0:
                ul_diag["recon_psnr_normal_img"].append(psnr)
            else:
                ul_diag["recon_psnr_anomalous_img"].append(psnr)
                m_bool = mask[0, 0].to(device) > 0.5
                if m_bool.any():
                    ul_diag["recon_mse_anomalous_px"].append(float(px_mse[m_bool].mean()))
                if (~m_bool).any():
                    ul_diag["recon_mse_normal_px"].append(float(px_mse[~m_bool].mean()))

        heatmap_np = heatmap.cpu().numpy()
        mask_np = mask[0, 0].numpy().astype(np.uint8)
        metrics = pixel_metrics(mask_np, heatmap_np)

        labels.append(lbl_int)
        scores.append(score)
        masks.append(mask_np)
        heatmaps.append(heatmap_np)
        per_image.append(metrics)
        names_list.append(name[0])
        rows.append({"class": class_name, "name": name[0], "label": lbl_int, "score": score,
                     **{f"pixel_{k}": metrics[k] for k in PIXEL_METRICS}})

        if args.save_vis:
            if use_cache:
                vis_img = vis_transform(Image.open(img_path[0]).convert("RGB"))
            elif space in ("vae", "ul_dino"):
                vis_img = vis_transform(Image.open(dataset.items[idx][0]).convert("RGB"))
            else:
                vis_img = denormalize(x_in[0], mean, std)
            save_visualization(vis_img, heatmap, mask[0], out_dir / "vis" / class_name / f"{name[0]}.png")

    masks_arr = np.stack(masks)
    heatmaps_arr = np.stack(heatmaps)
    summary = class_summary(labels, scores, masks_arr, heatmaps_arr, per_image)

    # MVTec-LOCO structural vs. logical anomaly split summary when present.
    loco_split = {}
    has_struct = any(n.startswith("structural_anomalies_") for n in names_list)
    has_logic = any(n.startswith("logical_anomalies_") for n in names_list)
    if has_struct or has_logic:
        for split_name, prefix in (("structural", "structural_anomalies_"), ("logical", "logical_anomalies_")):
            keep = [i for i, (lbl, nm) in enumerate(zip(labels, names_list)) if lbl == 0 or nm.startswith(prefix)]
            if keep and any(labels[i] == 1 for i in keep):
                loco_split[split_name] = class_summary(
                    [labels[i] for i in keep],
                    [scores[i] for i in keep],
                    masks_arr[keep],
                    heatmaps_arr[keep],
                    [per_image[i] for i in keep],
                )["pooled"]

    return rows, summary, latencies_ms, ul_diag, loco_split


def write_csv(path, columns, rows):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: f"{v:.6f}" if isinstance(v, float) else v for k, v in row.items()})


def print_table(title, rows):
    header = f"{'class':<14}" + "".join(f"{c.replace('_f1_max', '_f1'):>13}" for c in METRIC_COLUMNS)
    print(f"\n{title}\n{header}\n{'-' * len(header)}")
    for row in rows:
        print(f"{row['class']:<14}" + "".join(f"{100 * row[c]:>13.1f}" for c in METRIC_COLUMNS))


def main():
    args = parse_args()
    device = torch.device(args.device)
    if args.tf32 and device.type == "cuda":
        enable_tf32()
    amp_dtype = resolve_amp_dtype(args.amp, device)
    out_dir = Path(args.output_dir or Path("results") / args.dataset)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Peek checkpoint space so UL checkpoints discard decoder by default.
    ckpt_peek = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    space = ckpt_peek.get("space", "pixel") if isinstance(ckpt_peek, dict) else "pixel"
    del ckpt_peek

    online_encoder = None
    ul_encoder = None
    ul_decoder = None
    ul_sigma0 = None
    standardizer = None

    is_ul = space in ("ul", "ul_dino")
    if is_ul:
        model, ul_encoder, ul_decoder, meta = load_ul_components(args.ckpt, device, load_decoder=args.diag_recon)
        ul_cfg = meta["ul_config"]
        ul_sigma0 = float(ul_cfg["sigma0"])
        t1_eff = float(args.t1) if args.t1 is not None else float(ul_cfg["t1"])
        if meta.get("latent_stats"):
            standardizer = Standardizer.from_dict(meta["latent_stats"])
    else:
        model, meta = load_model(args.ckpt, device)
        t1_eff = float(args.t1) if args.t1 is not None else 1.0
        standardizer = Standardizer.from_dict(meta["latent_stats"]) if meta.get("latent_stats") else None

    mean = meta.get("norm_mean") or NORM_STATS[args.dataset][0]
    std = meta.get("norm_std") or NORM_STATS[args.dataset][1]

    if space in ("pixel", "ul"):
        if not args.data_root:
            raise SystemExit(f"--data_root is required for space={space} checkpoints")
        train_size = meta.get("img_size") or meta.get("args", {}).get("img_size")
        if train_size is not None and int(train_size) != args.img_size:
            print(f"Warning: the model was trained at {train_size}x{train_size}, evaluating at {args.img_size}.")
        available = discover_classes(args.data_root, "test")
    else:
        # Both vae and ul_dino can evaluate from precomputed --latent_cache or online via frozen encoder.
        if args.latent_cache:
            cache_meta = load_latent_cache_meta(args.latent_cache)
            available = list(cache_meta["class_names"])
        else:
            if not args.data_root:
                raise SystemExit(f"Provide either --latent_cache or --data_root for space={space} checkpoints")
            if space == "ul_stage2":
                enc_cfg = meta.get("encoder_config") or {}
                stage1_ckpt = enc_cfg.get("ckpt_path")
                if not stage1_ckpt:
                    raise ValueError("ul_stage2 checkpoint missing encoder_config.ckpt_path; pass --latent_cache")
                _, ul_encoder, _, _ = load_ul_components(stage1_ckpt, device, load_decoder=False)
            else:
                enc_cfg = meta.get("dino_encoder_config") if space == "ul_dino" else meta.get("encoder_config")
                if not enc_cfg:
                    raise ValueError("Checkpoint is missing frozen encoder config; pass --latent_cache or re-save checkpoint")
                online_encoder = build_encoder(
                    enc_cfg["type"],
                    device=device,
                    model_id=enc_cfg.get("model_id"),
                    input_size=enc_cfg.get("input_size"),
                    dtype=torch.float16 if device.type == "cuda" else torch.float32,
                )
            available = discover_classes(args.data_root, "test")

    # Class labels must follow the order used in training, which is stored in the checkpoint.
    num_classes = meta["num_classes"]
    if num_classes is None:
        class_to_idx = None
        classes = available
    else:
        class_names = meta.get("class_names") or available
        if len(class_names) != num_classes:
            raise ValueError(f"The checkpoint has {num_classes} classes but {len(class_names)} class names were found")
        class_to_idx = {c: i for i, c in enumerate(class_names)}
        classes = [c for c in class_names if c in available]
    if args.classes:
        unknown = sorted(set(args.classes) - set(classes))
        if unknown:
            raise ValueError(f"Unknown classes {unknown}; available: {classes}")
        classes = args.classes

    print(
        f"Loaded {args.ckpt} (space={space}, t_range=({args.t0:.3f}, {t1_eff:.4f})"
        + (f", decoder={'loaded (diag)' if ul_decoder is not None else 'discarded'}" if is_ul else "")
        + (f", epoch {meta['epoch']})" if "epoch" in meta else ")")
    )
    print(f"Evaluating {len(classes)} classes on {device}: {', '.join(classes)}")
    with open(out_dir / "config.json", "w") as f:
        json.dump({**vars(args), "space": space, "t1_effective": t1_eff, "ckpt_epoch": meta.get("epoch")}, f, indent=2)

    all_rows, summaries, all_latencies_ms, loco_splits = [], {}, [], {}
    agg_ul_diag = {
        "normal_nats_per_dim": [],
        "anomalous_nats_per_dim": [],
        "recon_psnr_normal_img": [],
        "recon_psnr_anomalous_img": [],
        "recon_mse_normal_px": [],
        "recon_mse_anomalous_px": [],
    }

    for class_name in classes:
        class_idx = class_to_idx[class_name] if class_to_idx else None
        rows, summary, lat_ms, ul_diag, loco_split = evaluate_class(
            model,
            space,
            class_name,
            class_idx,
            args,
            t1_eff,
            mean,
            std,
            device,
            amp_dtype,
            out_dir,
            online_encoder=online_encoder,
            ul_encoder=ul_encoder,
            ul_decoder=ul_decoder,
            ul_sigma0=ul_sigma0,
            standardizer=standardizer,
        )
        all_rows += rows
        summaries[class_name] = summary
        if loco_split:
            loco_splits[class_name] = loco_split
        all_latencies_ms.extend(lat_ms)
        for k, vals in ul_diag.items():
            agg_ul_diag[k].extend(vals)
        s = summary["pooled"]
        print(f"{class_name:<14} image AUROC {100 * s['image_auroc']:5.1f} | pixel AUROC "
              f"{100 * s['pixel_auroc']:5.1f} | pixel AP {100 * s['pixel_ap']:5.1f} | AUPRO {100 * s['pixel_aupro']:5.1f}")

    write_csv(out_dir / "per_image.csv", ["class", "name", "label", "score"] + [f"pixel_{k}" for k in PIXEL_METRICS],
              all_rows)
    for protocol, title in (("pooled", "pixels pooled per class"), ("per_image", "pixel metrics averaged over anomalous images")):
        rows = [{"class": c, **s[protocol]} for c, s in summaries.items()]
        rows.append({"class": "mean", **{k: float(np.mean([r[k] for r in rows])) for k in METRIC_COLUMNS}})
        write_csv(out_dir / f"metrics_{protocol}.csv", ["class"] + METRIC_COLUMNS, rows)
        print_table(f"Results (%), {title}:", rows)

    if loco_splits:
        loco_out = {"per_class": loco_splits, "mean": {}}
        for split_name in ("structural", "logical"):
            split_vals = [v[split_name] for v in loco_splits.values() if split_name in v]
            if split_vals:
                loco_out["mean"][split_name] = {k: float(np.mean([sv[k] for sv in split_vals])) for k in METRIC_COLUMNS}
        with open(out_dir / "loco_splits.json", "w") as f:
            json.dump(loco_out, f, indent=2)
        for split_name, m in loco_out["mean"].items():
            print(
                f"MVTec-LOCO {split_name:<10} mean | image AUROC {100 * m['image_auroc']:5.1f} | "
                f"pixel AUROC {100 * m['pixel_auroc']:5.1f} | pixel AP {100 * m['pixel_ap']:5.1f} | "
                f"AUPRO {100 * m['pixel_aupro']:5.1f}"
            )

    if all_latencies_ms:
        lat_arr = np.asarray(all_latencies_ms, dtype=np.float64)
        latency_info = {
            "space": space,
            "num_timed_images": int(lat_arr.size),
            "mean_ms_per_image": float(lat_arr.mean()),
            "median_ms_per_image": float(np.median(lat_arr)),
            "p95_ms_per_image": float(np.percentile(lat_arr, 95)),
            "images_per_sec": float(1000.0 / lat_arr.mean()),
            "num_paths": args.num_paths,
            "num_steps": args.num_steps,
            "used_latent_cache": bool(space in ("vae", "ul_dino", "ul_stage2") and args.latent_cache is not None),
        }
        with open(out_dir / "latency.json", "w") as f:
            json.dump(latency_info, f, indent=2)
        print(
            f"\nLatency: {latency_info['mean_ms_per_image']:.2f} ms/image "
            f"(median {latency_info['median_ms_per_image']:.2f} ms, {latency_info['images_per_sec']:.1f} img/s)"
        )

    if is_ul and (agg_ul_diag["normal_nats_per_dim"] or agg_ul_diag["recon_psnr_normal_img"]):
        diag_summary = {"sigma0": ul_sigma0, "t1": t1_eff}
        if agg_ul_diag["normal_nats_per_dim"]:
            n_nats = float(np.mean(agg_ul_diag["normal_nats_per_dim"]))
            diag_summary["normal_nats_per_dim"] = n_nats
            diag_summary["normal_bpd"] = n_nats / LN2
        if agg_ul_diag["anomalous_nats_per_dim"]:
            a_nats = float(np.mean(agg_ul_diag["anomalous_nats_per_dim"]))
            diag_summary["anomalous_nats_per_dim"] = a_nats
            diag_summary["anomalous_bpd"] = a_nats / LN2
        if agg_ul_diag["recon_psnr_normal_img"]:
            diag_summary["recon_psnr_normal_db"] = float(np.mean(agg_ul_diag["recon_psnr_normal_img"]))
        if agg_ul_diag["recon_psnr_anomalous_img"]:
            diag_summary["recon_psnr_anomalous_db"] = float(np.mean(agg_ul_diag["recon_psnr_anomalous_img"]))
        if agg_ul_diag["recon_mse_normal_px"] and agg_ul_diag["recon_mse_anomalous_px"]:
            mse_n = float(np.mean(agg_ul_diag["recon_mse_normal_px"]))
            mse_a = float(np.mean(agg_ul_diag["recon_mse_anomalous_px"]))
            diag_summary["recon_mse_normal_px"] = mse_n
            diag_summary["recon_mse_anomalous_px"] = mse_a
            diag_summary["recon_mse_anom_to_norm_ratio"] = mse_a / max(mse_n, 1e-10)
        with open(out_dir / "ul_diagnostics.json", "w") as f:
            json.dump(diag_summary, f, indent=2)
        print(f"UL diagnostics: {json.dumps(diag_summary)}")

    print(f"Saved results to {out_dir}")


if __name__ == "__main__":
    main()
