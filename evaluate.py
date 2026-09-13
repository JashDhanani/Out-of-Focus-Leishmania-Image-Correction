#!/usr/bin/env python
"""Evaluates a trained Gs (source/blur -> target/sharp) on the test split,
reporting PSNR / SSIM / PCC (Eq. 1-3), matching Table 2 / Table 3.

Usage:
    python evaluate.py --config configs/leishmania.yaml --checkpoint train_outputs/leishmania/checkpoints/final.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from comi.datasets import build_dataset
from comi.metrics import compute_metrics
from comi.models import ResnetGenerator
from comi.utils import load_config, load_checkpoint


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--save-images", default=None, help="optional directory to dump corrected images")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device(args.device)

    test_set = build_dataset(cfg, split="test")
    loader = DataLoader(test_set, batch_size=1, shuffle=False, num_workers=2)

    m = cfg["model"]
    G_s = ResnetGenerator(base_channels=m["base_channels"], n_blocks=m["n_resblocks"]).to(device)
    load_checkpoint(args.checkpoint, models={"G_s": G_s}, map_location=device)
    G_s.eval()

    if args.save_images:
        Path(args.save_images).mkdir(parents=True, exist_ok=True)

    all_metrics = []
    with torch.no_grad():
        for batch in tqdm(loader, desc="evaluating"):
            blur = batch["blur"].to(device)
            sharp = batch["sharp"].to(device)
            fake_sharp = G_s(blur)

            metrics = compute_metrics(fake_sharp[0], sharp[0])
            metrics["name"] = batch["name"][0]
            all_metrics.append(metrics)

            if args.save_images:
                save_image((fake_sharp[0] + 1) / 2, Path(args.save_images) / f"{batch['name'][0]}_corrected.png")

    psnr = np.mean([r["psnr"] for r in all_metrics])
    ssim = np.mean([r["ssim"] for r in all_metrics])
    pcc = np.mean([r["pcc"] for r in all_metrics])
    psnr_std = np.std([r["psnr"] for r in all_metrics])
    ssim_std = np.std([r["ssim"] for r in all_metrics])
    pcc_std = np.std([r["pcc"] for r in all_metrics])

    print(f"\nN = {len(all_metrics)}")
    print(f"PSNR: {psnr:.2f} ± {psnr_std:.2f}")
    print(f"SSIM: {ssim:.4f} ± {ssim_std:.4f}")
    print(f"PCC:  {pcc:.4f} ± {pcc_std:.4f}")


if __name__ == "__main__":
    #main function calling
    main()
