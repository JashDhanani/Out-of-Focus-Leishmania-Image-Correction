#!/usr/bin/env python
"""Runs a trained Gs generator over one image or a folder of images, with no
ground truth required. Used e.g. to reproduce the BBBC006 generalization
experiment (Section 4.3): feed out-of-focus images straight into a model
trained on BPAEC, no further training.

Usage:
    python inference.py --checkpoint train_outputs/bpaec_nucleus_z004/checkpoints/final.pt \\
        --input path/to/image_or_dir --output results/bbbc006_corrected
"""
import argparse
from pathlib import Path

import torch
import torchvision.transforms.functional as TF
from PIL import Image
from tqdm import tqdm

from comi.models import ResnetGenerator
from comi.utils import load_checkpoint

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--input", required=True, help="a single image, or a directory of images")
    p.add_argument("--output", required=True)
    p.add_argument("--size", type=int, default=None, help="optional resize before inference")
    p.add_argument("--base-channels", type=int, default=64)
    p.add_argument("--n-resblocks", type=int, default=9)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def load_image(path: Path, size: int | None) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    if size:
        img = TF.resize(img, [size, size])
    tensor = TF.to_tensor(img) * 2.0 - 1.0
    return tensor.unsqueeze(0)


def main():
    args = parse_args()
    device = torch.device(args.device)

    input_path = Path(args.input)
    if input_path.is_dir():
        paths = sorted(p for p in input_path.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    else:
        paths = [input_path]
    if not paths:
        raise FileNotFoundError(f"No images found at {input_path}")

    G_s = ResnetGenerator(base_channels=args.base_channels, n_blocks=args.n_resblocks).to(device)
    load_checkpoint(args.checkpoint, models={"G_s": G_s}, map_location=device)
    G_s.eval()

    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        for path in tqdm(paths, desc="correcting"):
            blur = load_image(path, args.size).to(device)
            corrected = G_s(blur)[0]
            corrected = ((corrected.clamp(-1, 1) + 1) / 2 * 255).round().byte()
            corrected = corrected.permute(1, 2, 0).cpu().numpy()
            Image.fromarray(corrected).save(out_dir / f"{path.stem}_corrected.png")


if __name__ == "__main__":
    main()
