#!/usr/bin/env python
"""Extracts just the Gs generator weights from a full training checkpoint,
for lightweight deployment (web demo / inference-only use). A training
checkpoint bundles G_s, G_t, D_s, D_t and both optimizers' state (needed to
resume training); none of that except G_s is needed to run corrections.

Usage:
    python scripts/export_generator.py \\
        --checkpoint train_outputs/leishmania/checkpoints/final.pt \\
        --output webapp/checkpoints/leishmania_generator.pt --fp16
"""
import argparse
from pathlib import Path

import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--fp16", action="store_true", help="halve file size by storing weights as float16")
    return p.parse_args()


def main():
    args = parse_args()
    state = torch.load(args.checkpoint, map_location="cpu")
    g_s_state = state["model_G_s"]
    if args.fp16:
        g_s_state = {k: v.half() for k, v in g_s_state.items()}

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(g_s_state, out_path)

    before = Path(args.checkpoint).stat().st_size / 1e6
    after = out_path.stat().st_size / 1e6
    print(f"{args.checkpoint} ({before:.1f} MB) -> {out_path} ({after:.1f} MB)")


if __name__ == "__main__":
    main()
