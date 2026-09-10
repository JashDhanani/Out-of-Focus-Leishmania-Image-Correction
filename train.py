#!/usr/bin/env python
"""Trains the CycleGAN-based out-of-focus correction model (Section 3 of the paper).

Usage:
    python train.py --config configs/leishmania.yaml
    python train.py --config configs/bpaec.yaml
"""
import argparse
from pathlib import Path

import itertools
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from comi.datasets import build_dataset
from comi.losses import GANLoss, ContentLoss, CycleLoss
from comi.models import ResnetGenerator, PatchDiscriminator
from comi.utils import (
    set_seed, load_config, infinite_loader, linear_decay_lambda,
    save_sample_grid, save_checkpoint, load_checkpoint,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--resume", default=None, help="path to a checkpoint to resume from")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)
    set_seed(cfg["train"].get("seed", 42))
    device = torch.device(args.device)
    if device.type == "cuda":
        # crop size is fixed per run, so let cuDNN pick the fastest conv
        # algorithms for these shapes instead of the generic default
        torch.backends.cudnn.benchmark = True

    out_dir = Path(cfg["train"]["output_dir"])
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    (out_dir / "samples").mkdir(parents=True, exist_ok=True)

    train_set = build_dataset(cfg, split="train")
    loader = DataLoader(
        train_set, batch_size=cfg["train"]["batch_size"], shuffle=True,
        num_workers=cfg["train"]["num_workers"], drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    data_iter = infinite_loader(loader)
    steps_per_epoch = max(1, len(train_set) // cfg["train"]["batch_size"])
    total_iters = cfg["train"]["total_iters"]
    print(f"Loaded {len(train_set)} training pairs from {cfg['dataset']}")
    print(f"{steps_per_epoch} steps/epoch, {total_iters} total iters "
          f"(~{total_iters / steps_per_epoch:.1f} epochs)")

    m = cfg["model"]
    G_s = ResnetGenerator(base_channels=m["base_channels"], n_blocks=m["n_resblocks"]).to(device)
    G_t = ResnetGenerator(base_channels=m["base_channels"], n_blocks=m["n_resblocks"]).to(device)
    D_s = PatchDiscriminator(base_channels=m["base_channels"], n_layers=m["n_discriminator_layers"]).to(device)
    D_t = PatchDiscriminator(base_channels=m["base_channels"], n_layers=m["n_discriminator_layers"]).to(device)

    l = cfg["loss"]
    gan_loss = GANLoss()
    content_loss = ContentLoss(layer_index=l.get("vgg_layer_index")).to(device)
    cycle_loss = CycleLoss()

    t = cfg["train"]
    opt_G = torch.optim.Adam(
        itertools.chain(G_s.parameters(), G_t.parameters()),
        lr=t["lr"], betas=(t["beta1"], t["beta2"]),
    )
    opt_D = torch.optim.Adam(
        itertools.chain(D_s.parameters(), D_t.parameters()),
        lr=t["lr"], betas=(t["beta1"], t["beta2"]),
    )

    lr_fn = linear_decay_lambda(t["total_iters"], t["decay_start_iter"])
    sched_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_fn)
    sched_D = torch.optim.lr_scheduler.LambdaLR(opt_D, lr_fn)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(
            args.resume,
            models={"G_s": G_s, "G_t": G_t, "D_s": D_s, "D_t": D_t},
            optimizers={"G": opt_G, "D": opt_D},
            map_location=device,
        )
        print(f"Resumed from {args.resume} at step {start_step}")

    pbar = tqdm(range(start_step, total_iters), initial=start_step, total=total_iters)
    for step in pbar:
        if step % steps_per_epoch == 0:
            epoch = step // steps_per_epoch + 1
            total_epochs = (total_iters + steps_per_epoch - 1) // steps_per_epoch
            pbar.write(f"=== epoch {epoch}/{total_epochs} (step {step}/{total_iters}) ===")

        batch = next(data_iter)
        blur = batch["blur"].to(device, non_blocking=True)   # source domain (out-of-focus)
        sharp = batch["sharp"].to(device, non_blocking=True)  # target domain (in-focus)

        # ---------------- Generators ----------------
        opt_G.zero_grad(set_to_none=True)

        fake_sharp = G_s(blur)     # Gs: source -> target
        fake_blur = G_t(sharp)     # Gt: target -> source
        rec_blur = G_t(fake_sharp)   # cycle: Gt(Gs(blur)) ~= blur
        rec_sharp = G_s(fake_blur)   # cycle: Gs(Gt(sharp)) ~= sharp

        loss_gan = (
            gan_loss(D_t(fake_sharp), is_real=True)
            + gan_loss(D_s(fake_blur), is_real=True)
        )
        loss_content = content_loss(fake_sharp, sharp) + content_loss(fake_blur, blur)
        loss_cycle = cycle_loss(rec_blur, blur) + cycle_loss(rec_sharp, sharp)

        loss_G = l["lambda_gan"] * loss_gan + l["lambda_content"] * loss_content + l["lambda_cycle"] * loss_cycle
        loss_G.backward()
        opt_G.step()

        # ---------------- Discriminators ----------------
        opt_D.zero_grad(set_to_none=True)

        loss_D_t = 0.5 * (
            gan_loss(D_t(sharp), is_real=True)
            + gan_loss(D_t(fake_sharp.detach()), is_real=False)
        )
        loss_D_s = 0.5 * (
            gan_loss(D_s(blur), is_real=True)
            + gan_loss(D_s(fake_blur.detach()), is_real=False)
        )
        loss_D = loss_D_t + loss_D_s
        loss_D.backward()
        opt_D.step()

        sched_G.step()
        sched_D.step()

        if step % t["log_every"] == 0:
            epoch = step // steps_per_epoch + 1
            total_epochs = (total_iters + steps_per_epoch - 1) // steps_per_epoch
            lr = sched_G.get_last_lr()[0]
            mem = f", mem {torch.cuda.memory_allocated() / 1e9:.2f}GB" if device.type == "cuda" else ""
            pbar.set_description(
                f"epoch {epoch}/{total_epochs} step {step}/{total_iters} lr {lr:.2e}{mem} | "
                f"G {loss_G.item():.3f} (gan {loss_gan.item():.3f} "
                f"cont {loss_content.item():.3f} cyc {loss_cycle.item():.3f}) "
                f"D {loss_D.item():.3f}"
            )

        if step % t["sample_every"] == 0:
            save_sample_grid(
                str(out_dir / "samples" / f"step_{step:07d}.png"),
                [blur[:1], fake_sharp[:1].detach(), sharp[:1]],
                nrow=3,
            )

        if step > 0 and step % t["checkpoint_every"] == 0:
            save_checkpoint(
                str(out_dir / "checkpoints" / f"step_{step:07d}.pt"),
                step,
                models={"G_s": G_s, "G_t": G_t, "D_s": D_s, "D_t": D_t},
                optimizers={"G": opt_G, "D": opt_D},
            )

    save_checkpoint(
        str(out_dir / "checkpoints" / "final.pt"),
        total_iters,
        models={"G_s": G_s, "G_t": G_t, "D_s": D_s, "D_t": D_t},
        optimizers={"G": opt_G, "D": opt_D},
    )
    print(f"Training complete. Final checkpoint saved to {out_dir / 'checkpoints' / 'final.pt'}")


if __name__ == "__main__":
    main()
