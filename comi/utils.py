import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torchvision.utils import save_image


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def infinite_loader(loader):
    while True:
        for batch in loader:
            yield batch


def linear_decay_lambda(total_iters: int, decay_start: int):
    """1.0 until `decay_start`, then linearly decays to 0 at `total_iters`."""

    def fn(step: int) -> float:
        if step < decay_start:
            return 1.0
        return max(0.0, 1.0 - (step - decay_start) / max(1, total_iters - decay_start))

    return fn


def save_sample_grid(path: str, tensors, nrow: int = 3):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    grid = torch.cat(tensors, dim=0)
    save_image((grid + 1.0) / 2.0, path, nrow=nrow)


def save_checkpoint(path: str, step: int, models: dict, optimizers: dict):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    state = {"step": step}
    state.update({f"model_{k}": v.state_dict() for k, v in models.items()})
    state.update({f"optim_{k}": v.state_dict() for k, v in optimizers.items()})
    torch.save(state, path)


def load_checkpoint(path: str, models: dict, optimizers: dict | None = None, map_location="cpu") -> int:
    state = torch.load(path, map_location=map_location)
    for k, model in models.items():
        model.load_state_dict(state[f"model_{k}"])
    if optimizers:
        for k, optim in optimizers.items():
            optim.load_state_dict(state[f"optim_{k}"])
    return state.get("step", 0)
