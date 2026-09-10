import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms.functional as TF

BPAEC_LAYERS = ["Z004", "Z005", "Z006", "Z008", "Z009", "Z010"]
BPAEC_GT_LAYER = "Z007"


class PairedAugment:
    """Applies identical random flip/rotation/crop to a (blur, sharp) pair,
    then resizes to `size` and normalizes to [-1, 1] (Eq. 10 uses tanh-range
    generators). Covers the "rotation, flip, translation, and scale" data
    augmentation mentioned in Section 3.3.
    """

    def __init__(self, size: int, train: bool = True):
        self.size = size
        self.train = train

    def __call__(self, blur: Image.Image, sharp: Image.Image):
        # resize shorter side to a bit larger than target, then crop, so the
        # "translation/scale" augmentation has room to move
        load_size = int(self.size * 1.12) if self.train else self.size
        blur = TF.resize(blur, [load_size, load_size])
        sharp = TF.resize(sharp, [load_size, load_size])

        if self.train:
            i, j, h, w = self._random_crop_params(load_size, self.size)
            blur = TF.crop(blur, i, j, h, w)
            sharp = TF.crop(sharp, i, j, h, w)

            if random.random() < 0.5:
                blur, sharp = TF.hflip(blur), TF.hflip(sharp)
            if random.random() < 0.5:
                blur, sharp = TF.vflip(blur), TF.vflip(sharp)
            angle = random.choice([0, 90, 180, 270])
            if angle:
                blur, sharp = TF.rotate(blur, angle), TF.rotate(sharp, angle)
        else:
            blur = TF.center_crop(blur, [self.size, self.size])
            sharp = TF.center_crop(sharp, [self.size, self.size])

        blur = TF.to_tensor(blur) * 2.0 - 1.0
        sharp = TF.to_tensor(sharp) * 2.0 - 1.0
        return blur, sharp

    @staticmethod
    def _random_crop_params(load_size: int, crop_size: int):
        max_offset = load_size - crop_size
        i = random.randint(0, max_offset) if max_offset > 0 else 0
        j = random.randint(0, max_offset) if max_offset > 0 else 0
        return i, j, crop_size, crop_size


def _load_rgb(path: Path) -> Image.Image:
    return Image.open(path).convert("RGB")


class LeishmaniaDataset(Dataset):
    """Paired blur/sharp Leishmania dataset.

    Expects the directory layout of the Mendeley dataset (m3jxgb54c9), i.e.
    `Leishmania_blurred_{train,test}/` and `Leishmania_clear_{train,test}/`
    with matching filenames between the two.
    """

    def __init__(self, root: str, split: str = "train", size: int = 256):
        assert split in ("train", "test")
        root = Path(root)
        self.blur_dir = root / f"Leishmania_blurred_{split}"
        self.sharp_dir = root / f"Leishmania_clear_{split}"
        self.filenames = sorted(p.name for p in self.blur_dir.glob("*.jpg"))
        if not self.filenames:
            raise FileNotFoundError(f"No images found in {self.blur_dir}")
        missing = [f for f in self.filenames if not (self.sharp_dir / f).exists()]
        if missing:
            raise FileNotFoundError(f"{len(missing)} blur images have no matching sharp pair, e.g. {missing[0]}")

        self.augment = PairedAugment(size=size, train=(split == "train"))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        name = self.filenames[idx]
        blur = _load_rgb(self.blur_dir / name)
        sharp = _load_rgb(self.sharp_dir / name)
        blur, sharp = self.augment(blur, sharp)
        return {"blur": blur, "sharp": sharp, "name": name}


class BPAECDataset(Dataset):
    """Paired blur/sharp BPAEC confocal dataset for a single cell structure
    (nucleus, actin or mitochondria) AND a single defocus layer.

    Per Section 4.2, a separate model is trained for each (structure, layer)
    combination -- the unique mapping from that one blurred layer to the
    z=7 sharp layer -- which is why Table 3 reports distinct metrics per
    layer. To reproduce the full table, train once per layer in
    BPAEC_LAYERS for each structure.

    The raw dataset has no train/test split, so one is created here and
    cached to `split_file` for reproducibility (paper's 8:2 protocol).
    """

    def __init__(self, root: str, structure: str, layer: str, split: str = "train",
                 size: int = 512, split_file: str | None = None,
                 train_ratio: float = 0.8, seed: int = 42):
        assert split in ("train", "test")
        assert structure in ("nucleus", "actin", "mitochondria")
        assert layer in BPAEC_LAYERS
        self.root = Path(root) / structure
        self.structure = structure
        self.layer = layer

        gt_dir = self.root / BPAEC_GT_LAYER
        indices = sorted(int(p.stem.split("_")[-1]) for p in gt_dir.glob("*.jpg"))
        if not indices:
            raise FileNotFoundError(f"No ground-truth images found in {gt_dir}")

        split_file = Path(split_file) if split_file else self.root / "split.json"
        splits = self._load_or_create_split(split_file, indices, train_ratio, seed)
        self.indices = splits[split]

        self.pairs = [(idx, layer) for idx in self.indices]
        self.augment = PairedAugment(size=size, train=(split == "train"))

    @staticmethod
    def _load_or_create_split(split_file: Path, indices, train_ratio: float, seed: int):
        if split_file.exists():
            with open(split_file) as f:
                return json.load(f)
        rng = random.Random(seed)
        shuffled = list(indices)
        rng.shuffle(shuffled)
        n_train = int(round(len(shuffled) * train_ratio))
        splits = {"train": sorted(shuffled[:n_train]), "test": sorted(shuffled[n_train:])}
        split_file.parent.mkdir(parents=True, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump(splits, f, indent=2)
        return splits

    def _layer_prefix(self, layer: str) -> str:
        # e.g. "Z004" -> "Z4", "Z010" -> "Z10"
        return f"Z{int(layer[1:])}"

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        gt_idx, layer = self.pairs[idx]
        blur_path = self.root / layer / f"{self._layer_prefix(layer)}_{gt_idx}.jpg"
        sharp_path = self.root / BPAEC_GT_LAYER / f"Z7_{gt_idx}.jpg"
        blur = _load_rgb(blur_path)
        sharp = _load_rgb(sharp_path)
        blur, sharp = self.augment(blur, sharp)
        return {"blur": blur, "sharp": sharp, "name": f"{layer}_{gt_idx}", "layer": layer}


def build_dataset(config: dict, split: str) -> Dataset:
    dataset_cfg = config["dataset"]
    kind = dataset_cfg["type"]
    if kind == "leishmania":
        return LeishmaniaDataset(root=dataset_cfg["root"], split=split, size=dataset_cfg["size"])
    if kind == "bpaec":
        return BPAECDataset(
            root=dataset_cfg["root"],
            structure=dataset_cfg["structure"],
            layer=dataset_cfg["layer"],
            split=split,
            size=dataset_cfg["size"],
            train_ratio=dataset_cfg.get("train_ratio", 0.8),
            seed=dataset_cfg.get("seed", 42),
        )
    raise ValueError(f"Unknown dataset type: {kind}")
