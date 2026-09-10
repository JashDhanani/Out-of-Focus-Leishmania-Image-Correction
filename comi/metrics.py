import numpy as np
import torch
from scipy.stats import pearsonr
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def _to_numpy_image(t: torch.Tensor) -> np.ndarray:
    """Converts a single CHW tensor in [-1, 1] to an HWC uint8 numpy array."""
    img = (t.detach().cpu().clamp(-1, 1) + 1.0) / 2.0
    img = (img.permute(1, 2, 0).numpy() * 255.0).round().astype(np.uint8)
    return img


def compute_metrics(generated: torch.Tensor, target: torch.Tensor) -> dict:
    """PSNR, SSIM and PCC (Eq. 1-3) between one generated/target image pair."""
    gen = _to_numpy_image(generated)
    tgt = _to_numpy_image(target)

    psnr = peak_signal_noise_ratio(tgt, gen, data_range=255)
    ssim = structural_similarity(tgt, gen, data_range=255, channel_axis=-1)
    pcc, _ = pearsonr(gen.ravel().astype(np.float64), tgt.ravel().astype(np.float64))

    return {"psnr": float(psnr), "ssim": float(ssim), "pcc": float(pcc)}
