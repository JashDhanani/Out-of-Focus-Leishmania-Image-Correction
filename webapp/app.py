"""Gradio demo: correct out-of-focus Leishmania microscopy images using the
CycleGAN generator trained in this repo (see ../train.py, ../readme.md).

Run locally:
    pip install -r requirements.txt gradio==5.27.0
    python app.py

(gradio's version is pinned via sdk_version in README.md rather than
requirements.txt, since Hugging Face Spaces provisions it from there --
match that version locally too.)

Deploy: push this folder to a Hugging Face Space (sdk: gradio) -- see
../DEPLOY.md for the exact steps.
"""
from pathlib import Path

import gradio as gr
import numpy as np
import torch
import torchvision.transforms.functional as TF
from PIL import Image

try:
    # Only meaningfully does anything inside an actual Hugging Face Space
    # with ZeroGPU hardware, which is what free-tier Gradio Spaces now run
    # on (CPU-basic is PRO-only for Gradio SDK spaces). Elsewhere (local
    # dev, other hosts) this decorator is a harmless no-op passthrough.
    import spaces
    gpu_decorator = spaces.GPU
except ImportError:
    def gpu_decorator(fn):
        return fn

from model import load_generator

HERE = Path(__file__).parent
WEIGHTS_PATH = HERE / "checkpoints" / "leishmania_generator.pt"
INFERENCE_SIZE = 256  # matches configs/leishmania.yaml dataset.size used at training time

_generator = load_generator(str(WEIGHTS_PATH))


@gpu_decorator
def _run_generator(tensor: torch.Tensor) -> torch.Tensor:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = _generator.to(device)
    with torch.no_grad():
        return model(tensor.to(device))[0].cpu()


def correct_image(image: Image.Image) -> Image.Image:
    if image is None:
        return None

    image = image.convert("RGB")
    original_size = image.size  # (W, H)

    resized = TF.resize(image, [INFERENCE_SIZE, INFERENCE_SIZE])
    tensor = TF.to_tensor(resized).unsqueeze(0) * 2.0 - 1.0

    corrected = _run_generator(tensor)

    corrected = ((corrected.clamp(-1, 1) + 1) / 2 * 255).round().byte()
    corrected = corrected.permute(1, 2, 0).numpy().astype(np.uint8)
    corrected_image = Image.fromarray(corrected).resize(original_size, Image.BICUBIC)
    return corrected_image


example_paths = sorted(str(p) for p in (HERE / "examples").glob("*.jpg"))

with gr.Blocks(title="Out-of-Focus Microscopy Correction") as demo:
    gr.Markdown(
        "# Out-of-Focus Microscopy Correction\n"
        "CycleGAN reimplementation of *Correction of out-of-focus microscopic "
        "images by deep learning* (Zhang et al., 2022), trained on the "
        "paper's Leishmania bright-field dataset. Upload a blurry microscopy "
        "image, or pick one of the samples below, to see the corrected result. "
        "This model was trained specifically on Leishmania parasite images -- "
        "results on unrelated images are not expected to be meaningful."
    )

    with gr.Row():
        input_image = gr.Image(type="pil", label="Out-of-focus input")
        output_image = gr.Image(type="pil", label="Corrected output")

    run_button = gr.Button("Correct image", variant="primary")
    run_button.click(fn=correct_image, inputs=input_image, outputs=output_image)

    gr.Examples(examples=example_paths, inputs=input_image, label="Sample out-of-focus images")

if __name__ == "__main__":
    demo.launch()
