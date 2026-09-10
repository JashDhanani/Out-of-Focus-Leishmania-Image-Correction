# Out-of-Focus Microscopic Image Correction Using CycleGAN

<!-- PyTorch reimplementation of:

> Zhang, C., Jiang, H., Liu, W., Li, J., Tang, S., Juhas, M., & Zhang, Y. (2022).
> *Correction of out-of-focus microscopic images by deep learning.*
> Computational and Structural Biotechnology Journal, 20, 1957-1966.
> https://doi.org/10.1016/j.csbj.2022.04.003 -->

## Live Demo

[![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-sm.svg)](https://huggingface.co/spaces/Jash-176/out-of-focus-leishmania-image-correction)

## Method

A CycleGAN with two generators (`G_s`: out-of-focus -> in-focus, `G_t`:
in-focus -> out-of-focus) and two PatchGAN discriminators (`D_s`, `D_t`),
trained with a **multi-component weighted loss**:

```
L = lambda1 * L_GAN + lambda2 * L_content + lambda3 * L_cycle
```

- `L_GAN`: vanilla adversarial loss 
- `L_content`: VGG-19 perceptual/feature loss between generated and the
  *paired* ground-truth image -- this dataset is paired, so the
  content loss can supervise directly rather than relying on cycle
  consistency alone
- `L_cycle`: L1 cycle-consistency loss
- Hyperparameters: `lambda1=1, lambda2=1, lambda3=0.001`

Generator: 7x7 conv + 2 stride-2 downsampling blocks + 9 ResNet blocks + 2
transposed-conv upsampling blocks.
Discriminator: 70x70 Markovian PatchGAN.

<!-- ## Project layout

```
.
├── comi/                  # library code
│   ├── models/
│   │   ├── generator.py       # ResnetGenerator (9 resblocks)
│   │   ├── discriminator.py   # PatchDiscriminator
│   │   └── vgg.py             # frozen VGG-19 feature extractor
│   ├── datasets.py         # LeishmaniaDataset, BPAECDataset + augmentation
│   ├── losses.py            # GANLoss, ContentLoss, CycleLoss
│   ├── metrics.py           # PSNR / SSIM / PCC (Eq. 1-3)
│   └── utils.py
├── configs/                # one YAML per experiment
│   ├── leishmania.yaml
│   └── bpaec.yaml
├── scripts/
│   └── download_bbbc006.py   # generalization-test dataset, not part of training data
├── train.py
├── evaluate.py             # PSNR/SSIM/PCC on the test split (Table 2/3)
├── inference.py            # run a trained G_s on new images (Table/Fig. 5)
└── data/                   # raw + processed datasets (git-ignored)
``` -->

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

## Data

The two publicaly available datasets used to train and test this paper are
in `data/m3jxgb54c9-4.zip` (Mendeley Data). See `data/README.md` for what
it contains. Unzip it in place:

```bash
cd data && unzip m3jxgb54c9-4.zip
```

This produces `data/Leishmania/` and `data/BPAEC/`, which is exactly
what `configs/leishmania.yaml` and `configs/bpaec.yaml` point at.

## Training

```bash
# Dataset 1: Leishmania
python train.py --config configs/leishmania.yaml

# Dataset 2: BPAEC
python train.py --config configs/bpaec.yaml
```

100,000 iterations, batch size 1, Adam (lr=1e-4, beta1=0.5, beta2=0.99),
linearly decayed to 0 after the first 50,000 iterations -- matching
Section 3.3.

## Evaluation

```bash
python evaluate.py --config configs/leishmania.yaml \
    --checkpoint train_outputs/leishmania/checkpoints/final.pt
```

Reports PSNR / SSIM / PCC averaged over the held-out test split.

## Inference / generalization test

```bash
python inference.py \
    --checkpoint train_outputs/bpaec_nucleus_z004/checkpoints/final.pt \
    --input path/to/images_or_a_single_image \
    --output results/corrected
```

To correct the BPAEC Nucleus z004 images, download the dataset and feed it straight into a BPAEC-trained
checkpoint with no further training.

<!-- ## Notes / deviations from the paper

- The paper trains in Keras/TensorFlow on a single Tesla K40C; this is a
  from-scratch PyTorch reimplementation, not a port of the original code.
- BPAEC results in Table 3 come from **18 separately trained models**
  (3 structures x 6 defocus layers), each specific to one blur level --
  `configs/bpaec.yaml` trains one such combination at a time.
- SSIM/PSNR here use `scikit-image`'s standard windowed implementations
  rather than a literal single-window transcription of Eq. 2. -->
