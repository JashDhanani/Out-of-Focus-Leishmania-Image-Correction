# Data

`m3jxgb54c9-4.zip` is the paper's self-collected dataset, published on
Mendeley Data: https://data.mendeley.com/datasets/m3jxgb54c9/4

It bundles more than the two datasets used by this paper:

| Folder        | Used by this paper? | Contents |
|---------------|----------------------|----------|
| `Leishmania/` | Yes (Dataset 1)      | Paired blur/sharp bright-field images, pre-split into `Leishmania_{blurred,clear}_{train,test}/` |
| `BPAEC/`      | Yes (Dataset 2)      | Confocal z-stacks of `nucleus/`, `actin/`, `mitochondria/`, each with layers `Z004`-`Z010` (`Z007` is the in-focus ground truth) |
| `Babesia/`, `Toxoplasma/`, `Trypanosoma/` | No | COCO-style parasite *detection* datasets (train/val/test2017 + instance annotations) from a different, unrelated study bundled in the same Mendeley record. Not needed for out-of-focus correction. |

The third dataset used in the paper for generalization testing (BBBC006,
from the Broad Bioimage Benchmark Collection) is **not** included in this
zip -- see `scripts/download_bbbc006.py`.

## Preparing the data

Unzip `m3jxgb54c9-4.zip` here directly:

```bash
cd data && unzip m3jxgb54c9-4.zip
```

That produces `data/Leishmania/` and `data/BPAEC/` in exactly the layout
`comi/datasets.py` expects (see `configs/*.yaml` -> `dataset.root`) --
no extra preparation step needed.
