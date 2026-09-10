#!/usr/bin/env python
"""Helper for the BBBC006 generalization experiment (Section 4.3, ref [41]).

BBBC006 is NOT bundled in the Mendeley zip used for training -- it must be
downloaded separately from the Broad Bioimage Benchmark Collection:

    https://bbbc.broadinstitute.org/BBBC006

That page lists the per-channel zips (Hoechst-stained nuclei and
phalloidin-stained actin, 32 z-stack layers each). This script only
organizes files you've already downloaded and unzipped into the layout
`inference.py` expects; it does not fetch them itself since the exact
archive filenames on that page can change over time.

Usage:
    1. Download and unzip the BBBC006 image sets from the URL above.
    2. python scripts/download_bbbc006.py --src /path/to/unzipped/BBBC006 --out data/processed/bbbc006
"""
import argparse
import shutil
from pathlib import Path

IMAGE_EXTS = {".tif", ".tiff", ".png"}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--src", required=True, help="directory containing the unzipped BBBC006 images")
    p.add_argument("--out", default="data/processed/bbbc006")
    return p.parse_args()


def main():
    args = parse_args()
    src = Path(args.src)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    files = [p for p in src.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
    if not files:
        raise FileNotFoundError(
            f"No images found under {src}. Download BBBC006 from "
            "https://bbbc.broadinstitute.org/BBBC006 and unzip it first."
        )
    for f in files:
        shutil.copy2(f, out / f.name)
    print(f"Copied {len(files)} images to {out}")
    print("Run inference.py with --input pointing at this directory to "
          "reproduce the zero-shot generalization test on a BPAEC-trained checkpoint.")


if __name__ == "__main__":
    main()
