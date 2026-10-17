#!/usr/bin/env python3
"""Worker for the `neural` strategy. Runs in the LINEART venv, not the main one.

candidates.py cannot import torch -- keeping it out is the whole point of the
split venv -- so it shells out to this, which writes a line-strength map to a
cache file. One process per cache miss; hits cost nothing.

  ~/.venvs/png2svg-lineart/bin/python lineart_backend.py <in.png> <out.npy> [res]

`res` sets detect_resolution AND image_resolution. Both default to 512 in
controlnet_aux; leaving them there means the map is computed at 512 and must be
upsampled to the working size, and thresholding an upsampled blur is what
produced ragged, chewed edges on clean source art.

Output: float16 array, source resolution, 0..1, HIGH == line.
"""
import sys
from pathlib import Path
import numpy as np
import cv2
from PIL import Image

Image.MAX_IMAGE_PIXELS = None


def main():
    if len(sys.argv) not in (3, 4):
        print(__doc__); return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    res = int(sys.argv[3]) if len(sys.argv) == 4 else 512
    im = Image.open(src); im.load()
    im = im.convert("RGB")

    import torch
    from controlnet_aux import LineartDetector
    det = LineartDetector.from_pretrained("lllyasviel/Annotators")

    # The annotator is the single largest per-image cost -- ~6.7s on this
    # laptop's CPU, and it runs once for every image. On a CUDA machine it is
    # well under a second. Autodetect rather than hardcode: this file is shared
    # between the AMD-iGPU laptop (no CUDA, stays on CPU, unchanged behaviour)
    # and the 24-core / RTX 4070 desktop. Output is identical either way; only
    # the wall clock changes.
    if torch.cuda.is_available():
        try:
            det = det.to("cuda")
        except Exception:
            pass                                   # any failure -> stay on CPU

    # Ask for the map at the size we actually want it, so nothing is upsampled.
    a = np.asarray(det(im, coarse=False, detect_resolution=res,
                       image_resolution=res).convert("L"), dtype=np.float32)
    # The detector returns its own internal resolution (512), never the input's.
    # Using it unresampled silently misaligns every mask built from it.
    if a.shape[:2] != (im.height, im.width):
        a = cv2.resize(a, (im.width, im.height), interpolation=cv2.INTER_LINEAR)
    if a.mean() > 127:
        a = 255.0 - a
    peak = float(a.max())
    a = a / peak if peak > 1e-6 else a

    dst.parent.mkdir(parents=True, exist_ok=True)
    np.save(dst, a.astype(np.float16))
    return 0


if __name__ == "__main__":
    sys.exit(main())
