#!/usr/bin/env python3
"""Dump the intermediate results stored in the package (vectors/stages_img_00.npz) as raw
little-endian files that fpgate-accel --selftest compares against: <name>.i64 for integer
tensors (int64), <name>.f32 for float tensors. Reads the package, writes elsewhere.

  docker run --rm -v <repo>/model:/model:ro -v <repo>/accel/hps/pkg:/out --entrypoint python \
      model-torch:latest /out/../tools/stages_to_raw.py /model/quant/release/e1_r7/vectors/stages_img_00.npz /out/stages_img_00
"""
import sys
from pathlib import Path

import numpy as np

src, out = Path(sys.argv[1]), Path(sys.argv[2])
out.mkdir(parents=True, exist_ok=True)
z = np.load(src)
for k in z.files:
    a = z[k]
    if np.issubdtype(a.dtype, np.integer):
        a.astype("<i8").tofile(out / f"{k}.i64")
    else:
        a.astype("<f4").tofile(out / f"{k}.f32")
    print(f"{k:14s} {a.dtype} {tuple(a.shape)}")
