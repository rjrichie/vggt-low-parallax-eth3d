"""Run frozen VGGT-1B on every subset and save predicted camera poses.

Saves results/vggt/<subset id>.npz with `extrinsic` (N, 3, 4) as
cam_from_world in OpenCV convention, in the subset's image order.
Run with PYTORCH_ENABLE_MPS_FALLBACK=1 on Apple silicon.
"""
import time

import numpy as np
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

from common import DATA, OUT, load_subsets

dev = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
out = OUT / "vggt"
out.mkdir(parents=True, exist_ok=True)
model = VGGT.from_pretrained("facebook/VGGT-1B").to(dev).eval()

subsets = load_subsets()
for k, s in enumerate(subsets):
    f = out / f"{s['id']}.npz"
    if f.exists():
        continue
    paths = [str(DATA / s["scene"] / "images" / n) for n in s["images"]]
    t = time.time()
    x = load_and_preprocess_images(paths).to(dev)
    with torch.no_grad():
        pred = model(x)
    ext, intr = pose_encoding_to_extri_intri(pred["pose_enc"], x.shape[-2:])
    np.savez(f, extrinsic=ext[0].float().cpu().numpy(), intrinsic=intr[0].float().cpu().numpy(),
             size=np.array(x.shape[-2:]), runtime=time.time() - t)
    print(f"[{k + 1}/{len(subsets)}] {s['id']} {time.time() - t:.1f}s", flush=True)
