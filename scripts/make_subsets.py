"""Build 5-image subsets per ETH3D scene spanning small to large baselines.

For each anchor image we take (a) its 4 nearest cameras -> small baseline and
(b) 4 random other cameras -> larger baseline. Each subset is tagged with its
baseline-to-depth ratio: median pairwise camera distance divided by the
median depth of ground-truth points seen by the subset.
"""
import itertools
import json
import random

import numpy as np

from common import DATA, OUT, load_gt, read_points

K = 5
random.seed(0)
OUT.mkdir(exist_ok=True)
subsets = []
for scene in sorted(p.name for p in DATA.iterdir() if p.is_dir()):
    _, imgs, gtdir = load_gt(scene)
    pts = read_points(gtdir / "points3D.txt")
    names = sorted(imgs)
    C = {n: -imgs[n]["R"].T @ imgs[n]["t"] for n in names}

    def ratio(sub):
        base = np.median([np.linalg.norm(C[a] - C[b]) for a, b in itertools.combinations(sub, 2)])
        depths = []
        for n in sub:
            ids = imgs[n]["point_ids"]
            X = np.array([pts[i] for i in ids if i in pts])
            if len(X):
                depths.append(np.median((X @ imgs[n]["R"].T + imgs[n]["t"])[:, 2]))
        return float(base / np.median(depths))

    seen = set()
    for a in names:
        near = sorted(names, key=lambda n: np.linalg.norm(C[n] - C[a]))[:K]
        rand = [a] + random.sample([n for n in names if n != a], K - 1)
        for kind, sub in (("near", near), ("random", rand)):
            key = tuple(sorted(sub))
            if key in seen:
                continue
            seen.add(key)
            subsets.append(dict(id=f"{scene}_{len(subsets):04d}", scene=scene, kind=kind,
                                images=sorted(sub), ratio=ratio(sub)))

json.dump(subsets, open(OUT / "subsets.json", "w"), indent=1)
r = np.array([s["ratio"] for s in subsets])
print(f"{len(subsets)} subsets, baseline/depth ratio min {r.min():.3f} "
      f"median {np.median(r):.3f} max {r.max():.3f}")
