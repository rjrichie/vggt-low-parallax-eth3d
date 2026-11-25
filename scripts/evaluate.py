"""Compare VGGT, VGGT+BA and COLMAP against ETH3D ground truth.

Pose error per image pair = max(relative rotation error, translation-direction
error), in degrees. Reports AUC@5/10 and failure rate per baseline-to-depth
bin, writes results/summary.csv and results/pose_error_vs_ratio.png.
"""
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import OUT, auc, load_gt, load_subsets, pairwise_errors

METHODS = ["vggt", "vggt_ba", "colmap"]
LABEL = {"vggt": "VGGT (feed-forward)", "vggt_ba": "VGGT + BA", "colmap": "COLMAP (SIFT, incremental)"}
BINS = [0, 0.1, 0.2, 0.4, 0.8, 10]

rows, gts = [], {}
for s in load_subsets():
    vf, cf = OUT / "vggt" / f"{s['id']}.npz", OUT / "colmap" / s["id"] / "poses.npz"
    if not (vf.exists() and cf.exists()):
        continue
    if s["scene"] not in gts:
        gts[s["scene"]] = load_gt(s["scene"])[1]
    g = gts[s["scene"]]
    gt = {n: (g[n]["R"], g[n]["t"]) for n in s["images"]}
    E = np.load(vf)["extrinsic"]
    C = np.load(cf)
    est = {"vggt": {n: (e[:, :3], e[:, 3]) for n, e in zip(s["images"], E)}}
    for m in ("vggt_ba", "colmap"):
        est[m] = {n: (C[f"{m}__{n}"][:, :3], C[f"{m}__{n}"][:, 3]) for n in s["images"] if f"{m}__{n}" in C}
    for m in METHODS:
        r, t = pairwise_errors(gt, est[m], s["images"])
        rows.append(dict(id=s["id"], scene=s["scene"], kind=s["kind"], ratio=s["ratio"], method=m,
                         registered=len(est[m]), n=len(s["images"]), err=np.maximum(r, t),
                         rot=np.median(r), tra=np.median(t)))

# sanity check: ground truth against itself must give zero error
s0 = load_subsets()[0]; g = gts[s0["scene"]]
gt0 = {n: (g[n]["R"], g[n]["t"]) for n in s0["images"]}
assert max(np.max(x) for x in pairwise_errors(gt0, gt0, s0["images"])) < 1e-3

n_sub = len(rows) // len(METHODS)
print(f"{n_sub} subsets evaluated\n")
hdr = f"{'baseline/depth':>15} {'n':>4} " + " ".join(f"{m + ' AUC@5/@10/fail':>26}" for m in METHODS)
print(hdr)
table = []
for lo, hi in zip(BINS[:-1], BINS[1:]):
    line = f"{f'{lo}-{hi}' if hi < 10 else f'>{lo}':>15}"
    cnt = None
    rec = dict(bin=f"{lo}-{hi}")
    for m in METHODS:
        sel = [r for r in rows if r["method"] == m and lo <= r["ratio"] < hi]
        cnt = len(sel)
        if not sel:
            continue
        err = np.concatenate([r["err"] for r in sel])
        fail = np.mean([r["registered"] < r["n"] for r in sel])
        a5, a10 = auc(err, 5), auc(err, 10)
        rec.update({f"{m}_auc5": a5, f"{m}_auc10": a10, f"{m}_fail": fail})
        line += f" {100 * a5:9.1f} {100 * a10:6.1f} {100 * fail:7.0f}%"
    rec["n"] = cnt
    table.append(rec)
    print(line[:15] + f" {cnt:>4}" + line[15:])

allrow = dict(bin="all", n=n_sub)
print("\noverall:")
for m in METHODS:
    err = np.concatenate([r["err"] for r in rows if r["method"] == m])
    fail = np.mean([r["registered"] < r["n"] for r in rows if r["method"] == m])
    allrow.update({f"{m}_auc5": auc(err, 5), f"{m}_auc10": auc(err, 10), f"{m}_fail": fail})
    print(f"  {LABEL[m]:28} AUC@5 {100 * auc(err, 5):5.1f}  AUC@10 {100 * auc(err, 10):5.1f}  "
          f"median pair error {np.median(err):6.2f} deg  subsets with failures {100 * fail:3.0f}%")
table.append(allrow)
with open(OUT / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(table[-1].keys()) + ["n"])
    w.writeheader()
    for r in table:
        w.writerow(r)

fig, ax = plt.subplots(figsize=(7, 4.2))
for m, c in zip(METHODS, ["#c8371e", "#2f6db5", "#555555"]):
    sel = [r for r in rows if r["method"] == m]
    x = np.array([r["ratio"] for r in sel]); y = np.array([np.median(r["err"]) for r in sel])
    ax.scatter(x, np.clip(y, 0.05, 180), s=10, alpha=0.35, color=c)
    centers, med = [], []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        k = (x >= lo) & (x < hi)
        if k.any():
            centers.append(np.median(x[k])); med.append(np.median(y[k]))
    ax.plot(centers, np.clip(med, 0.05, 180), "-o", color=c, label=LABEL[m])
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xlabel("baseline-to-depth ratio (log)"); ax.set_ylabel("median pairwise pose error [deg] (log)")
ax.set_title(f"ETH3D, {n_sub} five-image subsets, GT intrinsics")
ax.legend(frameon=False); fig.tight_layout()
fig.savefig(OUT / "pose_error_vs_ratio.png", dpi=150)
print(f"\nwrote {OUT / 'summary.csv'} and {OUT / 'pose_error_vs_ratio.png'}")
