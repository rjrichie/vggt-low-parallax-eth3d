"""Shared helpers: ETH3D ground-truth parsing and pose-error metrics.

Imports only numpy, so it can be used from both the torch (VGGT) and the
pycolmap processes, which must stay separate (their OpenMP runtimes clash).
"""
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "results"


def qvec2rotmat(q):
    w, x, y, z = np.asarray(q) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * y * y - 2 * z * z, 2 * x * y - 2 * w * z, 2 * x * z + 2 * w * y],
        [2 * x * y + 2 * w * z, 1 - 2 * x * x - 2 * z * z, 2 * y * z - 2 * w * x],
        [2 * x * z - 2 * w * y, 2 * y * z + 2 * w * x, 1 - 2 * x * x - 2 * y * y],
    ])


def read_cameras(path):
    cams = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        cams[int(e[0])] = dict(model=e[1], width=int(e[2]), height=int(e[3]),
                               params=[float(v) for v in e[4:]])
    return cams


def read_images(path):
    """Return {name: dict(R, t, camera_id, point_ids)} with R, t = cam_from_world."""
    imgs = {}
    lines = [l for l in open(path) if not l.startswith("#")]
    for head, obs in zip(lines[0::2], lines[1::2]):
        e = head.split()
        q = np.array([float(v) for v in e[1:5]])
        t = np.array([float(v) for v in e[5:8]])
        ids = [int(v) for v in obs.split()[2::3]]
        imgs[e[9]] = dict(R=qvec2rotmat(q), t=t, camera_id=int(e[8]),
                          point_ids=[i for i in ids if i >= 0])
    return imgs


def read_points(path):
    pts = {}
    for line in open(path):
        if line.startswith("#") or not line.strip():
            continue
        e = line.split()
        pts[int(e[0])] = np.array([float(v) for v in e[1:4]])
    return pts


def load_gt(scene):
    d = DATA / scene / "dslr_calibration_undistorted"
    return read_cameras(d / "cameras.txt"), read_images(d / "images.txt"), d


def rot_angle(R):
    """Rotation angle via atan2 (stable near 0 deg, unlike arccos of the trace)."""
    s = np.linalg.norm([R[2, 1] - R[1, 2], R[0, 2] - R[2, 0], R[1, 0] - R[0, 1]]) / 2
    c = (np.trace(R) - 1) / 2
    return np.degrees(np.arctan2(s, c))


def vec_angle(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na < 1e-12 or nb < 1e-12:
        return 180.0
    return np.degrees(np.arccos(np.clip(a @ b / (na * nb), -1, 1)))


def pairwise_errors(gt, est, names):
    """Relative rotation and translation-direction errors (deg) for all pairs.

    gt/est: {name: (R, t)} as cam_from_world. Images missing from est count
    as failures (180 deg), following the usual SfM evaluation protocol.
    """
    rot, tra = [], []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            if a not in est or b not in est:
                rot.append(180.0); tra.append(180.0); continue
            def rel(P):
                (Ra, ta), (Rb, tb) = P[a], P[b]
                Rab = Rb @ Ra.T
                return Rab, tb - Rab @ ta
            Rg, tg = rel(gt)
            Re, te = rel(est)
            rot.append(rot_angle(Re @ Rg.T))
            tra.append(vec_angle(te, tg))
    return np.array(rot), np.array(tra)


def auc(errors, thr):
    """Area under the recall curve up to thr degrees (as in GLOMAP/VGGT)."""
    e = np.sort(np.asarray(errors))
    recall = np.arange(1, len(e) + 1) / len(e)
    e = np.r_[0, e]; recall = np.r_[0, recall]
    last = np.searchsorted(e, thr)
    x = np.r_[e[:last], thr]; y = np.r_[recall[:last], recall[last - 1]]
    return float(np.trapz(y, x) / thr)


def load_subsets():
    return json.load(open(OUT / "subsets.json"))
