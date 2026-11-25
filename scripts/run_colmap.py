"""Classical baselines per subset, using ground-truth intrinsics (kept fixed).

1. COLMAP: SIFT + exhaustive matching + incremental SfM.
2. VGGT + BA: VGGT poses as initialization, points triangulated from the
   same SIFT matches, then standard bundle adjustment (poses + points).

Saves results/colmap/<id>/poses.npz with {method}__{image name} -> 3x4.
Must run in a separate process from torch (OpenMP clash).
"""
import shutil
import sys
import time

import numpy as np
import pycolmap

from common import DATA, OUT, load_gt, load_subsets

pycolmap.logging.minloglevel = 2  # quiet


def rotmat2qvec(R):
    return pycolmap.Rotation3d(R).quat[[3, 0, 1, 2]]  # xyzw -> wxyz


def poses_of(rec):
    out = {}
    for img in rec.images.values():
        if img.has_pose:
            T = img.cam_from_world()
            out[img.name] = np.hstack([T.rotation.matrix(), T.translation[:, None]])
    return out


def setup_database(s, work, cams, gt):
    db = work / "database.db"
    if db.exists():
        return db
    img_dir = DATA / s["scene"] / "images"
    reader = pycolmap.ImageReaderOptions(camera_model="PINHOLE")
    ext = pycolmap.FeatureExtractionOptions(max_image_size=1600, use_gpu=False)
    pycolmap.extract_features(db, img_dir, image_names=s["images"],
                              camera_mode=pycolmap.CameraMode.PER_IMAGE,
                              reader_options=reader, extraction_options=ext)
    with pycolmap.Database.open(db) as d:
        for n in s["images"]:
            im = d.read_image_with_name(n)
            cam = d.read_camera(im.camera_id)
            c = cams[gt[n]["camera_id"]]
            cam.width, cam.height = c["width"], c["height"]
            cam.params = c["params"]
            cam.has_prior_focal_length = True
            d.update_camera(cam)
    pycolmap.match_exhaustive(db, matching_options=pycolmap.FeatureMatchingOptions(use_gpu=False))
    return db


def fixed_intrinsics(opts):
    opts.ba_refine_focal_length = False
    opts.ba_refine_principal_point = False
    opts.ba_refine_extra_params = False
    return opts


def run_colmap(db, img_dir, work):
    out = work / "colmap"
    shutil.rmtree(out, ignore_errors=True); out.mkdir()
    recs = pycolmap.incremental_mapping(db, img_dir, out, fixed_intrinsics(pycolmap.IncrementalPipelineOptions()))
    if not recs:
        return {}
    best = max(recs.values(), key=lambda r: r.num_reg_images())
    return poses_of(best)


def run_vggt_ba(db, img_dir, work, s, ext):
    model = work / "vggt_init"
    shutil.rmtree(model, ignore_errors=True); model.mkdir()
    with pycolmap.Database.open(db) as d:
        ims = {im.name: im for im in d.read_all_images()}
        cams = {c.camera_id: c for c in d.read_all_cameras()}
    with open(model / "cameras.txt", "w") as f:
        for c in cams.values():
            f.write(f"{c.camera_id} PINHOLE {c.width} {c.height} {' '.join(map(str, c.params))}\n")
    with open(model / "images.txt", "w") as f:
        for n, E in zip(s["images"], ext):
            q, t = rotmat2qvec(E[:, :3]), E[:, 3]
            im = ims[n]
            f.write(f"{im.image_id} {' '.join(map(str, q))} {' '.join(map(str, t))} {im.camera_id} {n}\n\n")
    open(model / "points3D.txt", "w").close()
    rec = pycolmap.Reconstruction(model)
    tri = work / "vggt_ba"
    shutil.rmtree(tri, ignore_errors=True); tri.mkdir()
    # Coarse-to-fine: VGGT poses are off by a few degrees, i.e. hundreds of
    # pixels at 6048 px width, so COLMAP's default 4 px triangulation gates
    # reject almost everything. Start loose, refine, then tighten to defaults.
    for px, ang, loss in ((200.0, 10.0, "CAUCHY"), (40.0, 4.0, "CAUCHY"), (4.0, 2.0, "TRIVIAL")):
        opts = fixed_intrinsics(pycolmap.IncrementalPipelineOptions())
        t = opts.triangulation
        t.ignore_two_view_tracks = False
        t.complete_max_reproj_error = t.merge_max_reproj_error = px
        t.create_max_angle_error = t.continue_max_angle_error = ang
        t.min_angle = 1.0
        opts.mapper.filter_max_reproj_error = px
        rec = pycolmap.triangulate_points(rec, db, img_dir, tri, clear_points=True, options=opts)
        ba = pycolmap.BundleAdjustmentOptions(refine_focal_length=False, refine_principal_point=False,
                                              refine_extra_params=False)
        ba.ceres.loss_function_type = getattr(pycolmap.LossFunctionType, loss)
        ba.ceres.loss_function_scale = px
        if rec.num_points3D() >= 10:
            pycolmap.bundle_adjustment(rec, ba)
    return poses_of(rec), rec.num_points3D()


subsets = load_subsets()
only = sys.argv[1:]  # optional subset ids
for k, s in enumerate(subsets):
    if only and s["id"] not in only:
        continue
    work = OUT / "colmap" / s["id"]
    if (work / "poses.npz").exists():
        continue
    vf = OUT / "vggt" / f"{s['id']}.npz"
    if not vf.exists():
        continue  # wait for VGGT
    work.mkdir(parents=True, exist_ok=True)
    cams, gt, _ = load_gt(s["scene"])
    img_dir = DATA / s["scene"] / "images"
    t0 = time.time()
    db = setup_database(s, work, cams, gt)
    res, info = {}, {}
    t = time.time(); P = run_colmap(db, img_dir, work); info["colmap_time"] = time.time() - t
    res.update({f"colmap__{n}": v for n, v in P.items()})
    try:
        t = time.time(); P, npts = run_vggt_ba(db, img_dir, work, s, np.load(vf)["extrinsic"])
        info["vggt_ba_time"] = time.time() - t; info["vggt_ba_points"] = npts
        res.update({f"vggt_ba__{n}": v for n, v in P.items()})
    except Exception as e:  # record failure, keep going
        info["vggt_ba_error"] = repr(e)
    np.savez(work / "poses.npz", info=np.array(repr(info)), **res)
    print(f"[{k + 1}/{len(subsets)}] {s['id']} {time.time() - t0:.0f}s {info}", flush=True)
