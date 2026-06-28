"""Reconstruct one specimen/anatomy and visualize raw + denoised US clouds over CT.

A single script that ties the two parts together:

1. **Reconstruct** (optional) — run the reconstruction for one specimen + one
   anatomy and produce **two** clouds from a *single* segmentation pass:
   - ``intraoperative_dir/specimenNN_<anatomy>_raw.xyz`` — the raw merged union.
   - ``intraoperative_dir/specimenNN_<anatomy>.xyz``      — the **denoised** cloud
     (main_UltraBonesHip.py-ready). By default this uses the CT-free **acoustic-shadow** denoiser
     (``--denoise-method shadow``), which keeps
     points sitting above a dark acoustic shadow (true bone) and drops bright-below
     reverberation/soft-tissue detections — per anatomy (absolute shadow for femur,
     depth-normalized for pelvis). ``dbscan`` (the geometric denoiser) and ``none``
     are also selectable.
   It also exports ``preoperative_dir/specimenNN_<anatomy>.stl`` — the matching CT
   bone mesh — so the pair shares one frame.
2. **Visualize** — open blocking Open3D windows showing the **raw** cloud (orange)
   and then the **denoised** cloud (green), each overlaid on the CT mesh (bone), and
   print each cloud's US→CT surface distance so the before/after is quantified.

By default the pair is reconstructed only if it is missing; pass
``--force-reconstruct`` to always rebuild, or ``--no-reconstruct`` to visualize
whatever is already on disk.

Usage::

    # reconstruct specimen02/left_femur if needed (shadow denoiser), then show raw + denoised
    python reconstruction/visualize_reconstruction.py --specimen 2 --anatomy left_femur

    # always rebuild (coarser/faster with a larger frame stride), then show
    python reconstruction/visualize_reconstruction.py --force-reconstruct --frame-stride 10

    # compare against the old geometric denoiser
    python reconstruction/visualize_reconstruction.py --denoise-method dbscan --force-reconstruct

    # just visualize an existing pair, no reconstruction, no window (headless check)
    python reconstruction/visualize_reconstruction.py --no-reconstruct --no-show

    # bare host: point at the raw dataset / checkpoint explicitly
    python reconstruction/visualize_reconstruction.py \
        --dataset-root /mnt/UltraBonesHip \
        --checkpoint-path reconstruction/models/epoch_30_leave_12_out.pth
"""

import argparse
import os
import sys
from dataclasses import replace

import numpy as np
import open3d as o3d

# Allow running as a plain script (python reconstruction/visualize_reconstruction.py)
# as well as a module: make the repo root importable so `reconstruction` resolves.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)


def _merge_points(clouds):
    """Concatenate per-record cloud points (and the shadow features carried in the
    colours channel, if present) into ``(points, below, depth)`` arrays."""
    pts = np.concatenate([np.asarray(c.points) for c in clouds], axis=0)
    if all(c.has_colors() for c in clouds):
        cols = np.concatenate([np.asarray(c.colors) for c in clouds], axis=0)
        return pts, cols[:, 0], cols[:, 1]
    return pts, None, None


def reconstruct_pair(specimen_id, anatomy, config_path, dataset_root=None,
                     checkpoint_path=None, frame_stride=None, denoise_method="shadow"):
    """Reconstruct one specimen + anatomy and write the **raw** and **denoised**
    clouds from a single segmentation pass.

    Returns ``(raw_file, denoised_file, preop_file)`` (paths taken from the resolved
    config, so they always match what was written).
    """
    from reconstruction.calibration import construct_ultrasound_calibration_matrix
    from reconstruction.config import load_config
    from reconstruction.filtering import (
        denoise_point_cloud,
        denoise_with_shadow,
        downsample_to_n_random,
    )
    from reconstruction.merge import export_preoperative_mesh
    from reconstruction.pipeline import _list_records, _seed_everything
    from reconstruction.record import reconstruct_one_record
    from reconstruction.segmentation import BoneSegmenter

    cfg = load_config(config_path)
    spec = f"specimen{specimen_id:02d}"
    if anatomy not in cfg.merge_map:
        raise SystemExit(
            f"anatomy '{anatomy}' is not in the config merge_map "
            f"({', '.join(cfg.merge_map)}). Pick one of those or add it to "
            f"{config_path}."
        )

    changes = {
        "specimen_ids": [spec],
        "merge_map": {anatomy: cfg.merge_map[anatomy]},
        "denoise_method": denoise_method,
        "export_preoperative": True,  # we need the CT mesh for the overlay
        "overwrite": True,
        "write_per_record": False,    # we keep the per-record clouds in memory
    }
    if dataset_root is not None:
        changes["dataset_root"] = os.path.abspath(dataset_root)
    if checkpoint_path is not None:
        changes["checkpoint_path"] = os.path.abspath(checkpoint_path)
    if frame_stride is not None:
        changes["frame_stride"] = frame_stride
    cfg = replace(cfg, **changes)

    print(f"Reconstructing {spec}/{anatomy} (denoise_method={cfg.denoise_method}, "
          f"frame_stride={cfg.frame_stride}) ...")
    print(f"  dataset_root: {cfg.dataset_root}")
    print(f"  output      : {cfg.intraoperative_dir} | {cfg.preoperative_dir}")

    _seed_everything(cfg.seed)
    segmenter = BoneSegmenter(
        cfg.checkpoint_path, threshold=cfg.segmentation_threshold,
        batch_size=cfg.seg_batch_size, num_workers=cfg.seg_num_workers, amp=cfg.seg_amp,
    )
    print(f"  segmentation model on {segmenter.device}")

    preset = cfg.preset_for(spec)
    T_idealProbe_USimage, T_scale = construct_ultrasound_calibration_matrix(preset)

    # --- one segmentation pass: reconstruct every record of this anatomy ---
    records_root = os.path.join(cfg.dataset_root, spec, "ultrasound_records")
    clouds = []
    for folder in cfg.merge_map[anatomy]:
        folder_path = os.path.join(records_root, folder)
        for record_name in _list_records(folder_path):
            raw = reconstruct_one_record(
                os.path.join(folder_path, record_name), segmenter,
                T_idealProbe_USimage, T_scale, preset, cfg,
            )
            if raw is not None and len(raw.points) > 0:
                clouds.append(raw)
    if not clouds:
        raise SystemExit(f"No usable records reconstructed for {spec}/{anatomy}.")

    points, below, depth = _merge_points(clouds)

    # --- raw merged cloud (no denoise) ---
    raw_pcd = o3d.geometry.PointCloud()
    raw_pcd.points = o3d.utility.Vector3dVector(points)
    raw_pcd = downsample_to_n_random(raw_pcd, cfg.merge_downsample, seed=cfg.seed)

    # --- denoised merged cloud (CT-free) ---
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(points)
    if cfg.denoise_method == "shadow" and below is not None:
        p = cfg.shadow_params_for(anatomy)
        den = denoise_with_shadow(
            merged, below, depth=depth, work_points=cfg.denoise_work_points,
            below_thr=p["below_thr"], below_depthnorm=p["below_depthnorm"],
            dbscan=p["dbscan"], dbscan_eps=p["eps"], dbscan_min_points=p["min_points"],
            keep_frac=p["keep_frac"], keep_abs_frac=p["keep_abs"],
            sor_nb_neighbors=p["sor_nb"], sor_std_ratio=p["sor_std"], seed=cfg.seed,
            log=print,
        )
    elif cfg.denoise_method == "none" or not cfg.denoise_enable:
        den = merged
    else:  # dbscan
        den = denoise_point_cloud(
            merged, work_points=cfg.denoise_work_points,
            dbscan_eps=cfg.denoise_dbscan_eps, dbscan_min_points=cfg.denoise_dbscan_min_points,
            keep_frac=cfg.denoise_keep_frac, keep_abs_frac=cfg.denoise_keep_abs_frac,
            sor_nb_neighbors=cfg.sor_nb_neighbors, sor_std_ratio=cfg.sor_std_ratio,
            seed=cfg.seed, log=print,
        )
    den = downsample_to_n_random(den, cfg.merge_downsample, seed=cfg.seed)

    # --- write both versions + the matching CT mesh ---
    os.makedirs(cfg.intraoperative_dir, exist_ok=True)
    raw_file = os.path.join(cfg.intraoperative_dir, f"{spec}_{anatomy}_raw.xyz")
    denoised_file = os.path.join(cfg.intraoperative_dir, f"{spec}_{anatomy}.xyz")
    o3d.io.write_point_cloud(raw_file, raw_pcd)
    o3d.io.write_point_cloud(denoised_file, den)
    print(f"  wrote raw      -> {raw_file}  ({len(raw_pcd.points)} points)")
    print(f"  wrote denoised -> {denoised_file}  ({len(den.points)} points)")
    preop_file = export_preoperative_mesh(spec, anatomy, cfg)
    return raw_file, denoised_file, preop_file


def _surface_distance_mm(pcd, mesh):
    """Mean / HD95 nearest-surface distance (mm) from the US cloud to the CT mesh."""
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(o3d.t.geometry.TriangleMesh.from_legacy(mesh))
    query = np.asarray(pcd.points, dtype=np.float32)
    dist = scene.compute_distance(o3d.core.Tensor(query)).numpy()
    return float(dist.mean()), float(np.percentile(dist, 95))


def visualize_pair(raw_file, denoised_file, preop_file, show=True):
    """Overlay the raw (orange) then the denoised (green) cloud on the CT mesh and
    report each cloud's US→CT surface distance."""
    for f in (raw_file, denoised_file, preop_file):
        if f is None or not os.path.isfile(f):
            raise FileNotFoundError(f"missing input for visualization: {f}")

    mesh = o3d.io.read_triangle_mesh(preop_file)
    mesh.compute_vertex_normals()
    mesh.paint_uniform_color((0.85, 0.80, 0.70))

    raw = o3d.io.read_point_cloud(raw_file)
    den = o3d.io.read_point_cloud(denoised_file)
    raw.paint_uniform_color((0.95, 0.45, 0.10))   # orange = raw / noisy
    den.paint_uniform_color((0.00, 0.85, 0.10))   # green  = denoised / clean

    print(f"\nCT mesh  : {preop_file}  ({len(mesh.vertices)} verts, {len(mesh.triangles)} tris)")
    print(f"{'cloud':<10}{'points':>10}{'US->CT mean':>14}{'HD95':>10}")
    for name, pcd in (("raw", raw), ("denoised", den)):
        try:
            mean_d, hd95 = _surface_distance_mm(pcd, mesh)
            print(f"{name:<10}{len(pcd.points):>10}{mean_d:>11.2f} mm{hd95:>7.2f} mm")
        except Exception as exc:  # raycasting is optional; never block on it
            print(f"{name:<10}{len(pcd.points):>10}   (surface-distance skipped: {exc})")

    if not show:
        return raw, den, mesh

    extent = float(np.linalg.norm(mesh.get_max_bound() - mesh.get_min_bound()))
    frame = o3d.geometry.TriangleMesh.create_coordinate_frame(
        size=0.1 * extent, origin=mesh.get_min_bound()
    )
    stem = os.path.basename(denoised_file)
    # Two sequential windows so the before/after is unambiguous (close one to advance).
    print("\n[viewer] showing RAW vs CT — close the window to see the denoised cloud.")
    o3d.visualization.draw_geometries(
        [mesh, raw, frame], window_name=f"RAW (orange) vs CT — {stem}",
        mesh_show_back_face=True,
    )
    print("[viewer] showing DENOISED vs CT.")
    o3d.visualization.draw_geometries(
        [mesh, den, frame], window_name=f"DENOISED (green) vs CT — {stem}",
        mesh_show_back_face=True,
    )
    return raw, den, mesh


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--specimen", type=int, default=2,
                        help="specimen id (default: 2)")
    parser.add_argument("--anatomy", default="left_femur",
                        help="anatomy name; must be a config merge_map key, e.g. "
                             "left_femur / right_femur / pelvis (default: left_femur)")

    # Reconstruction controls.
    parser.add_argument("--config",
                        default=os.path.join("reconstruction", "conf", "reconstruction.yaml"),
                        help="reconstruction YAML config")
    parser.add_argument("--denoise-method", default="shadow",
                        choices=["shadow", "dbscan", "none"],
                        help="CT-free denoiser for the denoised cloud (default: shadow)")
    parser.add_argument("--dataset-root", default=None,
                        help="raw dataset root (overrides config; e.g. /mnt/UltraBonesHip)")
    parser.add_argument("--checkpoint-path", default=None,
                        help="segmentation checkpoint path (overrides config)")
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="subsample 1 in N frames (overrides config; larger = faster/coarser)")
    parser.add_argument("--force-reconstruct", action="store_true",
                        help="rebuild the clouds even if they already exist")
    parser.add_argument("--no-reconstruct", action="store_true",
                        help="never reconstruct; visualize whatever is already on disk")

    # Visualization-only path / output locations.
    parser.add_argument("--intraoperative-dir", default="data/UltraBonesHip/intraoperative",
                        help="directory holding the .xyz clouds (used when not reconstructing)")
    parser.add_argument("--preoperative-dir", default="data/UltraBonesHip/preoperative",
                        help="directory holding the .stl meshes (used when not reconstructing)")
    parser.add_argument("--no-show", action="store_true",
                        help="load and report only; do not open a window (headless)")
    args = parser.parse_args()

    stem = f"specimen{args.specimen:02d}_{args.anatomy}"
    raw_file = os.path.join(args.intraoperative_dir, f"{stem}_raw.xyz")
    denoised_file = os.path.join(args.intraoperative_dir, f"{stem}.xyz")
    preop_file = os.path.join(args.preoperative_dir, f"{stem}.stl")
    have_all = all(os.path.isfile(f) for f in (raw_file, denoised_file, preop_file))

    if args.no_reconstruct:
        if not have_all:
            print("[info] --no-reconstruct set but the raw/denoised/mesh trio is "
                  "incomplete; run without it (or with --force-reconstruct) to build it.")
    elif args.force_reconstruct or not have_all:
        if not have_all and not args.force_reconstruct:
            print(f"[info] {stem} clouds not all present — reconstructing first.")
        raw_file, denoised_file, preop_file = reconstruct_pair(
            args.specimen, args.anatomy, args.config,
            dataset_root=args.dataset_root, checkpoint_path=args.checkpoint_path,
            frame_stride=args.frame_stride, denoise_method=args.denoise_method,
        )

    visualize_pair(raw_file, denoised_file, preop_file, show=not args.no_show)


if __name__ == "__main__":
    main()
