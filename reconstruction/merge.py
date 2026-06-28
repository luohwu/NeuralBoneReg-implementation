"""Write the main_UltraBonesHip.py-ready outputs per anatomy.

`merge_anatomy` unions the filtered per-record clouds into the intraoperative
file `<intraoperative_dir>/<specimen>_<anatomy>.xyz`. `export_preoperative_mesh`
copies the matching CT bone segmentation to `<preoperative_dir>/<specimen>_<anatomy>.stl`
so the preoperative model is in the same (CT-segmentation) frame as the
reconstructed cloud. Existing files are never overwritten unless `overwrite` is set.
"""

import os
import shutil

import numpy as np
import open3d as o3d

from .filtering import (
    denoise_point_cloud,
    denoise_with_consensus,
    denoise_with_shadow,
    downsample_to_n_random,
)


def _folder_side(folder):
    """Hemipelvis side from the scan folder ('left_pelvis_axial' -> 'left'). CT-free."""
    head = folder.split("_", 1)[0]
    return head if head in ("left", "right") else "all"


def merge_anatomy(raw_clouds, output_anatomy, specimen_id, cfg, log=print, folders=None):
    """Union, CT-free denoise, downsample and write `<specimen>_<anatomy>.xyz`.

    The per-record clouds are unioned, then denoised geometrically (DBSCAN cluster
    keep + statistical outlier removal -- no CT mesh involved) to drop points from
    frames that did not image bone, and finally downsampled to ``merge_downsample``.
    Returns the output path on success, or ``None`` if there was nothing to merge.
    """
    if folders is None:
        folders = [None] * len(raw_clouds)
    pairs = [(c, f) for c, f in zip(raw_clouds, folders)
             if c is not None and len(c.points) > 0]
    if not pairs:
        log(f"  [skip merge] no clouds for {specimen_id}_{output_anatomy}")
        return None
    clouds = [c for c, _ in pairs]

    all_points = np.concatenate([np.asarray(c.points) for c in clouds], axis=0)
    merged = o3d.geometry.PointCloud()
    merged.points = o3d.utility.Vector3dVector(all_points)

    method = cfg.denoise_method_for(output_anatomy)

    # Recover the per-point features carried in the colours channel (see
    # record._make_pcd): [:,0] = below-window intensity (acoustic shadow), [:,1] =
    # normalised image depth, [:,2] = per-record frame id. The frame id is made
    # globally unique across records (offset by the cumulative frame count) so the
    # consensus denoiser counts DISTINCT sweeps per voxel. Side = scan folder.
    below_parts, depth_parts, frame_parts, side_parts = [], [], [], []
    frame_offset = 0
    for c, folder in pairs:
        n = len(c.points)
        if c.has_colors():
            cols = np.asarray(c.colors)
            below_parts.append(cols[:, 0])
            depth_parts.append(cols[:, 1])
            local_fid = cols[:, 2]
        else:
            below_parts.append(np.full(n, 0.5))
            depth_parts.append(np.full(n, 0.5))
            local_fid = np.zeros(n)
        frame_parts.append(local_fid + frame_offset)
        frame_offset += int(local_fid.max()) + 1 if n else 0
        side_parts.append(np.full(n, _folder_side(folder) if folder else "all"))
    all_below = np.concatenate(below_parts)
    all_depth = np.concatenate(depth_parts)
    all_frame = np.concatenate(frame_parts)
    all_side = np.concatenate(side_parts)

    if cfg.denoise_enable:
        if method == "consensus":
            p = cfg.consensus_params_for(output_anatomy)
            merged = denoise_with_consensus(
                merged,
                all_frame,
                all_side,
                voxel=p["voxel"],
                min_frames=p["min_frames"],
                dbscan_eps=p["eps"],
                dbscan_min_points=p["min_points"],
                sor_nb_neighbors=p["sor_nb"],
                sor_std_ratio=p["sor_std"],
                balance=p["balance"],
                seed=cfg.seed,
                log=log,
            )
        elif method == "shadow":
            p = cfg.shadow_params_for(output_anatomy, specimen_id)
            merged = denoise_with_shadow(
                merged,
                all_below,
                depth=all_depth,
                work_points=cfg.denoise_work_points,
                below_thr=p["below_thr"],
                below_depthnorm=p["below_depthnorm"],
                dbscan=p["dbscan"],
                dbscan_eps=p["eps"],
                dbscan_min_points=p["min_points"],
                keep_frac=p["keep_frac"],
                keep_abs_frac=p["keep_abs"],
                sor_nb_neighbors=p["sor_nb"],
                sor_std_ratio=p["sor_std"],
                seed=cfg.seed,
                log=log,
            )
        else:
            merged = denoise_point_cloud(
                merged,
                work_points=cfg.denoise_work_points,
                dbscan_eps=cfg.denoise_dbscan_eps,
                dbscan_min_points=cfg.denoise_dbscan_min_points,
                keep_frac=cfg.denoise_keep_frac,
                keep_abs_frac=cfg.denoise_keep_abs_frac,
                sor_nb_neighbors=cfg.sor_nb_neighbors,
                sor_std_ratio=cfg.sor_std_ratio,
                seed=cfg.seed,
                log=log,
            )

    merged = downsample_to_n_random(merged, cfg.merge_downsample, seed=cfg.seed)

    os.makedirs(cfg.intraoperative_dir, exist_ok=True)
    out_path = os.path.join(
        cfg.intraoperative_dir, f"{specimen_id}_{output_anatomy}.xyz"
    )
    if os.path.isfile(out_path) and not cfg.overwrite:
        log(f"  [exists] {out_path} preserved (set overwrite: true); skipping write")
        return out_path

    o3d.io.write_point_cloud(out_path, merged)
    log(f"  wrote {out_path}  ({len(merged.points)} points)")
    return out_path


def export_preoperative_mesh(specimen_id, output_anatomy, cfg, log=print):
    """Copy CT_bone_segmentations/<anatomy>.stl as the preoperative model.

    The reconstructed cloud is aligned to this mesh, so copying it (rather than a
    different segmentation) keeps the preop/intraop pair in one frame. Returns the
    destination path, or ``None`` if the source CT mesh is missing.
    """
    src = os.path.join(
        cfg.dataset_root, specimen_id, "CT_bone_segmentations", f"{output_anatomy}.stl"
    )
    if not os.path.isfile(src):
        log(f"  [skip preop] no CT mesh for {output_anatomy}: {src}")
        return None

    os.makedirs(cfg.preoperative_dir, exist_ok=True)
    dst = os.path.join(cfg.preoperative_dir, f"{specimen_id}_{output_anatomy}.stl")
    if os.path.isfile(dst) and not cfg.overwrite:
        log(f"  [exists] {dst} preserved (set overwrite: true); skipping copy")
        return dst

    # The preoperative model fed to NeuralUDF/registration should be the OUTER bone
    # surface (what US can image), not the raw CT segmentation (which carries internal
    # faces). Extract it CT-internally (ray-cast visible shell); fall back to a plain
    # copy if extraction is unavailable. See reconstruction/outer_surface.py.
    if getattr(cfg, "preop_outer_surface", True):
        try:
            from .outer_surface import ensure_outer_mesh

            outer = ensure_outer_mesh(specimen_id, output_anatomy,
                                      dataset_root=cfg.dataset_root, log=log)
            shutil.copyfile(outer, dst)
            log(f"  wrote outer-surface preop mesh -> {dst}")
            return dst
        except Exception as exc:  # noqa: BLE001
            log(f"  [warn] outer-surface extraction failed ({exc}); copying raw CT mesh")

    shutil.copyfile(src, dst)
    log(f"  copied preop mesh -> {dst}")
    return dst
