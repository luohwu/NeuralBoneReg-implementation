"""Reconstruct one ultrasound record into a 3D point cloud.

Runs the segmentation network on every frame, projects bone-surface pixels into
world millimetres, accumulates them, downsamples, filters to the target anatomy
against the CT segmentation, and removes statistical outliers.

Per-record clouds (when enabled) are written under a ``neuralbonereg`` subfolder
so the dataset's existing ``with_pred_labels`` reference clouds are never
overwritten.
"""

import os

import numpy as np
import open3d as o3d
import pandas as pd
from tqdm import tqdm

from utilities.converter import vectorToMatrix

from .filtering import downsample_to_n_random
from .projection import (
    pixels_to_world,
    pixels_to_world_with_shadow,
    postprocess_mask,
)

PER_RECORD_SUBDIR = os.path.join("3D_reconstructions", "neuralbonereg")


def _make_pcd(points, below=None, depth=None, frame_id=None):
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np.asarray(points, dtype=np.float64))
    if below is not None or frame_id is not None:
        # Carry per-point denoising features through the o3d pipeline in the
        # (otherwise unused) colours channel, so the downsample/merge steps preserve
        # them and the denoiser reads them back at merge time:
        #   R = below-window intensity (acoustic shadow, 0..1; femur shadow gate)
        #   G = normalised image depth (row/height, for the depth-normalized shadow)
        #   B = per-record frame id (the pelvis multi-view-consensus denoiser; an
        #       integer, NOT clipped -- it is globalised across records at merge).
        n = len(points)
        below_col = (np.clip(np.asarray(below, dtype=np.float64).reshape(-1, 1), 0.0, 1.0)
                     if below is not None else np.zeros((n, 1)))
        depth_col = (np.clip(np.asarray(depth, dtype=np.float64).reshape(-1, 1), 0.0, 1.0)
                     if depth is not None else np.zeros((n, 1)))
        frame_col = (np.asarray(frame_id, dtype=np.float64).reshape(-1, 1)
                     if frame_id is not None else np.zeros((n, 1)))
        colors = np.concatenate([below_col, depth_col, frame_col], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def _image_name(row):
    file_path = getattr(row, "file_path", None)
    if isinstance(file_path, str) and file_path:
        return file_path
    return f"{int(row.timestamp)}.png"


def reconstruct_one_record(
    record_folder,
    segmenter,
    T_idealProbe_USimage,
    T_scale,
    preset,
    cfg,
    log=print,
):
    """Return the raw reconstructed ``PointCloud`` for the record, or ``None``.

    No CT mesh is used here: every usable frame contributes points. Spurious
    points from frames that missed bone are removed later, geometrically and
    CT-free, when the per-record clouds are merged per anatomy (see ``merge.py`` /
    ``filtering.denoise_point_cloud``).
    """
    csv_path = os.path.join(record_folder, cfg.pose_csv_name)
    if not os.path.isfile(csv_path):
        log(f"    [skip] missing pose CSV: {csv_path}")
        return None

    df = pd.read_csv(csv_path)
    if df.empty:
        log(f"    [skip] empty pose CSV: {csv_path}")
        return None
    if cfg.frame_stride > 1:
        df = df.iloc[:: cfg.frame_stride]

    us_folder = os.path.join(record_folder, "UltrasoundImages")
    preset_h, preset_w = int(preset["image_height"]), int(preset["image_width"])

    # Build the frame list up front (a cheap stat per frame). Segmentation then
    # runs over all frames at once: images are loaded/preprocessed in parallel
    # worker processes and the GPU forward pass is batched (the real bottleneck
    # is disk I/O, not the GPU), which is far faster than one frame at a time.
    rows = list(df.itertuples(index=False))
    valid_rows, valid_paths = [], []
    for row in rows:
        img_path = os.path.join(us_folder, _image_name(row))
        if os.path.isfile(img_path):
            valid_rows.append(row)
            valid_paths.append(img_path)
    n_seen = len(valid_paths)

    # The shadow denoiser needs native image intensities (to measure the acoustic
    # shadow below each bone pixel), so it uses the heavier image-carrying inference
    # path. The default DBSCAN denoiser uses the lighter mask-only path unchanged.
    shadow_mode = getattr(cfg, "denoise_method", "dbscan") == "shadow"

    point_chunks = []
    below_chunks = []
    depth_chunks = []
    frame_chunks = []
    n_used = 0
    n_wrong_size = 0
    if shadow_mode:
        frames = segmenter.predict_masks_images(valid_paths)
    else:
        frames = segmenter.predict_masks(valid_paths)
    masks = tqdm(
        frames,
        total=n_seen,
        desc=f"    {os.path.basename(record_folder)}",
        unit="frame",
        leave=False,
    )
    for item in masks:
        if shadow_mode:
            idx, mask, native = item
        else:
            idx, mask = item
            native = None
        if cfg.strict_image_size and mask.shape != (preset_h, preset_w):
            n_wrong_size += 1
            continue

        skeleton = postprocess_mask(mask, cfg)
        if skeleton is None:
            continue

        row = valid_rows[idx]
        T_tracking = vectorToMatrix(
            [row.x, row.y, row.z], [row.euler_x, row.euler_y, row.euler_z]
        )
        if shadow_mode:
            result = pixels_to_world_with_shadow(
                skeleton, native, T_tracking, T_idealProbe_USimage, T_scale,
                gap=cfg.shadow_below_gap, height=cfg.shadow_below_height,
            )
            if result is None:
                continue
            points, below, depth = result
            below_chunks.append(below)
            depth_chunks.append(depth)
        else:
            points = pixels_to_world(
                skeleton, T_tracking, T_idealProbe_USimage, T_scale
            )
            if points is None:
                continue
        point_chunks.append(points)
        # Per-record frame id (one per USED frame), carried per point for the
        # pelvis multi-view-consensus denoiser; globalised across records at merge.
        frame_chunks.append(np.full(len(points), n_used, dtype=np.float64))
        n_used += 1

    if n_wrong_size:
        log(
            f"    [skip] {n_wrong_size} frame(s) with size != preset "
            f"({preset_h}, {preset_w}); set strict_image_size: false to allow"
        )

    if not point_chunks:
        log(f"    [skip] no usable frames in {record_folder}")
        return None

    below_all = np.concatenate(below_chunks) if shadow_mode and below_chunks else None
    depth_all = np.concatenate(depth_chunks) if shadow_mode and depth_chunks else None
    frame_all = np.concatenate(frame_chunks) if frame_chunks else None
    raw = _make_pcd(np.concatenate(point_chunks, axis=0), below=below_all,
                    depth=depth_all, frame_id=frame_all)
    raw = downsample_to_n_random(raw, cfg.raw_downsample, seed=cfg.seed)

    log(f"    frames {n_used}/{n_seen} used; raw {len(raw.points)} points")

    if cfg.write_per_record:
        out_dir = os.path.join(record_folder, PER_RECORD_SUBDIR)
        os.makedirs(out_dir, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(out_dir, "reconstruction_pcd.xyz"), raw)

    return raw
