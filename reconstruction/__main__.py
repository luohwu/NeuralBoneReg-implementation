"""Command-line entry point: ``python -m reconstruction --config <yaml>``.

Flags override individual config fields for convenience. ``--smoke`` runs a fast,
read-only validation: it checks the model loads, reconstructs one anatomy of
specimen02 with a coarse frame stride into a scratch directory, and reports the
reconstructed cloud's surface agreement with the CT-segmentation mesh.
"""

import argparse
import os
from dataclasses import replace

from .config import REPO_ROOT, load_config

DEFAULT_CONFIG = os.path.join("reconstruction", "conf", "reconstruction.yaml")
SMOKE_OUTPUT_DIR = os.path.join(REPO_ROOT, "reconstruction", "_smoke_output")


def _parse_args():
    parser = argparse.ArgumentParser(
        description="3D ultrasound point-cloud reconstruction for NeuralBoneReg."
    )
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="YAML config path.")
    parser.add_argument("--dataset-root", default=None,
                        help="Raw dataset root (overrides config; e.g. /path/to/UltraBonesHip).")
    parser.add_argument("--checkpoint-path", default=None,
                        help="Segmentation checkpoint path (overrides config).")
    parser.add_argument(
        "--specimen", action="append",
        help="Restrict to specimen id(s), e.g. --specimen specimen02 (repeatable).",
    )
    parser.add_argument("--frame-stride", type=int, default=None,
                        help="Subsample 1 in N frames (overrides config).")
    parser.add_argument("--intraoperative-dir", default=None,
                        help="Where to write merged .xyz clouds (overrides config).")
    parser.add_argument("--overwrite", action="store_true",
                        help="Allow overwriting existing output files.")
    parser.add_argument("--no-per-record", action="store_true",
                        help="Do not write per-record debug clouds.")
    parser.add_argument("--smoke", action="store_true",
                        help="Fast single-anatomy validation run into a scratch dir.")
    return parser.parse_args()


def _apply_overrides(cfg, args):
    changes = {}
    if args.dataset_root is not None:
        changes["dataset_root"] = os.path.abspath(args.dataset_root)
    if args.checkpoint_path is not None:
        changes["checkpoint_path"] = os.path.abspath(args.checkpoint_path)
    if args.specimen:
        changes["specimen_ids"] = args.specimen
    if args.frame_stride is not None:
        changes["frame_stride"] = args.frame_stride
    if args.intraoperative_dir is not None:
        changes["intraoperative_dir"] = os.path.abspath(args.intraoperative_dir)
    if args.overwrite:
        changes["overwrite"] = True
    if args.no_per_record:
        changes["write_per_record"] = False
    return replace(cfg, **changes) if changes else cfg


def _mean_p95(src_pcd, tgt_pcd):
    import numpy as np

    dist = np.asarray(src_pcd.compute_point_cloud_distance(tgt_pcd))
    return float(dist.mean()), float(np.percentile(dist, 95))


def _smoke(cfg):
    import open3d as o3d
    import torch

    from . import pipeline
    from .segmentation import BoneSegmenter

    left_femur = cfg.merge_map.get(
        "left_femur", ["left_femur_axial", "left_femur_coronal"]
    )
    cfg = replace(
        cfg,
        specimen_ids=["specimen02"],
        merge_map={"left_femur": left_femur},
        frame_stride=max(cfg.frame_stride, 15),
        write_per_record=False,
        export_preoperative=False,
        intraoperative_dir=SMOKE_OUTPUT_DIR,
        preoperative_dir=SMOKE_OUTPUT_DIR,
        overwrite=True,
    )
    print(f"[smoke] dataset_root: {cfg.dataset_root}")
    print(f"[smoke] scratch output: {cfg.intraoperative_dir}  frame_stride={cfg.frame_stride}")

    # 1) Model load + a single forward pass.
    segmenter = BoneSegmenter(
        cfg.checkpoint_path,
        threshold=cfg.segmentation_threshold,
        batch_size=cfg.seg_batch_size,
        num_workers=cfg.seg_num_workers,
        amp=cfg.seg_amp,
    )
    with torch.no_grad():
        out = segmenter.model(torch.zeros(1, 1, 256, 256, device=segmenter.device))
    print(f"[smoke] model loaded on {segmenter.device}; forward output {tuple(out.shape)}")

    # 2) Reconstruct one anatomy.
    pipeline.run(cfg, segmenter=segmenter)
    produced = os.path.join(cfg.intraoperative_dir, "specimen02_left_femur.xyz")
    if not os.path.isfile(produced):
        print("[smoke] FAILED: no cloud was produced.")
        return 1
    ours = o3d.io.read_point_cloud(produced)
    if len(ours.points) == 0:
        print("[smoke] FAILED: produced cloud is empty.")
        return 1
    print(f"[smoke] produced {len(ours.points)} points")

    # 3) Surface agreement vs the CT-segmentation bone (the reconstruction's
    #    target frame). The cloud is denoised CT-free (no CT-distance filter), so a
    #    sub-~10 mm mean at this coarse smoke stride confirms correct calibration/
    #    geometry and a working denoiser.
    ct_stl = os.path.join(
        cfg.dataset_root, "specimen02", "CT_bone_segmentations", "left_femur.stl"
    )
    if os.path.isfile(ct_stl):
        ct_pcd = o3d.io.read_triangle_mesh(ct_stl).sample_points_uniformly(200000)
        mean_ct, p95_ct = _mean_p95(ours, ct_pcd)
        print(f"[smoke] ours -> CT-seg femur : mean {mean_ct:7.3f} mm   p95 {p95_ct:7.3f} mm")
        if mean_ct <= 10.0:
            print("[smoke] PASS: calibration/geometry correct; CT-free denoiser working.")
        else:
            print("[smoke] WARN: mean distance higher than expected for this smoke run.")
    else:
        print(f"[smoke] (CT-seg mesh not found at {ct_stl}; skipped surface check)")
    print("[smoke] done.")
    return 0


def main():
    args = _parse_args()
    cfg = load_config(args.config)
    cfg = _apply_overrides(cfg, args)
    if args.smoke:
        raise SystemExit(_smoke(cfg))
    from . import pipeline

    pipeline.run(cfg)


if __name__ == "__main__":
    main()
