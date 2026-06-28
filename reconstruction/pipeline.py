"""Reconstruction orchestration.

Loops specimens x sweep folders x records, reconstructs each record, and merges
the filtered per-record clouds into NeuralBoneReg per-anatomy files. The
segmentation model is loaded once and used from the main process; within each
record frames are loaded in parallel worker processes and segmented in batches
(disk I/O dominates, so this is the main throughput lever -- see record.py).
"""

import os
import random

import numpy as np

from .calibration import construct_ultrasound_calibration_matrix
from .merge import export_preoperative_mesh, merge_anatomy
from .record import reconstruct_one_record
from .segmentation import BoneSegmenter


def _seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except ImportError:
        pass


def _list_records(folder_path):
    if not os.path.isdir(folder_path):
        return []
    return sorted(
        name
        for name in os.listdir(folder_path)
        if name.startswith("record")
        and os.path.isdir(os.path.join(folder_path, name))
    )


def run(cfg, log=print, segmenter=None):
    """Run the full reconstruction pipeline; returns the list of written paths.

    A preloaded ``BoneSegmenter`` may be passed in (e.g. by the smoke test) to
    avoid loading the checkpoint twice.
    """
    _seed_everything(cfg.seed)
    if segmenter is None:
        segmenter = BoneSegmenter(
            cfg.checkpoint_path,
            threshold=cfg.segmentation_threshold,
            batch_size=cfg.seg_batch_size,
            num_workers=cfg.seg_num_workers,
            amp=cfg.seg_amp,
        )
    log(f"Loaded segmentation model on {segmenter.device}")

    written = []
    for specimen_id in cfg.specimen_ids:
        specimen_folder = os.path.join(cfg.dataset_root, specimen_id)
        if not os.path.isdir(specimen_folder):
            log(f"[skip specimen] missing folder: {specimen_folder}")
            continue

        log(f"\n=== {specimen_id} ===")
        preset = cfg.preset_for(specimen_id)
        T_idealProbe_USimage, T_scale = construct_ultrasound_calibration_matrix(preset)

        records_root = os.path.join(specimen_folder, "ultrasound_records")
        raw_by_folder = {}
        for folder in cfg.source_folders:
            folder_path = os.path.join(records_root, folder)
            records = _list_records(folder_path)
            if not records:
                log(f"  [skip folder] no records in {folder_path}")
                raw_by_folder[folder] = []
                continue

            clouds = []
            for record_name in records:
                log(f"  {folder}/{record_name}")
                raw = reconstruct_one_record(
                    os.path.join(folder_path, record_name),
                    segmenter,
                    T_idealProbe_USimage,
                    T_scale,
                    preset,
                    cfg,
                    log=log,
                )
                if raw is not None and len(raw.points) > 0:
                    clouds.append(raw)
            raw_by_folder[folder] = clouds

        for output_anatomy, source_folders in cfg.merge_map.items():
            clouds, folders = [], []
            for folder in source_folders:
                fc = raw_by_folder.get(folder, [])
                clouds.extend(fc)
                folders.extend([folder] * len(fc))  # per-cloud folder (for side + frame id)
            path = merge_anatomy(clouds, output_anatomy, specimen_id, cfg, log=log,
                                 folders=folders)
            if path is not None:
                written.append(path)
                if cfg.export_preoperative:
                    export_preoperative_mesh(specimen_id, output_anatomy, cfg, log=log)

    log(f"\nDone. {len(written)} merged anatomy file(s) considered.")
    for path in written:
        log(f"  {path}")
    return written
