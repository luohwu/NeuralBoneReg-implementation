"""Configuration loading for the reconstruction pipeline.

A small dataclass mirrors the YAML in ``conf/reconstruction.yaml`` so the rest of
the package receives a typed object instead of a loose dict. Relative paths in
the config are resolved against the repository root (the parent of this package),
so the pipeline can be launched from anywhere.
"""

import os
from dataclasses import dataclass, field

import yaml

# Repo root = parent of the ``reconstruction`` package directory.
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Scan-direction suffixes appended to a base anatomy in the raw folder names,
# e.g. "left_femur_axial" -> base anatomy "left_femur".
SCAN_SUFFIXES = ("axial", "coronal")

# Default per-anatomy shadow-denoiser parameters (overridable in the YAML's
# ``shadow_params``). ``below_thr=None`` skips the shadow gate for that anatomy.
_SHADOW_DEFAULTS = dict(
    below_thr=0.35, below_depthnorm=None, dbscan=True, eps=2.0, min_points=10,
    keep_frac=0.0, keep_abs=0.01, sor_nb=20, sor_std=2.0,
)

# Multi-view-consensus denoiser parameters (the pelvis recipe; overridable in the
# YAML's ``consensus_params``). See filtering.denoise_with_consensus.
_CONSENSUS_DEFAULTS = dict(
    voxel=2.0, min_frames=8, eps=1.5, min_points=10, sor_nb=20, sor_std=1.5,
    balance=True,
)


def _default_num_workers():
    """Sensible default DataLoader worker count: parallel image loading is the win,
    but leave cores for the main process / GPU feeding (cap at 8)."""
    return min(8, max(1, (os.cpu_count() or 4) - 1))


def _resolve(path):
    """Absolute paths are kept; relative ones are taken from the repo root."""
    if path is None:
        return None
    path = os.path.expanduser(str(path))
    if os.path.isabs(path):
        return path
    return os.path.normpath(os.path.join(REPO_ROOT, path))


def base_anatomy(folder_name):
    """Strip a trailing scan-direction suffix: 'left_femur_axial' -> 'left_femur'."""
    for suffix in SCAN_SUFFIXES:
        token = f"_{suffix}"
        if folder_name.endswith(token):
            return folder_name[: -len(token)]
    return folder_name


@dataclass
class ReconstructionConfig:
    dataset_root: str
    specimen_ids: list
    checkpoint_path: str
    presets: dict
    specimen_to_preset_map: dict
    pose_csv_name: str
    segmentation_threshold: float
    seg_batch_size: int
    seg_num_workers: int
    seg_amp: bool
    postprocess: str
    mask_border_px: int
    keep_largest_components: int
    min_mask_pixels: int
    min_skeleton_pixels: int
    merge_map: dict
    record_selection: str
    raw_downsample: int
    merge_downsample: int
    # CT-free geometric denoiser (filtering.denoise_point_cloud), applied per merged
    # anatomy. No CT mesh is used -- the bone surface is recovered as the dominant
    # DBSCAN cluster(s), then statistical outlier removal cleans residual fuzz.
    denoise_enable: bool
    denoise_work_points: int
    denoise_dbscan_eps: float
    denoise_dbscan_min_points: int
    denoise_keep_frac: float
    denoise_keep_abs_frac: float
    sor_nb_neighbors: int
    sor_std_ratio: float
    # CT-free denoiser selection. "dbscan" = the geometric DBSCAN denoiser (default,
    # unchanged). "shadow" = the acoustic-shadow image-cue denoiser (per-anatomy);
    # needs the image-carrying inference path (see record.py).
    denoise_method: str
    shadow_below_gap: int
    shadow_below_height: int
    shadow_params: dict
    # Anatomies denoised by the multi-view-consensus recipe (distinct-frame voxel
    # consensus -> per-hemipelvis keep-largest -> SOR -> L/R balance) instead of the
    # shadow/DBSCAN denoiser. The pelvis off-bone noise is coherent and only
    # distinct-frame consensus removes it. See filtering.denoise_with_consensus.
    consensus_anatomies: list
    consensus_params: dict
    strict_image_size: bool
    write_per_record: bool
    frame_stride: int
    visualize: bool
    seed: int
    intraoperative_dir: str
    preoperative_dir: str
    export_preoperative: bool
    preop_outer_surface: bool
    overwrite: bool
    config_path: str = field(default="", compare=False)

    @property
    def source_folders(self):
        """All raw ultrasound_records sweep folders referenced by merge_map."""
        seen = []
        for folders in self.merge_map.values():
            for folder in folders:
                if folder not in seen:
                    seen.append(folder)
        return seen

    def preset_for(self, specimen_id):
        preset_name = self.specimen_to_preset_map[specimen_id]
        return self.presets[preset_name]

    def denoise_method_for(self, output_anatomy):
        """The denoiser for an anatomy: ``"consensus"`` for the configured
        ``consensus_anatomies`` (the pelvis), else the global ``denoise_method``."""
        if output_anatomy in self.consensus_anatomies:
            return "consensus"
        return self.denoise_method

    def consensus_params_for(self, output_anatomy):
        """Consensus-denoiser params (defaults filled from ``_CONSENSUS_DEFAULTS``)."""
        params = dict(_CONSENSUS_DEFAULTS)
        params.update(self.consensus_params)
        return params

    def shadow_params_for(self, output_anatomy, specimen_id=None):
        """Per-anatomy shadow-denoiser params. ``left_femur``/``right_femur`` map to the
        ``femur`` group; everything else falls back to its own key then ``default``.
        Missing fields are filled from ``_SHADOW_DEFAULTS``."""
        key = "femur" if output_anatomy in ("left_femur", "right_femur") else output_anatomy
        params = dict(_SHADOW_DEFAULTS)
        params.update(self.shadow_params.get(key, self.shadow_params.get("default", {})))
        return params


def load_config(config_path):
    """Load the reconstruction YAML and the calibration presets it references."""
    config_path = _resolve(config_path)
    with open(config_path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    preset_file = _resolve(raw["preset_calibration_file"])
    with open(preset_file, "r", encoding="utf-8") as f:
        presets = yaml.safe_load(f)
    return build_config(raw, presets, config_path)


def build_config(raw, presets, config_path=""):
    """Build a ``ReconstructionConfig`` from already-parsed ``raw`` config and
    ``presets`` dicts. ``load_config`` uses it after reading the YAML files; a
    self-contained script can call it directly to bake the config/calibration
    inline."""
    return ReconstructionConfig(
        dataset_root=_resolve(raw["dataset_root"]),
        specimen_ids=list(raw["specimen_ids"]),
        checkpoint_path=_resolve(raw["checkpoint_path"]),
        presets=presets,
        specimen_to_preset_map=dict(raw["specimen_to_preset_map"]),
        pose_csv_name=raw["pose_csv_name"],
        segmentation_threshold=float(raw["segmentation_threshold"]),
        seg_batch_size=int(raw.get("seg_batch_size") or 16),
        seg_num_workers=(
            int(raw["seg_num_workers"])
            if raw.get("seg_num_workers") is not None
            else _default_num_workers()
        ),
        seg_amp=bool(raw.get("seg_amp", False)),
        postprocess=raw.get("postprocess", "skeleton_only"),
        mask_border_px=int(raw.get("mask_border_px", 10)),
        keep_largest_components=int(raw.get("keep_largest_components", 1)),
        min_mask_pixels=int(raw.get("min_mask_pixels", 20)),
        min_skeleton_pixels=int(raw.get("min_skeleton_pixels", 5)),
        merge_map={k: list(v) for k, v in raw["merge_map"].items()},
        record_selection=raw.get("record_selection", "all"),
        raw_downsample=int(raw["raw_downsample"]),
        merge_downsample=int(raw["merge_downsample"]),
        denoise_enable=bool(raw.get("denoise_enable", True)),
        denoise_work_points=int(raw.get("denoise_work_points", 200000)),
        denoise_dbscan_eps=float(raw.get("denoise_dbscan_eps", 1.5)),
        denoise_dbscan_min_points=int(raw.get("denoise_dbscan_min_points", 10)),
        denoise_keep_frac=float(raw.get("denoise_keep_frac", 0.10)),
        denoise_keep_abs_frac=float(raw.get("denoise_keep_abs_frac", 0.01)),
        sor_nb_neighbors=int(raw.get("sor_nb_neighbors", 20)),
        sor_std_ratio=float(raw.get("sor_std_ratio", 1.5)),
        denoise_method=str(raw.get("denoise_method", "dbscan")),
        shadow_below_gap=int(raw.get("shadow_below_gap", 4)),
        shadow_below_height=int(raw.get("shadow_below_height", 60)),
        shadow_params={k: dict(v) for k, v in (raw.get("shadow_params") or {}).items()},
        consensus_anatomies=list(raw.get("consensus_anatomies") or ["pelvis"]),
        consensus_params=dict(raw.get("consensus_params") or {}),
        strict_image_size=bool(raw.get("strict_image_size", True)),
        write_per_record=bool(raw.get("write_per_record", True)),
        frame_stride=int(raw.get("frame_stride", 1)),
        visualize=bool(raw.get("visualize", False)),
        seed=int(raw.get("seed", 42)),
        intraoperative_dir=_resolve(raw["intraoperative_dir"]),
        preoperative_dir=_resolve(raw["preoperative_dir"]),
        export_preoperative=bool(raw.get("export_preoperative", True)),
        preop_outer_surface=bool(raw.get("preop_outer_surface", True)),
        overwrite=bool(raw.get("overwrite", False)),
        config_path=config_path,
    )
