"""3D ultrasound point-cloud reconstruction for NeuralBoneReg.

Rebuilds the intraoperative point clouds consumed by ``main_UltraBonesHip.py`` directly from
raw UltraBonesHip data (ultrasound images + tracked poses + a calibration
preset). Each frame is bone-segmented with a pretrained network, skeletonized to
the bone mid-line, and projected into CT/world millimetres via

    xyz_world = T_tracking @ T_idealProbe_USimage @ T_scale @ [col+0.5, row+0.5, 0, 1]

Per-record clouds are filtered to the target anatomy against the CT segmentation
and merged per anatomy into ``specimenNN_<anatomy>.xyz``.

This is a self-contained *live* pipeline (segmentation is run on every frame); it
is not a byte-for-byte reproduction of the shipped clouds. See ``README.md``.
"""

from .calibration import construct_ultrasound_calibration_matrix
from .config import ReconstructionConfig, load_config

__all__ = [
    "ReconstructionConfig",
    "load_config",
    "construct_ultrasound_calibration_matrix",
]
