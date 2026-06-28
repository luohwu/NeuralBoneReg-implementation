"""Ultrasound calibration matrices.

Ported from UltrasoundDatasetCollection's reconstruction script. Builds the two
transforms that map an ultrasound image pixel into the tracked ideal-probe frame:

    T_idealProbe_USimage : rigid ideal-probe <- US-image-plane transform
    T_scale              : pixel -> millimetre scaling with the image plane
                           origin recentred to the image centre

so that, with the per-frame tracking pose ``T_tracking``,

    xyz_world = T_tracking @ T_idealProbe_USimage @ T_scale @ [col+0.5, row+0.5, 0, 1]
"""

import numpy as np

from utilities.converter import vectorToMatrix


def construct_ultrasound_calibration_matrix(ultrasound_calibration_info):
    """Return ``(T_idealProbe_USimage, T_scale)`` for a calibration preset.

    Only ``image_plane_origin == 'center'`` is supported (the convention used by
    the UltraBonesHip / SpineUS acquisitions).
    """
    origin = ultrasound_calibration_info.get("image_plane_origin")
    if origin != "center":
        raise NotImplementedError(
            f"image_plane_origin={origin!r} is not supported; only 'center' is."
        )

    image_width = ultrasound_calibration_info["image_width"]
    image_height = ultrasound_calibration_info["image_height"]
    space_x = ultrasound_calibration_info["scale_x"]
    space_y = ultrasound_calibration_info["scale_y"]

    T_scale = np.eye(4)
    T_scale[0, 0] = space_x
    T_scale[1, 1] = space_y
    T_scale[0, 3] = -0.5 * image_width * space_x
    T_scale[1, 3] = -0.5 * image_height * space_y

    T_idealProbe_USimage = vectorToMatrix(
        ultrasound_calibration_info["translation"],
        ultrasound_calibration_info["euler_xyz"],
    )
    return T_idealProbe_USimage, T_scale
