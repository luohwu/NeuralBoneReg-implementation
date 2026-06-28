"""Mask post-processing and pixel -> world projection.

The bone surface is taken to be the mid-line of the segmentation mask, so masks
are skeletonized before projection (this also greatly reduces point count). Two
modes are offered:

* ``skeleton_only`` (default) -- binarise (> 0) then skeletonize. Matches the
  reconstruction script that produced the shipped clouds.
* ``icp_clean`` -- additionally zero a border margin and keep the largest
  connected component(s) before skeletonizing (from the ICP refinement script).

Projection applies the per-frame tracking pose and the calibration transforms to
each bone pixel ``(row, col)`` using the pixel-centre convention ``+0.5``.
"""

import numpy as np
from skimage.morphology import skeletonize


def _keep_top_k_largest_components(binary_mask, k):
    """Keep the ``k`` connected components with the largest pixel mass."""
    import cv2

    binary_mask = (binary_mask > 0).astype(np.uint8)
    num_labels, labels = cv2.connectedComponents(binary_mask)
    if num_labels <= 1:
        return np.zeros_like(binary_mask)

    masses = {
        label: int((labels == label).sum()) for label in range(1, num_labels)
    }
    keep = sorted(masses, key=masses.get, reverse=True)[:k]
    out = np.zeros_like(binary_mask)
    for label in keep:
        out[labels == label] = 1
    return out


def postprocess_mask(mask, cfg):
    """Return the bone-surface skeleton as a boolean array, or ``None``.

    ``None`` means the frame yields no usable surface (empty / degenerate mask).
    """
    if mask is None:
        return None

    binary = mask > 0
    if not binary.any():
        return None

    if cfg.postprocess == "icp_clean":
        thick = (binary.astype(np.uint8)) * 255
        b = int(cfg.mask_border_px)
        if b > 0:
            thick[:b, :] = 0
            thick[-b:, :] = 0
            thick[:, :b] = 0
            thick[:, -b:] = 0
        if int(np.count_nonzero(thick)) < cfg.min_mask_pixels:
            return None
        thick = _keep_top_k_largest_components(thick, cfg.keep_largest_components)
        if int(np.count_nonzero(thick)) < cfg.min_mask_pixels:
            return None
        binary = thick > 0

    skeleton = skeletonize(binary)
    nonzero = int(np.count_nonzero(skeleton))
    if cfg.postprocess == "icp_clean":
        if nonzero < cfg.min_skeleton_pixels:
            return None
    elif nonzero < 2:  # mirror the source guard (len(rows_index) < 2)
        return None
    return skeleton


def pixels_to_world(skeleton, T_tracking, T_idealProbe_USimage, T_scale):
    """Project skeleton pixels to world (CT) millimetres -> ``(N, 3)`` array."""
    rows_index, cols_index = np.where(skeleton)
    if rows_index.size < 2:
        return None

    n = rows_index.size
    pixels = np.stack(
        [
            cols_index + 0.5,
            rows_index + 0.5,
            np.zeros(n, dtype=np.float64),
            np.ones(n, dtype=np.float64),
        ],
        axis=0,
    )  # 4 x N homogeneous [x, y, 0, 1]
    transform = T_tracking @ T_idealProbe_USimage @ T_scale
    xyz = (transform @ pixels)[:3, :].T
    return xyz


def shadow_below(native_image, rows_index, cols_index, gap=4, height=60):
    """Mean image intensity (0..1) in a window ``height`` px *below* each pixel.

    The acoustic-shadow cue: a true bone reflector blocks the beam, leaving a dark
    region directly beneath it, whereas soft-tissue / reverberation artefacts do
    not. Returns a float array in [0, 1] (lower = stronger shadow = more bone-like).
    Uses a per-column cumulative sum so it is O(N) over the skeleton points.
    """
    img = native_image.astype(np.float32)
    if img.max() > 1.0:
        img = img / 255.0
    h, w = img.shape
    csum = np.zeros((h + 1, w), dtype=np.float64)
    np.cumsum(img, axis=0, out=csum[1:])
    a = np.clip(rows_index + gap, 0, h)
    b = np.clip(rows_index + gap + height, 0, h)
    n = np.maximum(b - a, 1)
    vals = (csum[b, cols_index] - csum[a, cols_index]) / n
    out = vals.astype(np.float32)
    out[(b - a) <= 0] = 0.5  # window off the image bottom -> neutral
    return out


def pixels_to_world_with_shadow(skeleton, native_image, T_tracking,
                                T_idealProbe_USimage, T_scale, gap=4, height=60):
    """Project skeleton pixels and also return per-point shadow + depth features.

    Returns ``(xyz (N,3), below (N,), depth (N,))`` or ``None``:
    * ``below`` -- mean intensity in a window beneath each point (acoustic shadow;
      see :func:`shadow_below`); lower = stronger shadow = more bone-like.
    * ``depth`` -- the source pixel row normalised to [0, 1] by image height. Needed
      by the *depth-normalized* shadow (pelvis), which compares ``below`` to the
      per-depth median to separate a genuine shadow from depth attenuation."""
    rows_index, cols_index = np.where(skeleton)
    if rows_index.size < 2:
        return None
    xyz = pixels_to_world(skeleton, T_tracking, T_idealProbe_USimage, T_scale)
    below = shadow_below(native_image, rows_index, cols_index, gap=gap, height=height)
    depth = (rows_index.astype(np.float32) / max(1, native_image.shape[0]))
    return xyz, below, depth
