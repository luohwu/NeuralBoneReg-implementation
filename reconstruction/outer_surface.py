"""UltraBonesHip outer-surface helper (path-aware wrapper over the generic extractor).

The dataset-agnostic ray-cast extraction now lives in ``utilities/outer_surface.py`` so
the registration core can apply it to any dataset. This module keeps the
UltraBonesHip-specific ``ensure_outer_mesh(specimen, anatomy)`` that resolves the raw CT
segmentation under the mounted dataset and caches the outer shell, re-using the generic
functions. Existing importers (e.g. ``reconstruction.merge``) are unchanged.
"""

import os

import open3d as o3d

# Re-export the generic helpers so existing `from reconstruction.outer_surface import ...`
# call sites keep working.
from utilities.outer_surface import (  # noqa: F401
    clean_triangle_mesh,
    extract_outer_surface_mesh,
    extract_outer_surface_point_cloud,
    fast_downsample_point_cloud,
    sample_fibonacci_sphere,
)

DATASET_ROOT = "/mnt/UltraBonesHip"
_CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_denoise_cache", "outer_meshes")


def ensure_outer_mesh(specimen, anatomy, dataset_root=DATASET_ROOT, cache_dir=_CACHE,
                      num_viewpoints=128, image_size=384, log=print):
    """Return the path to the cached outer-surface .stl for (specimen, anatomy),
    generating it from the CT segmentation mesh on first use. Deterministic."""
    os.makedirs(cache_dir, exist_ok=True)
    out = os.path.join(cache_dir, f"{specimen}_{anatomy}.stl")
    if os.path.isfile(out):
        return out
    src = os.path.join(dataset_root, specimen, "CT_bone_segmentations", f"{anatomy}.stl")
    if not os.path.isfile(src):
        raise FileNotFoundError(f"CT mesh not found: {src}")
    mesh = o3d.io.read_triangle_mesh(src)
    outer = extract_outer_surface_mesh(mesh, num_viewpoints=num_viewpoints, image_size=image_size)
    o3d.io.write_triangle_mesh(out, outer)
    n_in = len(mesh.triangles)
    n_out = len(outer.triangles)
    log(f"  outer surface {specimen}_{anatomy}: {n_in} -> {n_out} tris "
        f"({100*n_out/max(1,n_in):.0f}% kept) -> {out}")
    return out


if __name__ == "__main__":
    import sys

    specs = [f"specimen{i:02d}" for i in range(5)]
    anatomies = ["left_femur", "right_femur", "pelvis"]
    if len(sys.argv) > 1:
        specs = [sys.argv[1]]
    for sp in specs:
        for an in anatomies:
            try:
                ensure_outer_mesh(sp, an)
            except Exception as exc:  # noqa: BLE001
                print(f"  [skip] {sp} {an}: {exc}")
    print("done")
