"""Extract the externally-visible (outer) surface of a CT bone mesh.

An intra-operative sensor (ultrasound, RGB-D, ...) can only observe the **outer** bone
surface, but a CT bone-segmentation mesh often carries internal structure (inner cortical
wall, medullary canal, non-manifold inner faces). Training the registration target
(NeuralUDF) on those internal surfaces is wrong -- the intra-operative cloud can never
match them. This module ray-casts the mesh from many viewpoints on a surrounding sphere
and keeps only the triangles a camera can actually see, yielding the outer shell.

Dataset-agnostic: it operates on any mesh, so the registration core applies it to the
preoperative model of any dataset. The UltraBonesHip-specific path helper
(``ensure_outer_mesh(specimen, anatomy)``) lives in ``reconstruction/outer_surface.py``
and re-uses the functions here.
"""

import os

import numpy as np
import open3d as o3d


def fast_downsample_point_cloud(pcd, target_num_points):
    points = np.asarray(pcd.points)
    if len(points) <= target_num_points:
        return pcd

    sampled_idx = np.random.choice(len(points), size=target_num_points, replace=False)
    pcd_downsampled = o3d.geometry.PointCloud()
    pcd_downsampled.points = o3d.utility.Vector3dVector(points[sampled_idx])

    if pcd.has_colors():
        pcd_downsampled.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[sampled_idx])
    if pcd.has_normals():
        pcd_downsampled.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[sampled_idx])

    return pcd_downsampled


def sample_fibonacci_sphere(num_points):
    directions = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for i in range(num_points):
        y = 1.0 - (2.0 * i) / max(num_points - 1, 1)
        radius = np.sqrt(max(0.0, 1.0 - y * y))
        theta = golden_angle * i
        directions.append([np.cos(theta) * radius, y, np.sin(theta) * radius])
    return np.asarray(directions, dtype=np.float64)


def clean_triangle_mesh(mesh):
    mesh = o3d.geometry.TriangleMesh(mesh)
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.remove_degenerate_triangles()
    mesh.compute_triangle_normals()
    mesh.compute_vertex_normals()
    return mesh


def _safe_camera_up(direction):
    up = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(np.dot(direction, up)) > 0.9:
        up = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return up


def _raycast_outer_triangle_ids(mesh, num_viewpoints=128, image_size=384, fov_deg=30.0):
    mesh = clean_triangle_mesh(mesh)
    tmesh = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(tmesh)

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center(), dtype=np.float32)
    extent = np.asarray(bbox.get_extent(), dtype=np.float32)
    diameter = float(np.linalg.norm(extent))
    camera_distance = diameter * 2.5

    hit_triangle_ids = set()
    center_tensor = o3d.core.Tensor(center, dtype=o3d.core.Dtype.Float32)
    for direction in sample_fibonacci_sphere(num_viewpoints).astype(np.float32):
        eye_tensor = o3d.core.Tensor(center + direction * camera_distance, dtype=o3d.core.Dtype.Float32)
        up_tensor = o3d.core.Tensor(_safe_camera_up(direction), dtype=o3d.core.Dtype.Float32)
        rays = scene.create_rays_pinhole(
            fov_deg,
            center_tensor,
            eye_tensor,
            up_tensor,
            image_size,
            image_size,
        )
        ray_hits = scene.cast_rays(rays)
        primitive_ids = ray_hits["primitive_ids"].numpy().reshape(-1)
        valid_ids = primitive_ids[primitive_ids != np.iinfo(np.uint32).max]
        hit_triangle_ids.update(valid_ids.tolist())

    return mesh, sorted(hit_triangle_ids)


def extract_outer_surface_mesh(mesh, num_viewpoints=128, image_size=384):
    """Return the outer-shell mesh (triangles visible from a surrounding sphere).

    Deterministic -- the ray-cast uses a fixed Fibonacci viewpoint set, no RNG.
    """
    cleaned_mesh, triangle_ids = _raycast_outer_triangle_ids(
        mesh, num_viewpoints=num_viewpoints, image_size=image_size,
    )
    if not triangle_ids:
        raise RuntimeError("Ray casting did not hit any outer triangles.")
    outer_mesh = o3d.geometry.TriangleMesh(cleaned_mesh)
    triangle_mask = np.ones(len(np.asarray(outer_mesh.triangles)), dtype=bool)
    triangle_mask[np.asarray(triangle_ids, dtype=np.int64)] = False
    outer_mesh.remove_triangles_by_mask(triangle_mask)
    outer_mesh.remove_unreferenced_vertices()
    outer_mesh.compute_triangle_normals()
    outer_mesh.compute_vertex_normals()
    return outer_mesh


def extract_outer_surface_point_cloud(mesh, num_viewpoints=128, image_size=384, target_num_points=40000):
    outer_mesh = extract_outer_surface_mesh(mesh, num_viewpoints, image_size)
    outer_pcd = outer_mesh.sample_points_uniformly(max(target_num_points * 2, 80000))
    return fast_downsample_point_cloud(outer_pcd, target_num_points)


def ensure_outer_surface_stl(src_stl, dst_stl, num_viewpoints=128, image_size=384, log=print):
    """Generate (once) and cache the outer-surface ``.stl`` of ``src_stl`` at ``dst_stl``.

    Returns ``dst_stl`` on success. Deterministic, so the cache is reused on later runs.
    Raises if ``src_stl`` is missing or the mesh has no visible triangles.
    """
    if os.path.isfile(dst_stl):
        return dst_stl
    if not os.path.isfile(src_stl):
        raise FileNotFoundError(f"source mesh not found: {src_stl}")
    os.makedirs(os.path.dirname(os.path.abspath(dst_stl)), exist_ok=True)
    mesh = o3d.io.read_triangle_mesh(src_stl)
    outer = extract_outer_surface_mesh(mesh, num_viewpoints=num_viewpoints, image_size=image_size)
    o3d.io.write_triangle_mesh(dst_stl, outer)
    n_in, n_out = len(mesh.triangles), len(outer.triangles)
    log(f"  outer surface {os.path.basename(src_stl)}: {n_in} -> {n_out} tris "
        f"({100 * n_out / max(1, n_in):.0f}% kept) -> {dst_stl}")
    return dst_stl
