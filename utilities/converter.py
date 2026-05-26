import copy
import math

import numpy as np
import open3d as o3d
import torch
from scipy.spatial.transform import Rotation as R

from utilities.OwnPytorch3d import quaternion_to_matrix


def filter_src_by_tgt_distance_o3d(src_pcd, tgt_pcd, dis_threshold):
    if not isinstance(src_pcd, o3d.geometry.PointCloud) or not isinstance(tgt_pcd, o3d.geometry.PointCloud):
        raise TypeError("src_pcd and tgt_pcd must be open3d.geometry.PointCloud")

    src = np.asarray(src_pcd.points)
    tgt = np.asarray(tgt_pcd.points)

    if tgt.shape[0] == 0:
        return o3d.geometry.PointCloud()

    kdtree = o3d.geometry.KDTreeFlann(tgt_pcd)
    nn_dist = np.empty((src.shape[0],), dtype=np.float64)

    for i, point in enumerate(src):
        _, _, d2 = kdtree.search_knn_vector_3d(point, 1)
        nn_dist[i] = math.sqrt(d2[0]) if len(d2) > 0 else np.inf

    src_filtered = o3d.geometry.PointCloud()
    mask = nn_dist <= float(dis_threshold)
    src_filtered.points = o3d.utility.Vector3dVector(src[mask])
    if src_pcd.has_colors():
        src_filtered.colors = o3d.utility.Vector3dVector(np.asarray(src_pcd.colors)[mask])
    if src_pcd.has_normals():
        src_filtered.normals = o3d.utility.Vector3dVector(np.asarray(src_pcd.normals)[mask])
    return src_filtered


def vectorToMatrix(t, rotation_vector, quat=False):
    transform = np.eye(4)
    if quat:
        transform[:3, :3] = quaternion_to_matrix(torch.tensor(rotation_vector)).cpu().numpy()
    else:
        transform[:3, :3] = R.from_euler("xyz", rotation_vector, degrees=True).as_matrix()
    transform[:3, 3] = np.asarray(t)
    return transform


def compute_RTE_RRE_pcds(T_est, T_gt, translation_scale, src_pcd_moved, tgt_pcd):
    R1, t1 = T_est[:3, :3], T_est[:3, 3]
    R2, t2 = T_gt[:3, :3], T_gt[:3, 3]

    rte = np.linalg.norm(t1 - t2)

    R_diff = np.dot(R1.T, R2)
    trace = np.trace(R_diff)
    rre = np.arccos(min(max((trace - 1) / 2, -1), 1))
    rre_degrees = np.degrees(rre)

    pcd_registered = copy.deepcopy(src_pcd_moved).transform(T_est)
    dis_src_2_tgt = np.asarray(pcd_registered.compute_point_cloud_distance(tgt_pcd)) * translation_scale
    dis_tgt_2_src = np.asarray(tgt_pcd.compute_point_cloud_distance(pcd_registered)) * translation_scale
    cd = 0.5 * (dis_tgt_2_src.mean() + dis_src_2_tgt.mean())
    hd95 = 0.5 * (np.percentile(dis_src_2_tgt, 95) + np.percentile(dis_tgt_2_src, 95))

    return rte * translation_scale, rre_degrees, cd, hd95, pcd_registered


def invert_transformation_matrix(transform):
    rotation = transform[:3, :3]
    translation = transform[:3, 3]

    rotation_inv = rotation.T
    translation_inv = -rotation_inv @ translation

    transform_inv = np.eye(4)
    transform_inv[:3, :3] = rotation_inv
    transform_inv[:3, 3] = translation_inv
    return transform_inv
