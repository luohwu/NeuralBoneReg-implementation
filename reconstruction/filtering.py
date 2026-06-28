"""Point-cloud denoising and downsampling.

The reconstruction accumulates points from *all* frames with raw tracking poses,
so frames that did not actually image bone contribute spurious points (often far
from the true surface). These are removed **without any reference to the CT mesh**
(the registration target) by a purely geometric denoiser:

* ``downsample_to_n_random`` -- exact-size random subsample (seeded, reproducible).
* ``denoise_point_cloud`` -- DBSCAN cluster keep (drop small/disconnected noise
  blobs, keep the dominant bone surface region(s)) followed by statistical outlier
  removal. CT-free.
* ``remove_outliers_statistical`` -- thin wrapper over Open3D statistical outlier
  removal.
"""

import numpy as np
import open3d as o3d


def downsample_to_n_random(pcd, target_n, seed=42):
    """Return a cloud with exactly ``target_n`` points (random, no replacement).

    If the cloud already has ``<= target_n`` points it is returned unchanged.
    Colours/normals are preserved if present.
    """
    if target_n <= 0:
        raise ValueError("target_n must be > 0")

    n = len(pcd.points)
    if n <= target_n:
        return o3d.geometry.PointCloud(pcd)

    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=int(target_n), replace=False)

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(np.asarray(pcd.points)[idx])
    if pcd.has_colors():
        out.colors = o3d.utility.Vector3dVector(np.asarray(pcd.colors)[idx])
    if pcd.has_normals():
        out.normals = o3d.utility.Vector3dVector(np.asarray(pcd.normals)[idx])
    return out


def remove_outliers_statistical(pcd, nb_neighbors=20, std_ratio=2.0):
    """Statistical outlier removal; returns the filtered cloud."""
    if len(pcd.points) == 0:
        return pcd
    filtered, _ = pcd.remove_statistical_outlier(
        nb_neighbors=max(1, int(nb_neighbors)), std_ratio=float(std_ratio)
    )
    return filtered


def _depthnorm_shadow_keep(below, depth, delta=0.0, nbins=24):
    """Depth-normalized acoustic shadow keep-mask: keep points whose below-window
    intensity is darker than the per-depth median (+ ``delta``). Isolates a genuine
    shadow from mere depth attenuation -- the cue that works on the pelvis, where
    off-bone noise is also deep/dark."""
    below = np.asarray(below)
    depth = np.asarray(depth)
    lo, hi = float(depth.min()), float(depth.max()) + 1e-6
    bins = np.clip(((depth - lo) / (hi - lo) * nbins).astype(int), 0, nbins - 1)
    med = np.zeros(nbins)
    for b in range(nbins):
        m = bins == b
        med[b] = np.median(below[m]) if m.any() else 0.0
    return below <= med[bins] + float(delta)


def denoise_with_shadow(
    pcd,
    below,
    depth=None,
    work_points=200000,
    below_thr=0.35,
    below_depthnorm=None,
    dbscan=True,
    dbscan_eps=2.0,
    dbscan_min_points=10,
    keep_frac=0.0,
    keep_abs_frac=0.01,
    sor_nb_neighbors=20,
    sor_std_ratio=2.0,
    seed=42,
    log=None,
):
    """CT-free **acoustic-shadow** denoiser (per-point image cue + connectivity).

    A true bone reflector casts an acoustic shadow: the image region directly
    *below* a bone pixel is dark. Reverberation / soft-tissue false detections are
    bright below. ``below`` is the per-point mean intensity in a window beneath the
    source pixel (0..1; computed at projection time, CT-free). Two shadow gates:

    * **Absolute** (``below_thr``, femur): keep ``below <= below_thr``.
    * **Depth-normalized** (``below_depthnorm`` = delta, pelvis): keep points darker
      than the per-depth median ``below`` (needs ``depth`` = per-point normalised
      image row). Use this where off-bone noise is itself deep/dark and an absolute
      threshold fails (the pelvis).

    The kept points then optionally go through DBSCAN with ``keep_frac=0`` (keep
    every cluster above an absolute floor -- preserves the full bone extent/coverage
    while dropping tiny off-bone blobs) and a final statistical outlier removal.
    On the pelvis DBSCAN craters coverage, so ``dbscan=False`` (SOR only) is used
    there. Set both gates to ``None`` to skip shadow filtering. Returns a new
    ``PointCloud``.
    """
    pts = np.asarray(pcd.points)
    below = np.asarray(below)
    n_in = len(pts)
    if n_in == 0:
        return o3d.geometry.PointCloud(pcd)

    gate = None
    if below_depthnorm is not None and depth is not None and len(depth) == n_in:
        gate = "depthnorm"
        keep = _depthnorm_shadow_keep(below, depth, delta=float(below_depthnorm))
        pts = pts[keep]
    elif below_thr is not None and len(below) == n_in:
        gate = f"below<{below_thr}"
        pts = pts[below <= float(below_thr)]

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(pts)
    if dbscan:
        out = denoise_point_cloud(
            out, work_points=work_points, dbscan_eps=dbscan_eps,
            dbscan_min_points=dbscan_min_points, keep_frac=keep_frac,
            keep_abs_frac=keep_abs_frac, sor_nb_neighbors=sor_nb_neighbors,
            sor_std_ratio=sor_std_ratio, seed=seed, log=None,
        )
    elif sor_nb_neighbors:
        out = remove_outliers_statistical(out, nb_neighbors=sor_nb_neighbors,
                                          std_ratio=sor_std_ratio)
    if log is not None:
        log(f"    denoise(shadow): {n_in} -> kept {len(out.points)} "
            f"(gate={gate}, dbscan={dbscan}, eps={dbscan_eps}mm, keep_frac={keep_frac})")
    return out


def _dbscan_keep_largest(pts, eps=1.5, min_points=10, work=200000, seed=42):
    """Keep only the single largest DBSCAN cluster of ``pts``. Used per-hemipelvis
    so a both-sided pelvis keeps the dominant on-bone region on *each* side (a
    keep-largest on the merged cloud would keep only one hemipelvis). Clusters on a
    ``work``-point subsample so the ``eps`` neighbourhood sees a stable density (and
    for speed); the result is downsampled further downstream anyway."""
    if len(pts) > work:
        rng = np.random.default_rng(seed)
        pts = pts[rng.choice(len(pts), work, replace=False)]
    if len(pts) < min_points + 1:
        return pts
    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(pts.astype(np.float64))
    labels = np.asarray(pc.cluster_dbscan(eps=float(eps), min_points=int(min_points)))
    sizes = {int(l): int((labels == l).sum()) for l in set(labels.tolist()) if l >= 0}
    if not sizes:
        return pts
    biggest = max(sizes, key=sizes.get)
    return pts[labels == biggest]


def denoise_with_consensus(
    pcd,
    frame_id,
    side,
    voxel=2.0,
    min_frames=8,
    dbscan_eps=1.0,
    dbscan_min_points=10,
    sor_nb_neighbors=20,
    sor_std_ratio=1.5,
    balance=True,
    seed=42,
    log=None,
):
    """CT-free **multi-view consensus** denoiser for the pelvis.

    The pelvis off-bone noise (reverberation, soft tissue) is coherent -- dense,
    clustered, and multi-frame within a single sweep -- so the acoustic-shadow cue
    (no on/off separation on the pelvis), DBSCAN cluster-keep (the off-bone *is* a
    big cluster), and point-density all fail to remove it. What separates on- from
    off-bone is **distinct-frame consensus**: the true surface is corroborated by
    many independent sweeps per voxel, while reverberation packs many points into a
    voxel from *few* frames. The pipeline is:

    1. keep points whose ``voxel``-mm cell holds >= ``min_frames`` DISTINCT frames
       (``frame_id``);
    2. per-hemipelvis (``side`` from the scan folder) keep the largest DBSCAN
       cluster, then merge -- preserves *both* hemipelves (a merged keep-largest
       would drop one side and flip the registration);
    3. statistical outlier removal;
    4. optional L/R ``balance`` (downsample the denser side) -- an imbalanced pelvis
       lets the registration slide tangentially.

    All CT-free (frame ids + scan-folder side only).
    """
    pts = np.asarray(pcd.points)
    frame_id = np.asarray(frame_id)
    side = np.asarray(side)
    n_in = len(pts)
    if n_in == 0:
        return o3d.geometry.PointCloud(pcd)

    # 1. distinct-frame voxel consensus (pandas groupby nunique)
    import pandas as pd

    keys = np.floor(pts / float(voxel)).astype(np.int64)
    h = (keys[:, 0] * 73856093) ^ (keys[:, 1] * 19349663) ^ (keys[:, 2] * 83492791)
    support = pd.Series(frame_id).groupby(h).transform("nunique").to_numpy()
    keep = support >= int(min_frames)
    p, s = pts[keep], side[keep]

    # 2. per-hemipelvis keep-largest
    kept_pts, kept_side = [], []
    for lab in np.unique(s):
        q = p[s == lab]
        if len(q) < dbscan_min_points:
            continue
        k = _dbscan_keep_largest(q, eps=dbscan_eps, min_points=dbscan_min_points)
        kept_pts.append(k)
        kept_side.append(np.full(len(k), lab))
    if not kept_pts:
        out = o3d.geometry.PointCloud()
        out.points = o3d.utility.Vector3dVector(p)
        return out
    P = np.concatenate(kept_pts)
    S = np.concatenate(kept_side)

    # 3. SOR (track side labels through the survivor subset)
    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(P)
    if sor_nb_neighbors and len(P) > sor_nb_neighbors + 1:
        filtered, idx = out.remove_statistical_outlier(
            nb_neighbors=int(sor_nb_neighbors), std_ratio=float(sor_std_ratio)
        )
        P = np.asarray(filtered.points)
        S = S[np.asarray(idx)]

    # 4. L/R balance (downsample the denser hemipelvis)
    if balance and len(np.unique(S)) == 2:
        rng = np.random.default_rng(seed)
        idxs = [np.where(S == lab)[0] for lab in np.unique(S)]
        k = min(len(i) for i in idxs)
        sel = np.concatenate([rng.choice(i, k, replace=False) for i in idxs])
        P = P[sel]

    final = o3d.geometry.PointCloud()
    final.points = o3d.utility.Vector3dVector(P)
    if log is not None:
        log(f"    denoise(consensus): {n_in} -> consensus {int(keep.sum())} "
            f"-> kept {len(P)} (voxel={voxel}mm, min_frames={min_frames}, balance={balance})")
    return final


def denoise_point_cloud(
    pcd,
    work_points=200000,
    dbscan_eps=2.0,
    dbscan_min_points=10,
    keep_frac=0.10,
    keep_abs_frac=0.01,
    sor_nb_neighbors=20,
    sor_std_ratio=1.5,
    seed=42,
    log=None,
):
    """CT-free denoiser: DBSCAN cluster keep + statistical outlier removal.

    The reconstructed cloud is the union of many frames; the true bone surface is
    sampled consistently by overlapping sweeps and forms the dominant, spatially
    connected region(s), while points from frames that missed bone form smaller,
    disconnected clusters. DBSCAN (``dbscan_eps`` mm neighbourhood) groups points;
    clusters smaller than ``max(keep_abs_frac * N, keep_frac * largest_cluster)``
    are dropped as noise. Keeping clusters *relative* to the largest (rather than a
    single largest) preserves multi-region anatomies (e.g. left+right hemipelvis).
    A final statistical outlier removal cleans residual fuzz.

    Clustering is run on a random subsample of ``work_points`` so the distance
    parameters see a consistent density regardless of the input size (the final
    cloud is downsampled further downstream anyway). Returns a new ``PointCloud``.
    """
    n_in = len(pcd.points)
    if n_in == 0:
        return o3d.geometry.PointCloud(pcd)

    work = downsample_to_n_random(pcd, work_points, seed=seed)
    pts = np.asarray(work.points)
    n = len(pts)
    if n < dbscan_min_points + 1:
        return work

    labels = np.asarray(
        work.cluster_dbscan(eps=float(dbscan_eps), min_points=int(dbscan_min_points))
    )
    sizes = {int(l): int((labels == l).sum()) for l in set(labels.tolist()) if l >= 0}
    if sizes:
        biggest = max(sizes.values())
        threshold = max(keep_abs_frac * n, keep_frac * biggest)
        keep_labels = [l for l, s in sizes.items() if s >= threshold]
        mask = np.isin(labels, keep_labels)
        kept = pts[mask]
    else:  # no cluster found (all noise) -> fall back to the working cloud
        kept = pts

    out = o3d.geometry.PointCloud()
    out.points = o3d.utility.Vector3dVector(kept)

    if len(out.points) > sor_nb_neighbors + 1:
        out = remove_outliers_statistical(
            out, nb_neighbors=sor_nb_neighbors, std_ratio=sor_std_ratio
        )

    if log is not None:
        log(
            f"    denoise: {n_in} -> work {n} -> kept {len(out.points)} "
            f"({len(sizes)} clusters, eps={dbscan_eps}mm)"
        )
    return out
