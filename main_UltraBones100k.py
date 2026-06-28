"""UltraBones100k entry point — CT <-> ultrasound registration (fibula & tibia).

UltraBones100k ships a 3D reconstruction per ultrasound record. For each (specimen, anatomy)
this script:
  1. merges the per-record reconstructions into one intraoperative cloud, and
  2. removes cross-bone contamination with a CT-free DBSCAN keep-largest-cluster denoiser
     (a fibula sweep contains some tibia points and vice versa; the target bone is the
     dominant cluster, the other bone a smaller separable cluster), then SOR;
  3. registers the result against the preoperative CT mesh via the shared registration_core.

Only fibula and tibia are used (foot is excluded). The per-head optimization loss is mean()
(as in registration_core, identical to UltraBonesHip); the per-point ``score_mode`` is the
knob that makes it work on these long thin bones. ``trimmed`` (trim_keep=0.7) is used: it
scores each pose on only the closest 70% of points, dropping the off-bone tail (tibiofibular-
junction contamination + reverb) that otherwise makes a ~180 deg flipped fibula pose
competitive under mean/meanmax. This removes the flips; the tibia tolerates the same setting,
so one global score_mode is used (see configs/ultrabones100k.yaml for the rationale).
Validated over 28 cases x 5 perturbations: RR 0.957 (tibia 1.000, fibula 0.914),
RTE 2.12+/-1.33 mm, RRE 1.72+/-1.12 deg, zero flips.

Run:  python main_UltraBones100k.py --config configs/ultrabones100k.yaml
"""
import argparse
import glob
import os
import shutil

import numpy as np
import open3d as o3d

import registration_core as core
from models.NeuralUDF.dataset_NormalizeSpace import Dataset
from utilities.converter import filter_src_by_tgt_distance_o3d

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configs", "ultrabones100k.yaml")
DATASET_ROOT = "/mnt/UltraBones100k"
DEFAULT_SPECIMENS = [f"specimen{i:02d}" for i in range(1, 15)]   # 14 specimens
DEFAULT_ANATOMIES = ["fibula", "tibia"]
DEFAULT_RECORD_GLOB = "ultrasound_records/{anatomy}/record*/3D_reconstructions/with_pred_labels/reconstruction_pcd.xyz"


def case_stem(specimen, anatomy):
    """UltraBones100k file stem, e.g. ``specimen01_fibula``."""
    return f"{specimen}_{anatomy}"


def merge_records_and_denoise(record_files, eps, min_points, work_points, sor_nb, sor_std):
    """Union the per-record reconstructions and keep the dominant bone (CT-free).

    DBSCAN groups the merged cloud; the largest cluster is the target bone, smaller clusters
    are the other-bone contamination (and stray noise) and are dropped; SOR cleans the rest.
    """
    clouds = [np.asarray(o3d.io.read_point_cloud(f).points) for f in record_files]
    clouds = [c for c in clouds if len(c)]
    if not clouds:
        return None
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(clouds)))
    # Cap the working size so DBSCAN stays fast (density-stable random subsample).
    if work_points and len(pcd.points) > work_points:
        idx = np.random.choice(len(pcd.points), work_points, replace=False)
        pcd = pcd.select_by_index(idx.tolist())
    labels = np.asarray(pcd.cluster_dbscan(eps=eps, min_points=min_points))
    if labels.max() >= 0:
        biggest = np.bincount(labels[labels >= 0]).argmax()
        pcd = pcd.select_by_index(np.where(labels == biggest)[0].tolist())
    if len(pcd.points) > 0:
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=sor_nb, std_ratio=sor_std)
    return pcd


def ensure_dataset(config, specimens, anatomies):
    """Build data/UltraBones100k/{preoperative,intraoperative} from the raw dataset if absent.

    intraoperative/<stem>.xyz = merged + denoised per-record reconstructions.
    preoperative/<stem>.stl   = the CT bone segmentation mesh (copied).
    """
    root = config.get("dataset_root", DATASET_ROOT)
    rec_glob = config.get("record_glob", DEFAULT_RECORD_GLOB)
    eps = float(config.get("denoise_dbscan_eps", 2.0))
    min_points = int(config.get("denoise_dbscan_min_points", 10))
    work_points = int(config.get("denoise_work_points", 200000))
    sor_nb = int(config.get("sor_nb_neighbors", 20))
    sor_std = float(config.get("sor_std_ratio", 2.0))
    os.makedirs(core.PREOPERATIVE_DATA_DIR, exist_ok=True)
    os.makedirs(core.INTRAOPERATIVE_DATA_DIR, exist_ok=True)

    for specimen in specimens:
        for anatomy in anatomies:
            stem = case_stem(specimen, anatomy)
            intra_out = f"{core.INTRAOPERATIVE_DATA_DIR}/{stem}.xyz"
            preop_out = f"{core.PREOPERATIVE_DATA_DIR}/{stem}.stl"

            if not os.path.isfile(intra_out):
                pattern = os.path.join(root, specimen, rec_glob.format(anatomy=anatomy))
                records = sorted(glob.glob(pattern))
                if not records:
                    print(f"[skip] no records for {stem}: {pattern}")
                else:
                    pcd = merge_records_and_denoise(records, eps, min_points, work_points, sor_nb, sor_std)
                    if pcd is None or len(pcd.points) == 0:
                        print(f"[skip] empty merged cloud for {stem}")
                    else:
                        o3d.io.write_point_cloud(intra_out, pcd)
                        print(f"  {stem}: merged {len(records)} records -> {len(pcd.points)} pts -> {intra_out}")

            if not os.path.isfile(preop_out):
                ct = os.path.join(root, specimen, "CT_bone_segmentations", f"{anatomy}.stl")
                if os.path.isfile(ct):
                    shutil.copyfile(ct, preop_out)
                    print(f"  {stem}: copied CT mesh -> {preop_out}")
                else:
                    print(f"[warn] CT mesh not found: {ct}")


def main(config_file):
    config = core.initialize_runtime(config_file)
    specimens = config.get("specimens", DEFAULT_SPECIMENS)
    anatomies = config.get("anatomies", DEFAULT_ANATOMIES)

    ensure_dataset(config, specimens, anatomies)

    all_metrics = []
    for specimen in specimens:
        for anatomy in anatomies:
            stem = case_stem(specimen, anatomy)
            if not os.path.isfile(f"{core.PREOPERATIVE_DATA_DIR}/{stem}.stl"):
                print(f"[skip] missing preoperative mesh: {stem}.stl")
                continue
            if not os.path.isfile(f"{core.INTRAOPERATIVE_DATA_DIR}/{stem}.xyz"):
                print(f"[skip] missing intraoperative cloud: {stem}.xyz")
                continue
            print(f"\n=== {stem} ===")

            dataset_preoperative = Dataset(
                data_dir=core.PREOPERATIVE_DATA_DIR, dataname=core.preoperative_dataname(stem)
            )
            pcd_preoperative = core.load_preoperative_point_cloud(stem, dataset_preoperative)
            udf_network = core.load_udf_network(dataset_preoperative, stem)
            pcd_intraoperative = core.load_intraoperative_point_cloud(
                stem, dataset_preoperative.shape_center, dataset_preoperative.shape_scale
            )
            # EVALUATION ONLY: the CD/HD95 metric target (the preop surface region the US
            # plausibly observed). It is NEVER passed to the pose optimization -- registration
            # scores the intraoperative cloud against the frozen NeuralUDF only (see
            # registration_core.estimate_transformation) -- so it cannot leak into the estimate.
            pcd_preoperative_filtered = filter_src_by_tgt_distance_o3d(
                pcd_preoperative, pcd_intraoperative, 2 / dataset_preoperative.shape_scale
            )

            case_metrics = []
            for run_idx in range(core.NUM_RUNS):
                transformation_matrix_disturbance = core.generate_random_perturbation_matrix(
                    dataset_preoperative
                )
                metrics = core.run_one_experiment(
                    pcd_preoperative=pcd_preoperative,
                    pcd_intraoperative=pcd_intraoperative,
                    udf_network=udf_network,
                    dataset_preoperative=dataset_preoperative,
                    transformation_matrix_disturbance=transformation_matrix_disturbance,
                    num_heads=core.NUM_HEADS,
                    pcd_preoperative_filtered=pcd_preoperative_filtered,
                    size_coarse_pcd=core.SIZE_COARSE_PCD,
                    refine_heads=core.REFINE_HEADS,
                    run_idx=run_idx,
                )
                case_metrics.append(metrics)
                all_metrics.append(metrics)
            core.summarize_metrics(case_metrics, label=f"{stem} summary")

    core.summarize_metrics(all_metrics, label="UltraBones100k - all cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help="Path to the YAML configuration file."
    )
    main(parser.parse_args().config)
