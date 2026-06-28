"""SpineDepth entry point — per-view CT ↔ RGB-D registration over the dataset.

Differences from UltraBonesHip handled here:
  * File naming ``<specimen>_<level>`` (e.g. ``2_L1``), with two intraoperative clouds
    per vertebra — one per depth-camera view: ``<specimen>_<level>_camera{0,1}.xyz``.
    The preoperative CT mesh ``<specimen>_<level>.stl`` is shared by both views.
  * Two views per vertebra: each camera cloud is registered separately against the
    same preoperative model (and UDF), so each vertebra yields two registrations.
  * One config covers ALL anatomy — this iterates every specimen x level x camera.
  * The dataset is auto-downloaded from the Hugging Face Hub if it is not present.

The registration and evaluation are shared with UltraBonesHip via
``registration_core.py``; only the naming, the auto-download, and the case loop are here.

Run:  python main_SpineDepth.py --config configs/spinedepth.yaml
"""
import argparse
import os

import registration_core as core
from models.NeuralUDF.dataset_NormalizeSpace import Dataset
from utilities.converter import filter_src_by_tgt_distance_o3d

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configs", "spinedepth.yaml")
DEFAULT_HF_REPO = "luohwu/SpineDepth_segmented"
DEFAULT_SPECIMENS = [2, 3, 4, 5, 6, 7, 8]   # 1, 9, 10 excluded (insufficient anatomical coverage)
DEFAULT_LEVELS = ["L1", "L2", "L3", "L4", "L5"]
DEFAULT_CAMERAS = [0, 1]


def case_stem(specimen_id, level):
    """SpineDepth preoperative stem, e.g. ``2_L1`` (no ``specimen``/zero-pad prefix)."""
    return f"{specimen_id}_{level}"


def ensure_dataset(hf_repo):
    """Download the SpineDepth data from the Hugging Face Hub if it is absent locally.

    The HF dataset repo holds ``preoperative/`` and ``intraoperative/`` at its root, so
    it is fetched into the parent of the configured preoperative dir (e.g.
    ``./data/SpineDepth``). A public repo needs no auth; a private one uses your cached
    ``hf auth login`` token.
    """
    pre = core.PREOPERATIVE_DATA_DIR
    if os.path.isdir(pre) and any(name.endswith(".stl") for name in os.listdir(pre)):
        return
    root = os.path.dirname(os.path.normpath(pre))
    print(f"SpineDepth data not found at {root}; downloading from Hugging Face ({hf_repo}) ...")
    from huggingface_hub import snapshot_download

    snapshot_download(repo_id=hf_repo, repo_type="dataset", local_dir=root)
    print(f"  downloaded to {root}")


def main(config_file):
    config = core.initialize_runtime(config_file)
    specimens = config.get("specimens", DEFAULT_SPECIMENS)
    levels = config.get("levels", DEFAULT_LEVELS)
    cameras = config.get("cameras", DEFAULT_CAMERAS)

    ensure_dataset(config.get("hf_repo", DEFAULT_HF_REPO))

    all_metrics = []
    for specimen_id in specimens:
        for level in levels:
            stem = case_stem(specimen_id, level)
            if not os.path.isfile(f"{core.PREOPERATIVE_DATA_DIR}/{stem}.stl"):
                print(f"[skip] missing preoperative mesh: {stem}.stl")
                continue

            # Preoperative model + UDF are built once per vertebra and shared across views.
            # Resolve the dataname through the core so outer-surface extraction (if enabled)
            # applies consistently to the UDF target and the evaluation cloud.
            dataset_preoperative = Dataset(
                data_dir=core.PREOPERATIVE_DATA_DIR, dataname=core.preoperative_dataname(stem)
            )
            pcd_preoperative = core.load_preoperative_point_cloud(stem, dataset_preoperative)
            udf_network = core.load_udf_network(dataset_preoperative, stem)

            for view in cameras:
                intra_name = f"{stem}_camera{view}"
                if not os.path.isfile(f"{core.INTRAOPERATIVE_DATA_DIR}/{intra_name}.xyz"):
                    print(f"[skip] missing intraoperative cloud: {intra_name}.xyz")
                    continue
                print(f"\n=== {intra_name} ===")
                pcd_intraoperative = core.load_intraoperative_point_cloud(
                    intra_name, dataset_preoperative.shape_center, dataset_preoperative.shape_scale
                )
                # EVALUATION ONLY: the CD/HD95 metric target, NEVER passed to the pose
                # optimization (registration scores against the frozen NeuralUDF), so it
                # cannot leak into the estimated transform. Restrict the preop model to the
                # part this view plausibly observes (recomputed per view, since each camera
                # sees a different portion).
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
                core.summarize_metrics(case_metrics, label=f"{intra_name} summary")

    core.summarize_metrics(all_metrics, label="SpineDepth - all cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help="Path to the YAML configuration file."
    )
    main(parser.parse_args().config)
