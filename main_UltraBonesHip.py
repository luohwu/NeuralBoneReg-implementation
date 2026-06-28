"""UltraBonesHip entry point — CT ↔ ultrasound bone-surface registration.

Registers the reconstructed ultrasound point cloud against the preoperative CT model for
every (specimen, anatomy) case in the dataset. The registration and evaluation live in
``registration_core.py``; this module only adds the UltraBonesHip file naming
(``specimenNN_<anatomy>``), the per-anatomy score routing, and the case loop.

Run:  python main_UltraBonesHip.py --config configs/UltraBonesHip.yaml
"""
import argparse
import os

import registration_core as core
from models.NeuralUDF.dataset_NormalizeSpace import Dataset
from utilities.converter import filter_src_by_tgt_distance_o3d

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configs", "UltraBonesHip.yaml")
DEFAULT_SPECIMENS = [0, 1, 2, 3, 4]
DEFAULT_ANATOMIES = ["left_femur", "right_femur", "pelvis"]


def case_stem(specimen_id, anatomy):
    """UltraBonesHip file stem, e.g. ``specimen02_left_femur``."""
    return f"specimen{specimen_id:02d}_{anatomy}"


def main(config_file):
    config = core.initialize_runtime(config_file)
    specimens = config.get("specimens", DEFAULT_SPECIMENS)
    anatomies = config.get("anatomies", DEFAULT_ANATOMIES)
    # The config's score is the FEMUR score (ships trimmed/0.6); the pelvis is routed to
    # "mean" per case below (trimming unbalances its two-sided shape -- discards one
    # hemipelvis and slides the L/R balance, sp01 pelvis ~3.8deg -> ~5.5deg). initialize_runtime
    # already set core.SCORE_MODE/TRIM_KEEP from the config; capture them as the femur score
    # since the loop overrides them per anatomy.
    femur_score_mode = core.SCORE_MODE
    femur_trim_keep = core.TRIM_KEEP

    all_metrics = []
    for specimen_id in specimens:
        for anatomy in anatomies:
            stem = case_stem(specimen_id, anatomy)
            intra_file = f"{core.INTRAOPERATIVE_DATA_DIR}/{stem}.xyz"
            preop_file = f"{core.PREOPERATIVE_DATA_DIR}/{stem}.stl"
            if not (os.path.isfile(intra_file) and os.path.isfile(preop_file)):
                print(f"[skip] missing pair: {stem}")
                continue

            # Per-anatomy score routing: the pelvis uses plain "mean"; the femur uses the
            # config's trimmed score. Reset every iteration because the previous pelvis case
            # overrode the globals.
            if anatomy == "pelvis":
                core.SCORE_MODE, core.TRIM_KEEP = "mean", 1.0
            else:
                core.SCORE_MODE, core.TRIM_KEEP = femur_score_mode, femur_trim_keep

            # Dataset preprocessing provides the center/scale used to move between mm and
            # normalized space (one shared frame for the preop model and NeuralUDF). The
            # dataname is resolved through the core so outer-surface extraction (if enabled)
            # applies consistently to both the UDF target and the evaluation cloud.
            dataset_preoperative = Dataset(
                data_dir=core.PREOPERATIVE_DATA_DIR, dataname=core.preoperative_dataname(stem)
            )
            print(f"\n=== {stem} (score_mode={core.SCORE_MODE}, trim_keep={core.TRIM_KEEP}) ===")

            pcd_preoperative = core.load_preoperative_point_cloud(stem, dataset_preoperative)
            udf_network = core.load_udf_network(dataset_preoperative, stem)
            pcd_intraoperative = core.load_intraoperative_point_cloud(
                stem, dataset_preoperative.shape_center, dataset_preoperative.shape_scale
            )

            # EVALUATION ONLY: this filtered cloud is the CD/HD95 metric target and is NEVER
            # passed to the pose optimization (registration scores the intraoperative cloud
            # against the frozen NeuralUDF; see registration_core.estimate_transformation). It
            # therefore cannot leak into the estimated transform. Keep only the portion of the
            # preoperative model plausibly observed by ultrasound before evaluating CD/HD95.
            # This convention follows the authors' prior work:
            #   Wu, L., Seibold, M., Cavalcanti, N.A., Loggia, G., Reissner, L., Sigrist, B.,
            #   Hein, J., Calvet, L., Viehoefer, A. and Fuernstahl, P., 2025. UltraBoneUDF:
            #   Self-supervised bone surface reconstruction from ultrasound based on neural
            #   unsigned distance functions. Computerized Medical Imaging and Graphics, p.102690.
            pcd_preoperative_filtered = filter_src_by_tgt_distance_o3d(
                pcd_preoperative, pcd_intraoperative, 2 / dataset_preoperative.shape_scale
            )

            case_metrics = []
            for run_idx in range(core.NUM_RUNS):
                # Re-sample the disturbance each run to estimate average robustness rather
                # than reporting a single favorable trial.
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

    core.summarize_metrics(all_metrics, label="UltraBonesHip - all cases")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config", default=DEFAULT_CONFIG_FILE, help="Path to the YAML configuration file."
    )
    main(parser.parse_args().config)
