import argparse
import copy
import os
import random
import time

import numpy as np
import open3d as o3d
import torch
import yaml
from models.NeuralReg.network import NeuralReg
from models.NeuralUDF.network import NeuralUDF
from models.NeuralUDF.dataset_NormalizeSpace import Dataset
from models.NeuralUDF.train_NeuralUDF import train_NeuralUDF
from pyhocon import ConfigFactory

from utilities.converter import (
    compute_RTE_RRE_pcds,
    filter_src_by_tgt_distance_o3d,
    invert_transformation_matrix,
    vectorToMatrix,
)

DEFAULT_CONFIG_FILE = os.path.join(os.path.dirname(__file__), "configs", "example.yaml")

SEED_NUMBER = None
POINT_CLOUD_SIZE = None
NUM_OPTIMIZATION_STEPS = None
NUM_SAMPLED_POINTS = None
PERTURBATION_ANGLE_RANGE = None
PERTURBATION_TRANSLATION_RANGE = None

SPECIMEN_ID = None
ANATOMY = None
NUM_HEADS = None
SIZE_COARSE_PCD = None
REFINE_HEADS = None
NUM_RUNS = None
SHOW_REGISTRATION_VISUALIZATION = None

PREOPERATIVE_DATA_DIR = None
INTRAOPERATIVE_DATA_DIR = None
CHECKPOINT_ROOT = None
UDF_CONFIG_FILE = None


def load_config(config_file):
    # Keep runtime configuration outside code so experiments can be repeated
    # by changing YAML values instead of editing the script itself.
    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_runtime(config_file):
    global SEED_NUMBER
    global POINT_CLOUD_SIZE
    global NUM_OPTIMIZATION_STEPS
    global NUM_SAMPLED_POINTS
    global PERTURBATION_ANGLE_RANGE
    global PERTURBATION_TRANSLATION_RANGE
    global SPECIMEN_ID
    global ANATOMY
    global NUM_HEADS
    global SIZE_COARSE_PCD
    global REFINE_HEADS
    global NUM_RUNS
    global SHOW_REGISTRATION_VISUALIZATION
    global PREOPERATIVE_DATA_DIR
    global INTRAOPERATIVE_DATA_DIR
    global CHECKPOINT_ROOT
    global UDF_CONFIG_FILE

    config = load_config(config_file)

    # These values control the stochastic registration experiment
    SEED_NUMBER = config["seed_number"]
    POINT_CLOUD_SIZE = config["point_cloud_size"]
    NUM_OPTIMIZATION_STEPS = config["num_optimization_steps"]
    NUM_SAMPLED_POINTS = config["num_sampled_points"]
    PERTURBATION_ANGLE_RANGE = config["perturbation_angle_range"]
    PERTURBATION_TRANSLATION_RANGE = config["perturbation_translation_range"]

    SPECIMEN_ID = config["specimen_id"]
    ANATOMY = config["anatomy"]
    NUM_HEADS = config["num_heads"]
    SIZE_COARSE_PCD = config["size_coarse_pcd"]
    REFINE_HEADS = config["refine_heads"]
    NUM_RUNS = config["num_runs"]
    SHOW_REGISTRATION_VISUALIZATION = config["show_registration_visualization"]

    PREOPERATIVE_DATA_DIR = config["preoperative_data_dir"]
    INTRAOPERATIVE_DATA_DIR = config["intraoperative_data_dir"]
    CHECKPOINT_ROOT = config["checkpoint_root"]
    os.makedirs(CHECKPOINT_ROOT,exist_ok=True)
    UDF_CONFIG_FILE = config["udf_config_file"]

    # Fix the random state so repeated runs are comparable.
    random.seed(SEED_NUMBER)
    np.random.seed(SEED_NUMBER)
    torch.manual_seed(SEED_NUMBER)
    torch.cuda.manual_seed_all(SEED_NUMBER)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_random_perturbation_matrix(dataset_preoperative):
    # The registration pipeline is evaluated by first perturbing the
    # intraoperative point cloud with a known rigid transform and then
    # measuring how accurately the model recovers it.
    #
    # This perturbation is sampled in normalized space, not in millimeters.
    # The config translation range is specified in mm and converted here with
    # the dataset normalization scale so it matches the coordinates used by
    # NeuralUDF and NeuralReg.
    disturbance_r = np.random.uniform(-PERTURBATION_ANGLE_RANGE, PERTURBATION_ANGLE_RANGE, 3)
    translation_range_normalized = (
        PERTURBATION_TRANSLATION_RANGE / dataset_preoperative.shape_scale
    )
    disturbance_t = np.random.uniform(
        -translation_range_normalized,
        translation_range_normalized,
        3,
    )
    return vectorToMatrix(t=disturbance_t, rotation_vector=disturbance_r)


def load_udf_network(dataset_preoperative, specimen_id, anatomy):
    with open(UDF_CONFIG_FILE) as f:
        conf = ConfigFactory.parse_string(f.read())

    # The UDF network represents the preoperative anatomy as a continuous
    # distance field. Downstream registration only needs inference, so the
    # network is frozen after loading.
    udf_network = NeuralUDF(**conf["model.udf_network"]).float().to(DEVICE)
    checkpoint_dir = f"{CHECKPOINT_ROOT}/specimen{specimen_id:02d}_{anatomy}"
    checkpoint_file = f"{checkpoint_dir}/checkpoints/ckpt_030000_NeuralUDF.pth"

    # Train on demand if the checkpoint has not been prepared yet.
    if not os.path.isfile(checkpoint_file):
        train_NeuralUDF(dataset_preoperative, udf_network, checkpoint_dir)

    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    udf_network.load_state_dict(checkpoint["udf_network_fine"])
    for param in udf_network.parameters():
        param.requires_grad = False
    return udf_network


def load_preoperative_point_cloud(specimen_id, anatomy, dataset_preoperative):
    preoperative_model_file = f"{PREOPERATIVE_DATA_DIR}/specimen{specimen_id:02d}_{anatomy}.stl"
    preoperative_mesh = o3d.io.read_triangle_mesh(preoperative_model_file)
    # The raw mesh is stored in mm. NeuralUDF is trained in normalized space,
    # so subtract the dataset center and divide by the dataset scale before
    # sampling points for registration.
    preoperative_mesh.vertices = o3d.utility.Vector3dVector(
        (np.asarray(preoperative_mesh.vertices) - dataset_preoperative.shape_center)
        / dataset_preoperative.shape_scale
    )
    return preoperative_mesh.sample_points_uniformly(POINT_CLOUD_SIZE)


def load_intraoperative_point_cloud(specimen_id, anatomy,pcd_pre_center,pcd_pre_sclae):
    intraoperative_file = f"{INTRAOPERATIVE_DATA_DIR}/specimen{specimen_id:02d}_{anatomy}.xyz"
    pcd_intra_mm_np=np.asarray(o3d.io.read_point_cloud(intraoperative_file).points)
    # The intraoperative point cloud is also read in mm and mapped into the
    # same normalized space as the preoperative model and NeuralUDF.
    pcd_intra_normalized=(pcd_intra_mm_np-pcd_pre_center)/pcd_pre_sclae
    pcd_intra_normalized=o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_intra_normalized))
    return pcd_intra_normalized


def estimate_transformation(
    pcd_intraoperative_moved,
    udf_network,
    dataset_preoperative,
    num_heads,
    size_coarse_pcd,
    refine_heads,
):
    # NeuralReg expects a tensor shaped [B, 3, N] in normalized space.
    pcd_intraoperative_tensor = torch.tensor(
        np.asarray(pcd_intraoperative_moved.points).T.reshape(1, 3, -1)
    ).float().to(DEVICE)

    # The model proposes multiple candidate rigid transforms ("heads") and
    # refines the most promising ones based on UDF values.
    model = NeuralReg(
        udf_network=udf_network.float(),
        num_heads=num_heads,
        refine_topk_heads=refine_heads,
    ).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_t = None
    best_q = None

    for _ in range(NUM_OPTIMIZATION_STEPS):
        optimizer.zero_grad()
        # Random subsampling keeps optimization tractable for dense clouds and
        # acts as a mild stochastic regularizer across iterations.
        idx = torch.randperm(pcd_intraoperative_tensor.shape[2])[:NUM_SAMPLED_POINTS]
        udf_list, _, _, t, q = model(
            pcd_intraoperative_tensor[:, :, idx],
            pcd_intraoperative_tensor[:, :, idx[:size_coarse_pcd]],
        )
        # The UDF predicts distances in normalized units. Convert them back to
        # mm before scoring candidates so reported losses align with the
        # physical scale of the anatomy.
        udf_list = udf_list * dataset_preoperative.shape_scale
        loss = udf_list.mean()
        loss.backward()
        optimizer.step()

        # Keep the best candidate seen over all optimization steps
        current_best = udf_list.min().item()
        if current_best < best_loss:
            best_loss = current_best
            best_t = t
            best_q = q

    return vectorToMatrix(
        best_t.detach().cpu().numpy(),
        best_q.detach().cpu().numpy(),
        quat=True,
    )


def run_one_experiment(
    pcd_preoperative,
    pcd_intraoperative,
    udf_network,
    dataset_preoperative,
    transformation_matrix_disturbance,
    num_heads,
    pcd_preoperative_filtered,
    size_coarse_pcd,
    refine_heads,
    run_idx,
):
    # Apply a known perturbation first so registration quality can be measured
    # against a synthetic ground-truth transform.
    transformation_matrix_disturbance_inv = invert_transformation_matrix(transformation_matrix_disturbance)
    pcd_intraoperative_disturbed = copy.deepcopy(pcd_intraoperative).transform(
        transformation_matrix_disturbance
    )

    # Center the source cloud before optimization. This improves numerical
    # stability and reduces the burden on the translation parameters.
    center_input = np.asarray(pcd_intraoperative_disturbed.points).mean(0)
    pcd_intraoperative_centered = copy.deepcopy(pcd_intraoperative_disturbed).translate(-center_input)

    start = time.time()
    transformation_matrix = estimate_transformation(
        pcd_intraoperative_moved=pcd_intraoperative_centered,
        udf_network=udf_network,
        dataset_preoperative=dataset_preoperative,
        num_heads=num_heads,
        size_coarse_pcd=size_coarse_pcd,
        refine_heads=refine_heads,
    )
    runtime = time.time() - start

    # The predicted transform is compared against the inverse disturbance plus
    # the centering translation applied just before optimization. Metrics are
    # reported in mm, so the normalization scale is passed into the evaluator.
    rte, rre, cd, hd95, pcd_intraoperative_registered = compute_RTE_RRE_pcds(
        transformation_matrix,
        transformation_matrix_disturbance_inv @ vectorToMatrix(center_input, [0, 0, 0]),
        translation_scale=dataset_preoperative.shape_scale,
        src_pcd_moved=pcd_intraoperative_centered,
        tgt_pcd=pcd_preoperative_filtered,
    )
    print(
        f"run {run_idx}: registration: RTE[{rte:8.4f}], RRE[{rre:8.4f}], "
        f"CD[{cd:8.4f}], HD95[{hd95:8.4f}], used time: {runtime}"
    )

    if SHOW_REGISTRATION_VISUALIZATION:
        o3d.visualization.draw_geometries(
            [
                pcd_intraoperative_disturbed.paint_uniform_color((0, 1, 0)),
                pcd_preoperative.paint_uniform_color((1, 0, 0)),
            ]
        )
        o3d.visualization.draw_geometries(
            [
                pcd_intraoperative_registered.paint_uniform_color((0, 1, 0)),
                pcd_preoperative.paint_uniform_color((1, 0, 0)),
            ]
        )

    return {
        "rte": rte,
        "rre": rre,
        "cd": cd,
        "hd95": hd95,
        "time": runtime,
    }


def summarize_metrics(metrics_history):
    print("\nsummary:")
    for metric_name in metrics_history[0]:
        metric_values = np.asarray([metrics[metric_name] for metrics in metrics_history], dtype=float)
        print(f"{metric_name}: mean[{metric_values.mean():8.4f}] std[{metric_values.std():8.4f}]")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        default=DEFAULT_CONFIG_FILE,
        help="Path to the YAML configuration file.",
    )
    return parser.parse_args()


def main(config_file):
    initialize_runtime(config_file)

    # This pipeline uses two coordinate systems:
    # 1) mm space for the original meshes, point clouds, and reported metrics
    # 2) normalized space for NeuralUDF/NeuralReg inputs and optimization
    # Dataset preprocessing provides the center/scale needed to move between
    # those two spaces consistently.
    dataset_preoperative = Dataset(
        data_dir=PREOPERATIVE_DATA_DIR,
        dataname=f"specimen{SPECIMEN_ID:02d}_{ANATOMY}",
    )
    print(f"working on {SPECIMEN_ID}_{ANATOMY}")

    pcd_preoperative = load_preoperative_point_cloud(
        SPECIMEN_ID,
        ANATOMY,
        dataset_preoperative,
    )
    udf_network = load_udf_network(dataset_preoperative, SPECIMEN_ID, ANATOMY)
    pcd_intraoperative = load_intraoperative_point_cloud(SPECIMEN_ID, ANATOMY,pcd_pre_center=dataset_preoperative.shape_center,pcd_pre_sclae=dataset_preoperative.shape_scale)

    # Keep only the portion of the preoperative model that is plausibly
    # observed by ultrasound before evaluating registration quality, such as CD and HD.
    # This is the convention in prior work
    # Wu, L., Seibold, M., Cavalcanti, N.A., Loggia, G., Reissner, L., Sigrist, B., Hein, J.,
    # Calvet, L., Viehöfer, A. and Fürnstahl, P., 2025. 
    # UltraBoneUDF: Self-supervised bone surface reconstruction from ultrasound based on neural unsigned distance functions. 
    # Computerized Medical Imaging and Graphics, p.102690.
    pcd_preoperative_filtered = filter_src_by_tgt_distance_o3d(
        pcd_preoperative,
        pcd_intraoperative,
        2 / dataset_preoperative.shape_scale,
    )


    metrics_history = []
    for run_idx in range(NUM_RUNS):
        # Re-sample the disturbance each run to estimate average robustness
        # rather than reporting a single favorable trial.
        transformation_matrix_disturbance = generate_random_perturbation_matrix(
            dataset_preoperative
        )
        metrics = run_one_experiment(
            pcd_preoperative=pcd_preoperative,
            pcd_intraoperative=pcd_intraoperative,
            udf_network=udf_network,
            dataset_preoperative=dataset_preoperative,
            transformation_matrix_disturbance=transformation_matrix_disturbance,
            num_heads=NUM_HEADS,
            pcd_preoperative_filtered=pcd_preoperative_filtered,
            size_coarse_pcd=SIZE_COARSE_PCD,
            refine_heads=REFINE_HEADS,
            run_idx=run_idx,
        )
        metrics_history.append(metrics)

    summarize_metrics(metrics_history)


if __name__ == "__main__":
    args = parse_args()
    main(args.config)

