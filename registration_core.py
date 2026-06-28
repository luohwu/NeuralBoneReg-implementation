"""Shared registration runtime + evaluation for NeuralBoneReg.

Dataset-agnostic core used by the per-dataset entry points
(``main_UltraBonesHip.py`` and ``main_SpineDepth.py``). It owns the runtime
configuration, the on-demand NeuralUDF loading, the NeuralReg pose optimization,
the synthetic-perturbation evaluation protocol, and the RTE/RRE/CD/HD95 metrics.

The pipeline uses two coordinate systems and the core moves between them:
  1) mm space     — original meshes, intraoperative clouds, and reported metrics
  2) normalized   — NeuralUDF/NeuralReg inputs and optimization (built from the
                    preoperative cloud's center/scale)

Each entry point only adds its dataset's file naming and case loop; everything
here is shared so the evaluation math cannot drift between datasets.
"""
import copy
import os
import random
import time

import numpy as np
import open3d as o3d
import torch
from pyhocon import ConfigFactory

from models.NeuralReg.network import NeuralReg
from models.NeuralUDF.network import NeuralUDF
from models.NeuralUDF.train_NeuralUDF import train_NeuralUDF
from utilities.converter import (
    compute_RTE_RRE_pcds,
    invert_transformation_matrix,
    vectorToMatrix,
)

# --- shared runtime configuration (populated by initialize_runtime) -----------
SEED_NUMBER = None
POINT_CLOUD_SIZE = None
NUM_OPTIMIZATION_STEPS = None
NUM_SAMPLED_POINTS = None
PERTURBATION_ANGLE_RANGE = None
PERTURBATION_TRANSLATION_RANGE = None
NUM_HEADS = None
SIZE_COARSE_PCD = None
REFINE_HEADS = None
NUM_RUNS = None
SHOW_REGISTRATION_VISUALIZATION = None

# Registration score aggregation. SCORE_MODE "mean" is the original plain mean-UDF;
# "trimmed" keeps the closest TRIM_KEEP fraction of points per head so a residual
# off-bone fragment cannot bias the pose.
SCORE_MODE = "mean"
TRIM_KEEP = 1.0

# Extract the externally-visible (outer) surface of the preoperative CT mesh before
# building the NeuralUDF target. An intra-operative cloud (US/RGB-D) only ever observes
# the outer bone surface, so internal CT faces (inner cortical wall, medullary canal) are
# a target the cloud can never match. Default on for every dataset; the extraction is
# idempotent on meshes that are already outer shells (e.g. SpineDepth). See
# utilities/outer_surface.py. The ray budget (viewpoints x pixels) controls how completely
# the shell is captured: 128 x 384 leaves the femur watertight; sparser budgets drop
# grazing-angle triangles as tiny holes (a pelvis loses ~15% at 64 x 192).
PREOP_OUTER_SURFACE = True
OUTER_SURFACE_VIEWS = 128
OUTER_SURFACE_PIXELS = 384

PREOPERATIVE_DATA_DIR = None
INTRAOPERATIVE_DATA_DIR = None
CHECKPOINT_ROOT = None
UDF_CONFIG_FILE = None

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_config(config_file):
    # Keep runtime configuration outside code so experiments can be repeated by
    # changing YAML values instead of editing the script.
    import yaml

    with open(config_file, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def initialize_runtime(config_file):
    """Populate the shared runtime globals from the YAML config and seed RNGs.

    Returns the parsed config so each entry point can read its own case selection
    (specimen/anatomy/cameras) without re-loading the file.
    """
    global SEED_NUMBER, POINT_CLOUD_SIZE, NUM_OPTIMIZATION_STEPS, NUM_SAMPLED_POINTS
    global PERTURBATION_ANGLE_RANGE, PERTURBATION_TRANSLATION_RANGE
    global NUM_HEADS, SIZE_COARSE_PCD, REFINE_HEADS, NUM_RUNS
    global SHOW_REGISTRATION_VISUALIZATION, SCORE_MODE, TRIM_KEEP
    global PREOP_OUTER_SURFACE, OUTER_SURFACE_VIEWS, OUTER_SURFACE_PIXELS
    global PREOPERATIVE_DATA_DIR, INTRAOPERATIVE_DATA_DIR, CHECKPOINT_ROOT, UDF_CONFIG_FILE

    config = load_config(config_file)

    SEED_NUMBER = config["seed_number"]
    POINT_CLOUD_SIZE = config["point_cloud_size"]
    NUM_OPTIMIZATION_STEPS = config["num_optimization_steps"]
    NUM_SAMPLED_POINTS = config["num_sampled_points"]
    PERTURBATION_ANGLE_RANGE = config["perturbation_angle_range"]
    PERTURBATION_TRANSLATION_RANGE = config["perturbation_translation_range"]
    NUM_HEADS = config["num_heads"]
    SIZE_COARSE_PCD = config["size_coarse_pcd"]
    REFINE_HEADS = config["refine_heads"]
    NUM_RUNS = config["num_runs"]
    SHOW_REGISTRATION_VISUALIZATION = config["show_registration_visualization"]
    # Optional robust score (back-compatible: absent -> original mean-UDF behavior).
    SCORE_MODE = config.get("score_mode", "mean")
    TRIM_KEEP = float(config.get("trim_keep", 1.0))
    # Outer-surface extraction of the preoperative mesh (default on for any dataset).
    PREOP_OUTER_SURFACE = bool(config.get("preop_outer_surface", True))
    OUTER_SURFACE_VIEWS = int(config.get("outer_surface_views", 128))
    OUTER_SURFACE_PIXELS = int(config.get("outer_surface_pixels", 384))

    PREOPERATIVE_DATA_DIR = config["preoperative_data_dir"]
    INTRAOPERATIVE_DATA_DIR = config["intraoperative_data_dir"]
    CHECKPOINT_ROOT = config["checkpoint_root"]
    os.makedirs(CHECKPOINT_ROOT, exist_ok=True)
    UDF_CONFIG_FILE = config["udf_config_file"]

    # Fix the random state so repeated runs are comparable.
    random.seed(SEED_NUMBER)
    np.random.seed(SEED_NUMBER)
    torch.manual_seed(SEED_NUMBER)
    torch.cuda.manual_seed_all(SEED_NUMBER)
    return config


def load_udf_network(dataset_preoperative, stem):
    """Load (or train on demand) the frozen NeuralUDF for one case ``stem``."""
    with open(UDF_CONFIG_FILE) as f:
        conf = ConfigFactory.parse_string(f.read())

    # The UDF network represents the preoperative anatomy as a continuous distance
    # field; downstream registration only needs inference, so it is frozen.
    udf_network = NeuralUDF(**conf["model.udf_network"]).float().to(DEVICE)
    checkpoint_dir = f"{CHECKPOINT_ROOT}/{stem}"
    checkpoint_file = f"{checkpoint_dir}/checkpoints/ckpt_030000_NeuralUDF.pth"

    if not os.path.isfile(checkpoint_file):
        train_NeuralUDF(dataset_preoperative, udf_network, checkpoint_dir)

    checkpoint = torch.load(checkpoint_file, map_location=DEVICE)
    udf_network.load_state_dict(checkpoint["udf_network_fine"])
    for param in udf_network.parameters():
        param.requires_grad = False
    return udf_network


def preoperative_dataname(stem):
    """Dataset name (file stem) of the preoperative mesh the pipeline should read.

    With ``PREOP_OUTER_SURFACE`` on (default), the externally-visible shell of the CT mesh is
    extracted once and cached next to the original as ``<stem>_outer_surface.stl`` in the
    preoperative dir; the NeuralUDF Dataset and the evaluation cloud both read it, so the
    registration target is exactly what an intra-operative sensor observes. It is a sibling
    file (not a subdirectory), so generated meshes never mix with dataset files that are
    auto-downloaded into the same folder. Dataset-agnostic and a near no-op on meshes already
    reduced to a shell. Falls back to the original stem if extraction is disabled or fails.
    """
    if not PREOP_OUTER_SURFACE:
        return stem
    src = os.path.join(PREOPERATIVE_DATA_DIR, f"{stem}.stl")
    outer_stem = f"{stem}_outer_surface"
    dst = os.path.join(PREOPERATIVE_DATA_DIR, f"{outer_stem}.stl")
    if not os.path.isfile(dst):
        if not os.path.isfile(src):
            return stem  # let the downstream reader raise a clear error
        try:
            from utilities.outer_surface import ensure_outer_surface_stl

            ensure_outer_surface_stl(
                src, dst, num_viewpoints=OUTER_SURFACE_VIEWS, image_size=OUTER_SURFACE_PIXELS
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] outer-surface extraction failed for {stem} ({exc}); using raw mesh")
            return stem
    return outer_stem


def load_preoperative_point_cloud(stem, dataset_preoperative):
    preoperative_model_file = f"{PREOPERATIVE_DATA_DIR}/{preoperative_dataname(stem)}.stl"
    preoperative_mesh = o3d.io.read_triangle_mesh(preoperative_model_file)
    # The raw mesh is in mm. NeuralUDF is trained in normalized space, so subtract
    # the dataset center and divide by the scale before sampling points.
    preoperative_mesh.vertices = o3d.utility.Vector3dVector(
        (np.asarray(preoperative_mesh.vertices) - dataset_preoperative.shape_center)
        / dataset_preoperative.shape_scale
    )
    return preoperative_mesh.sample_points_uniformly(POINT_CLOUD_SIZE)


def load_intraoperative_point_cloud(intra_name, pcd_pre_center, pcd_pre_scale):
    intraoperative_file = f"{INTRAOPERATIVE_DATA_DIR}/{intra_name}.xyz"
    pcd_intra_mm_np = np.asarray(o3d.io.read_point_cloud(intraoperative_file).points)
    # Read in mm and map into the same normalized space as the preoperative model
    # and NeuralUDF, using the *preoperative* center/scale so both clouds share one frame.
    pcd_intra_normalized = (pcd_intra_mm_np - pcd_pre_center) / pcd_pre_scale
    return o3d.geometry.PointCloud(o3d.utility.Vector3dVector(pcd_intra_normalized))


def generate_random_perturbation_matrix(dataset_preoperative):
    # The pipeline is evaluated by perturbing the intraoperative cloud with a known
    # rigid transform and measuring how accurately the model recovers it. The
    # perturbation is sampled in normalized space; the config translation range is in
    # mm and converted here with the dataset scale to match NeuralUDF/NeuralReg coords.
    disturbance_r = np.random.uniform(-PERTURBATION_ANGLE_RANGE, PERTURBATION_ANGLE_RANGE, 3)
    translation_range_normalized = (
        PERTURBATION_TRANSLATION_RANGE / dataset_preoperative.shape_scale
    )
    disturbance_t = np.random.uniform(
        -translation_range_normalized, translation_range_normalized, 3
    )
    return vectorToMatrix(t=disturbance_t, rotation_vector=disturbance_r)


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

    # The model proposes multiple candidate rigid transforms ("heads") and refines
    # the most promising ones based on UDF values.
    model = NeuralReg(
        udf_network=udf_network.float(),
        num_heads=num_heads,
        refine_topk_heads=refine_heads,
        score_mode=SCORE_MODE,
        trim_keep=TRIM_KEEP,
    ).float().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    best_loss = float("inf")
    best_t = None
    best_q = None

    for _ in range(NUM_OPTIMIZATION_STEPS):
        optimizer.zero_grad()
        # Random subsampling keeps optimization tractable for dense clouds and acts
        # as a mild stochastic regularizer across iterations.
        idx = torch.randperm(pcd_intraoperative_tensor.shape[2])[:NUM_SAMPLED_POINTS]
        udf_list, _, _, t, q = model(
            pcd_intraoperative_tensor[:, :, idx],
            pcd_intraoperative_tensor[:, :, idx[:size_coarse_pcd]],
        )
        # Convert UDF distances from normalized units back to mm before scoring so
        # reported losses align with the physical scale of the anatomy.
        udf_list = udf_list * dataset_preoperative.shape_scale
        loss = udf_list.mean()
        loss.backward()
        optimizer.step()

        # Keep the best candidate seen over all optimization steps.
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
    """Run one perturb-then-register trial and return its metrics.

    ``pcd_preoperative_filtered`` is EVALUATION ONLY: it is the CD/HD95 target and is handed
    solely to ``compute_RTE_RRE_pcds`` below. The pose optimization (``estimate_transformation``)
    receives only the intraoperative cloud and the frozen ``udf_network`` -- never the filtered
    preoperative cloud -- so the evaluation mask cannot influence the estimated transform.
    """
    # Apply a known perturbation first so registration quality can be measured
    # against a synthetic ground-truth transform.
    transformation_matrix_disturbance_inv = invert_transformation_matrix(transformation_matrix_disturbance)
    pcd_intraoperative_disturbed = copy.deepcopy(pcd_intraoperative).transform(
        transformation_matrix_disturbance
    )

    # Center the source cloud before optimization for numerical stability and to
    # reduce the burden on the translation parameters.
    center_input = np.asarray(pcd_intraoperative_disturbed.points).mean(0)
    pcd_intraoperative_centered = copy.deepcopy(pcd_intraoperative_disturbed).translate(-center_input)

    start = time.time()
    # Pose optimization sees ONLY the intraoperative cloud and the frozen NeuralUDF here --
    # pcd_preoperative_filtered is deliberately not an argument to it (evaluation-only).
    transformation_matrix = estimate_transformation(
        pcd_intraoperative_moved=pcd_intraoperative_centered,
        udf_network=udf_network,
        dataset_preoperative=dataset_preoperative,
        num_heads=num_heads,
        size_coarse_pcd=size_coarse_pcd,
        refine_heads=refine_heads,
    )
    runtime = time.time() - start

    # Compare the predicted transform against the inverse disturbance plus the
    # centering translation applied just before optimization. Metrics are in mm,
    # so the normalization scale is passed into the evaluator.
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

    return {"rte": rte, "rre": rre, "cd": cd, "hd95": hd95, "time": runtime}


def summarize_metrics(metrics_history, label="summary"):
    if not metrics_history:
        print(f"\n{label}: (no runs)")
        return
    print(f"\n{label}:")
    for metric_name in metrics_history[0]:
        metric_values = np.asarray(
            [metrics[metric_name] for metrics in metrics_history], dtype=float
        )
        print(f"{metric_name}: mean[{metric_values.mean():8.4f}] std[{metric_values.std():8.4f}]")
