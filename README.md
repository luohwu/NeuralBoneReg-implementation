# NeuralBoneReg: An Instance-Specific Label-Free Point Cloud-Based Method for Multi-Modal Bone Surface Registration

This repository contains the official implementation of NeuralBoneReg. For the paper, see [NeuralBoneReg](https://doi.org/10.1016/j.media.2026.104133); for the project page (overview and qualitative result videos), see [neuralbonereg.github.io](https://neuralbonereg.github.io/).

This is the full release: the implementation together with all three evaluation datasets (**UltraBonesHip**, **UltraBones100k**, and **SpineDepth**) is now publicly available (see [Dataset](#3-dataset)).


## Table of Contents

- [1. Repository Layout](#1-repository-layout)
- [2. Requirements](#2-requirements)
- [3. Dataset](#3-dataset)
- [4. Installation](#4-installation)
- [5. Results](#5-results)
- [6. Configuration](#6-configuration)
- [7. Code reference](#7-code-reference)
- [8. Limitations and Future Work](#8-limitations-and-future-work)
- [9. Citation](#9-citation)
- [10. Questions or Feedback?](#10-questions-or-feedback)

## 1. Repository Layout

```text
.
├── checkpoints_NeuralUDF/   # pretrained NeuralUDF checkpoints
│   └── UltraBonesHip/
├── configs/                 # runtime configuration files
│   ├── UltraBonesHip.yaml
│   ├── spinedepth.yaml
│   └── ultrabones100k.yaml
├── data/                    # datasets (downloaded / reconstructed here)
│   ├── UltraBonesHip/
│   ├── SpineDepth/
│   └── UltraBones100k/
├── models/                  # model implementations
│   ├── NeuralUDF/
│   └── NeuralReg/
├── reconstruction/          # raw ultrasound -> UltraBonesHip point-cloud reconstruction
├── extensions/              # CUDA Chamfer distance extension
├── utilities/               # helper utilities
├── Dockerfile               # CUDA-enabled container image
├── main_UltraBonesHip.py    # UltraBonesHip entry script
├── main_SpineDepth.py       # SpineDepth entry script (auto-downloads dataset from HF)
├── main_UltraBones100k.py   # UltraBones100k entry script (fibula & tibia)
├── registration_core.py     # shared registration + evaluation
├── README.md
└── requirements.txt         # Python dependencies
```

## 2. Requirements

Both installation options below are container-based, so the host only needs:

- An **NVIDIA GPU** with a recent driver. The image uses the CUDA 12.8 PyTorch build.
- **Docker** and the **NVIDIA Container Toolkit** (for GPU passthrough, `--gpus all`).
- For the Dev Container option: **VS Code** with the **Dev Containers** extension.

All Python dependencies and the CUDA Chamfer extension are prebuilt inside the image
(from the `Dockerfile`), so there is no separate Python/`pip` setup on the host.

## 3. Dataset

NeuralBoneReg is evaluated on three datasets, all now publicly available. Each provides, per
case, a **preoperative CT bone mesh** (`.stl`) and one or more **intraoperative point clouds**
(`.xyz`); everything is read from / written to `./data/<dataset>/`.

| Dataset | Modality | Anatomy | Source | How it becomes ready |
|---|---|---|---|---|
| **SpineDepth** | RGB-D | lumbar vertebrae (L1–L5) | [Hugging Face](https://huggingface.co/datasets/luohwu/SpineDepth_segmented) | auto-downloaded on first run (no manual step) |
| **UltraBonesHip** | ultrasound | femur (left / right), pelvis | [Hugging Face](https://huggingface.co/datasets/luohwu/UltraBonesHip) | download raw US; reconstructed automatically on first run |
| **UltraBones100k** | ultrasound | fibula, tibia | [GitHub](https://github.com/luohwu/UltraBones100k) | download, then built on first run |

For the datasets you download yourself (UltraBonesHip, UltraBones100k), unpack each to a host
folder and **mount it into the container**. Where to set that path is described in
[Installation](#4-installation). SpineDepth needs nothing: the entry point fetches it for you.

### Data layout

```text
./data/UltraBonesHip/
├── preoperative/    specimenNN_<anatomy>.stl     (CT bone meshes; anatomy = left_femur | right_femur | pelvis)
└── intraoperative/  specimenNN_<anatomy>.xyz     (reconstructed US clouds)

./data/SpineDepth/
├── preoperative/    <specimen>_<level>.stl
└── intraoperative/  <specimen>_<level>_camera{0,1}.xyz

./data/UltraBones100k/
├── preoperative/    <specimen>_<anatomy>.stl     (CT bone meshes; anatomy = fibula | tibia)
└── intraoperative/  <specimen>_<anatomy>.xyz     (merged + denoised US clouds)

./checkpoints_NeuralUDF/<dataset>/<case>/checkpoints/ckpt_030000_NeuralUDF.pth
```

If a UDF checkpoint is missing, the entry point trains it automatically (~30k iterations per
case) before registration.

### SpineDepth (automatic)

`main_SpineDepth.py` downloads SpineDepth automatically from the Hugging Face Hub
([luohwu/SpineDepth_segmented](https://huggingface.co/datasets/luohwu/SpineDepth_segmented))
into `./data/SpineDepth/` on first run; no manual step is required.

### UltraBonesHip (download, then auto-reconstructed)

UltraBonesHip is reconstructed from raw ultrasound. Download the dataset from the Hugging Face
Hub ([luohwu/UltraBonesHip](https://huggingface.co/datasets/luohwu/UltraBonesHip)), unpack it
to a host folder, and mount it at `/mnt/UltraBonesHip` (see [Installation](#4-installation)). The
unpacked folder is expected to look like:

```text
/mnt/UltraBonesHip/                          # 5 specimens: specimen00 .. specimen04
└── specimenNN/
    ├── CT_bone_segmentations/               # preoperative CT bone meshes
    │   ├── left_femur.stl
    │   ├── right_femur.stl
    │   └── pelvis.stl
    └── ultrasound_records/                  # raw tracked ultrasound sweeps
        ├── left_femur_axial/                #   each sweep contains record*/ folders;
        ├── left_femur_coronal/              #   each record*/ contains:
        ├── right_femur_axial/               #     UltrasoundImages/<timestamp>.png  (US frames)
        ├── right_femur_coronal/             #     poses.csv                         (tracked poses)
        ├── left_pelvis_axial/
        ├── left_pelvis_coronal/
        ├── right_pelvis_axial/
        └── right_pelvis_coronal/
```

**No separate reconstruction step is needed**: on its first run `main_UltraBonesHip.py`
reconstructs the data itself: it segments the raw ultrasound, projects it to 3D, and writes the
intraoperative point clouds plus the matching preoperative CT meshes into `./data/UltraBonesHip/`
(the bone-segmentation model is fetched from the Hugging Face Hub on first use). See
[`reconstruction/README.md`](reconstruction/README.md) for details. The dataset covers
`left_femur`, `right_femur`, and `pelvis` for specimens `specimen00`–`specimen04`.

### UltraBones100k (download, then build)

UltraBones100k provides a 3D reconstruction per ultrasound record; this project uses only the
**fibula** and **tibia** (the foot is excluded). Download the dataset from its project
repository ([luohwu/UltraBones100k](https://github.com/luohwu/UltraBones100k)), unpack it to a
host folder, and mount it at `/mnt/UltraBones100k` (see [Installation](#4-installation)). The
unpacked folder is expected to look like:

```text
/mnt/UltraBones100k/                         # 14 specimens: specimen01 .. specimen14
└── specimenNN/
    ├── CT_bone_segmentations/               # preoperative CT bone meshes
    │   ├── fibula.stl
    │   └── tibia.stl
    └── ultrasound_records/                  # one 3D reconstruction per record
        ├── fibula/   record*/3D_reconstructions/with_pred_labels/reconstruction_pcd.xyz
        └── tibia/    record*/3D_reconstructions/with_pred_labels/reconstruction_pcd.xyz
```

On its first run `main_UltraBones100k.py` builds `./data/UltraBones100k/` itself: for each (specimen,
anatomy) it merges the per-record reconstructions into one intraoperative cloud, removes
cross-bone contamination with a DBSCAN keep-largest-cluster denoiser (a fibula sweep
contains some tibia points and vice versa) plus statistical outlier removal, copies the matching
CT mesh, and then registers. No separate reconstruction step is needed. The dataset covers
`fibula` and `tibia` for `specimen01`–`specimen14`.

## 4. Installation

Two supported options, both built from the included `Dockerfile`. All Python dependencies and
the CUDA Chamfer extension are prebuilt in the image, so there is no host-side Python setup.

### Where to set the dataset path

UltraBonesHip and UltraBones100k are downloaded to a folder on your host (see
[Dataset](#3-dataset)) and **mounted into the container** at a fixed `/mnt` path. The container
side never changes; you only set the **host path**, in `.devcontainer/devcontainer.json`
(Option A) or with `-v` (Option B). SpineDepth needs no mount (it auto-downloads).

| Dataset | Mounted in container at | Set the host path via |
|---|---|---|
| UltraBonesHip | `/mnt/UltraBonesHip` | `mounts` in `.devcontainer/devcontainer.json` (A) / `-v` (B) |
| UltraBones100k | `/mnt/UltraBones100k` | `mounts` in `.devcontainer/devcontainer.json` (A) / `-v` (B) |
| SpineDepth | n/a (auto-downloaded) | n/a |

### Option A: Dev Container in VS Code (recommended)

A ready-to-run environment in one click (the dev container also ships the Claude Code CLI).

1. Open the repository in VS Code (`code .`).
2. Edit the `mounts` in `.devcontainer/devcontainer.json` so each dataset's **host path**
   (`source=...`) points at your local copy; the container side (`target=/mnt/...`) stays as-is:

   ```jsonc
   "mounts": [
     "source=/path/to/UltraBonesHip,target=/mnt/UltraBonesHip,type=bind,consistency=cached",
     "source=/path/to/UltraBones100k,target=/mnt/UltraBones100k,type=bind,consistency=cached"
   ]
   ```

   Keep only the line(s) for the dataset(s) you use; SpineDepth needs no mount.
3. Run **Dev Containers: Reopen in Container** from the Command Palette (F1). The container builds
   from the `Dockerfile` and mounts the repo at `/workspace`. When the build finishes, prepare the
   data (see [Dataset](#3-dataset)) and run it (see [Running](#running) below).

### Option B: Docker

Build the image and run it with GPU access, mounting the repo and (for UltraBonesHip /
UltraBones100k) the dataset(s) at the `/mnt` paths above:

```bash
docker build -t neuralbonereg .

docker run --gpus all -it --shm-size=16g \
  -v "$(pwd)":/workspace \
  -v /path/to/UltraBonesHip:/mnt/UltraBonesHip \
  -v /path/to/UltraBones100k:/mnt/UltraBones100k \
  neuralbonereg
```

Keep only the dataset `-v` lines you need (SpineDepth needs none). Everything is already
installed in the image; prepare the data (see [Dataset](#3-dataset)) and run it as below.

### Running

From the repository root:

UltraBonesHip (reconstructed automatically on first run, see [Dataset](#3-dataset)):

```bash
python main_UltraBonesHip.py --config configs/UltraBonesHip.yaml
```

SpineDepth (all specimens, levels, and camera views; auto-downloaded from Hugging Face if absent):

```bash
python main_SpineDepth.py --config configs/spinedepth.yaml
```

UltraBones100k (all specimens × {fibula, tibia}; builds `./data/UltraBones100k` from the mounted
dataset on first run, see [Dataset](#3-dataset)):

```bash
python main_UltraBones100k.py --config configs/ultrabones100k.yaml
```

## 5. Results

Registration accuracy under the synthetic-perturbation protocol: each intraoperative cloud is
displaced by a random rigid transform (rotation up to ±180°, translation up to ±500 mm) and the
method must recover it. **RTE** = translation error (mm), **RRE** = rotation error (°), **CD** /
**HD95** = Chamfer / 95th-percentile surface distance (mm) between the registered cloud and the
observed region of the CT surface. **RR** (registration recall) = fraction of runs with
**RTE < 4 mm and RRE < 5°** (the paper's criterion). Rows are percentiles over all runs; the last row is mean ± std.

### UltraBones100k (fibula & tibia)

14 specimens × {fibula, tibia} = 28 cases × 5 perturbation runs (140 trials). **RR = 0.900 (126 / 140).**

| percentile | RTE (mm) | RRE (°) | CD (mm) | HD95 (mm) |
|---|---|---|---|---|
| p25 | 1.05 | 0.70 | 0.61 | 1.42 |
| p50 (median) | 1.77 | 1.42 | 0.67 | 1.65 |
| p75 | 2.81 | 2.57 | 0.75 | 2.25 |
| p90 | 3.92 | 3.18 | 1.11 | 4.69 |
| p95 | 4.78 | 3.76 | 1.31 | 6.84 |
| **mean ± std** | **2.12 ± 1.32** | **1.72 ± 1.12** | **0.72 ± 0.22** | **2.26 ± 1.75** |

### UltraBonesHip (femur & pelvis)

5 specimens × {left_femur, right_femur, pelvis} = 15 clouds × 10 runs (150 trials). **RR = 0.940 (141 / 150).**

| percentile | RTE (mm) | RRE (°) | CD (mm) | HD95 (mm) |
|---|---|---|---|---|
| p25 | 1.28 | 0.85 | 1.04 | 2.45 |
| p50 (median) | 2.30 | 1.29 | 1.45 | 4.12 |
| p75 | 3.20 | 2.25 | 2.84 | 11.87 |
| p90 | 3.75 | 3.10 | 4.58 | 18.65 |
| p95 | 4.04 | 3.42 | 5.63 | 18.69 |
| **mean ± std** | **2.27 ± 1.11** | **1.61 ± 0.95** | **2.09 ± 1.44** | **7.49 ± 5.90** |


### SpineDepth (lumbar vertebrae, RGB-D)

7 specimens × 5 levels × 2 camera views = 70 views × 10 runs (700 trials).
**RR = 0.907 (635 / 700).**

| percentile | RTE (mm) | RRE (°) | CD (mm) | HD95 (mm) |
|---|---|---|---|---|
| p25 | 0.45 | 0.72 | 1.43 | 2.97 |
| p50 (median) | 0.79 | 1.17 | 1.51 | 3.22 |
| p75 | 1.34 | 2.11 | 1.58 | 3.38 |
| p90 | 2.48 | 4.62 | 1.63 | 3.70 |
| p95 | 3.08 | 7.20 | 1.67 | 4.07 |
| **mean ± std** | **1.54 ± 3.66** | **4.30 ± 17.41** | **1.77 ± 1.79** | **3.92 ± 4.62** |

## 6. Configuration

The pipeline is configured through a YAML file. By default it uses `configs/UltraBonesHip.yaml`, but you can pass a different file at runtime with `--config`.

Key fields:

- `show_registration_visualization`: opens an Open3D viewer when `true`
- `specimens` and `anatomies`: which specimen IDs × anatomies to register (UltraBonesHip runs the full dataset by default; missing pairs are skipped)
- `preoperative_data_dir`: directory containing preoperative meshes
- `intraoperative_data_dir`: directory containing intraoperative point clouds
- `checkpoint_root`: root directory for NeuralUDF checkpoints
- `udf_config_file`: UDF network definition
- `num_heads`: number of hypothesis generation heads
- `size_coarse_pcd`: number of sampled points used for the coarse registration branch
- `refine_heads`: number of top candidate transformations kept for refinement
- `num_optimization_steps`: number of optimization iterations used during registration
- `num_runs`: number of randomized perturbation trials

## 7. Code reference

This code is adapted from the following repositories (many thanks to the contributors):
- FUNSR: https://github.com/chenhbo/FUNSR
- UltraBoneUDF: https://github.com/luohwu/UltraBoneUDF

## 8. Limitations and Future Work

- **The registration loss is sensitive to noise.** Pose hypotheses are scored by the
  preoperative unsigned distance field, so off-surface (noisy) points degrade the score,
  which is why the intraoperative point clouds are aggressively denoised before registration.
  More advanced loss/score formulations could relax this requirement: for example, taking
  **surface-normal direction alignment** into account (matching the intraoperative cloud
  normals to the UDF gradient), or integrating a per-point **bone-mask probability or noise
  probability** into the loss so that uncertain points contribute less.

- **Records are merged per anatomy.** The multiple intraoperative records (sweeps) of an
  anatomy are merged into a single point cloud before registration, for two reasons: (1) it
  reduces the number of point-cloud pairs, which speeds up evaluation, and (2) the merged
  cloud has better anatomical coverage than any single-record reconstruction, which improves
  registration accuracy. As a result, the current pipeline can still fail on some individual
  records in isolation (e.g. low-coverage or ambiguous views); making single-record
  registration robust is left for future work.

## 9. Citation

If you use NeuralBoneReg in your research, please cite:

```bibtex
@article{wu2026neuralbonereg,
  title   = {NeuralBoneReg: An instance-specific label-free point cloud-based method for multi-modal bone surface registration},
  author  = {Wu, Luohong and Seibold, Matthias and Cavalcanti, Nicola A. and Ao, Yunke and Flepp, Roman and Massalimova, Aidana and Calvet, Lilian and F{\"u}rnstahl, Philipp},
  journal = {Medical Image Analysis},
  year    = {2026},
  doi     = {10.1016/j.media.2026.104133}
}
```

If you use the bundled datasets, please also cite their original sources: SpineDepth ([Liebmann et al., 2021](https://doi.org/10.3390/jimaging7090164)) and [UltraBones100k](https://github.com/luohwu/UltraBones100k).

## 10. Questions or Feedback?

If you have questions, you can open a new GitHub issue within this repository, and we'll get back to you!
