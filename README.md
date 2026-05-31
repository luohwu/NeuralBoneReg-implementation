# NeuralBoneReg: An Instance-Specific Label-Free Point Cloud-Based Method for Multi-Modal Bone Surface Registration

This repository contains the official implementation of NeuralBoneReg. For the paper, see [NeuralBoneReg](https://doi.org/10.1016/j.media.2026.104133).

The repository is currently just a working version, and we are preparing the final full release, including the dataset, which is due at the end of June 2026. Stay tuned.


## Table of Contents

- [1. Repository Layout](#1-repository-layout)
- [2. Requirements](#2-requirements)
- [3. Installation](#3-installation)
- [4. Data](#4-data)
- [6. Configuration](#6-configuration)
- [7. Running](#7-running)
- [8. Code reference](#8-code-reference)
- [9. Questions or Feedback?](#9-questions-or-feedback)

## 1. Repository Layout

```text
.
├── checkpoints_NeuralUDF/   # pretrained NeuralUDF checkpoints 
│   └── UltraBonesHip/             
├── configs/                 # runtime configuration files
│   ├── example.yaml      
│   └── failure_case.yaml            
├── data/                    # dataset folder   
│   └── UltraBonesHip/       
├── models/                  # model implementations
│   ├── NeuralUDF/           
│   └── NeuralReg/           
├── extensions/              
├── utilities/               # helper utilities
├── Dockerfile               # CUDA-enabled container image
├── main.py                  # entry script
├── README.md                
└── requirements.txt         # Python dependencies
```

## 2. Requirements

- Python 3.10 or 3.11
- PyTorch environment compatible with CUDA 12.6
- Build tooling for the local CUDA extension in `extensions/chamfer_dist`



## 3. Installation

### Requirements.txt

Create and activate a Python environment, then install the project directly from the repository root:

```bash
pip install -r requirements.txt
pip install --no-build-isolation ./extensions/chamfer_dist
```


### Docker

The repository also includes a `Dockerfile` based on `pytorch/pytorch:2.7.1-cuda12.6-cudnn9-devel`. 


## 4. Data

### Data Layout

The default config uses `./data` for the dataset and `./checkpoints_NeuralUDF` for checkpoints.

For example:

```text
./data/UltraBonesHip/
├── preoperative/
│   └── specimen00_left_femur.stl
└── intraoperative/
    └── specimen00_left_femur.xyz

./checkpoints_NeuralUDF/UltraBonesHip/
└── specimen00_left_femur/
    └── checkpoints/
        └── ckpt_030000_NeuralUDF.pth
```

Notes:
- If the UDF checkpoint is missing, `main.py` will train it automatically before registration.

### The UltraBonesHip dataset

The full UltraBonesHip dataset includes the `left_femur`, `right_femur`, and `pelvis` for five specimens: `specimen00`, `specimen01`, `specimen02`, `specimen03`, and `specimen04`. This early-access release provides point clouds for `specimen00` and `specimen01`. The final release will include the complete dataset, including raw B-mode ultrasound data, in a format similar to our prior work, [UltraBones100k](https://github.com/luohwu/UltraBones100k).


## 6. Configuration

The pipeline is configured through a YAML file. By default it uses `configs/example.yaml`, but you can pass a different file at runtime with `--config`.

Key fields:

- `show_registration_visualization`: opens an Open3D viewer when `true`
- `specimen_id` and `anatomy`: select the case to run
- `preoperative_data_dir`: directory containing preoperative meshes
- `intraoperative_data_dir`: directory containing intraoperative point clouds
- `checkpoint_root`: root directory for NeuralUDF checkpoints
- `udf_config_file`: UDF network definition
- `num_heads`: number of hypothesis generation heads
- `size_coarse_pcd`: number of sampled points used for the coarse registration branch
- `refine_heads`: number of top candidate transformations kept for refinement
- `num_optimization_steps`: number of optimization iterations used during registration
- `num_runs`: number of randomized perturbation trials





## 7. Running

From the repository root, run the provided example configuration with:

```bash
python main.py --config configs/example.yaml
```

For a failure case, run:
```bash
python main.py --config configs/failure_case.yaml
```



## 8. Code reference

This code is adapted from the following repositories (many thanks to the contributors):
- FUNSR: https://github.com/chenhbo/FUNSR
- UltraBoneUDF: https://github.com/luohwu/UltraBoneUDF



## 9. Questions or Feedback?

If you have questions, you can open a new GitHub issue within this repository, and we'll get back to you!
