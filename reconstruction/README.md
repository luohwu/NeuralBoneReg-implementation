# `reconstruction` — 3D ultrasound point-cloud reconstruction

Rebuilds the intraoperative bone point clouds used by NeuralBoneReg directly
from raw UltraBonesHip data (ultrasound images + tracked poses + a calibration
preset). It is a port of the reconstruction logic in
`UltrasoundDatasetCollection` into a self-contained package.

## Pipeline

For every ultrasound frame in a record:

1. **Segment** the bone with the pretrained network (`segmentation.py`).
2. **Skeletonize** the mask to the bone mid-line (`projection.py`).
3. **Project** the bone pixels into CT/world millimetres using the tracked pose
   and the calibration preset:

   ```
   xyz_world = T_tracking @ T_idealProbe_USimage @ T_scale @ [col+0.5, row+0.5, 0, 1]
   ```

Per record the points are accumulated and randomly downsampled (`record.py`); no
CT mesh is used at this stage.

Per-record clouds are then **merged per anatomy**, **denoised CT-free**, and
downsampled to `merge_downsample` points (`merge.py`). Two CT-free denoisers are
available via `denoise_method`:

- `shadow` **(default)**: an **acoustic-shadow** image cue
  (`filtering.denoise_with_shadow`) that keeps points sitting above a dark acoustic
  shadow (true bone blocks the beam; reverberation/soft-tissue detections are bright
  below) — per-anatomy (absolute shadow for femur, depth-normalized for pelvis).
  Preserves the full bone extent (coverage); uses the heavier image-carrying
  inference path. Stays CT-free (the cue is computed from the
  US image, never the CT mesh).
- `dbscan`: the prior geometric DBSCAN cluster-keep + statistical outlier removal
  (`filtering.denoise_point_cloud`) — faster (no image features); a fallback.

For each anatomy the package writes a `main_UltraBonesHip.py`-ready, frame-consistent pair:

- `intraoperative_dir/specimenNN_<anatomy>.xyz` — the reconstructed US cloud.
- `preoperative_dir/specimenNN_<anatomy>.stl` — the matching CT bone mesh, copied
  from `CT_bone_segmentations/<anatomy>.stl` (when `export_preoperative: true`).

## Coordinate frame (important)

Reconstructed clouds live in the **CT-segmentation frame** — they align with
`<dataset_root>/specimenNN/CT_bone_segmentations/<anatomy>.stl`.

Because the exported preoperative mesh **is** that CT segmentation, the
preop/intraop pair is in one frame and registers correctly — so point them at
`main_UltraBonesHip.py` directly (`preoperative_data_dir: ./data/UltraBonesHip/preoperative`,
`intraoperative_data_dir: ./data/UltraBonesHip/intraoperative`). Note this is a *different*
frame than the original shipped clouds, which were placed per-instance; that is
expected and intentional (we align to the CT segmentations).

## Design choices baked in (configurable)

- **Live segmentation** on every frame (no dependency on precomputed labels).
- **All frames** from `poses.csv` with **raw tracking poses** (no
  dependency on the intersected-frame CSVs and no CT-based pose refinement).
  Because non-bone frames produce spurious masks, the merged cloud is cleaned by a
  **CT-free geometric denoiser** (`denoise_*` keys; DBSCAN cluster-keep + SOR). It is
  CT-free by design — it never uses the registration target mesh.

## Install

The core NeuralBoneReg environment already provides `torch`, `open3d`, `numpy`,
`scipy`, `pyyaml`. Add the extra dependencies:

```bash
pip install -r reconstruction/requirements.txt
```

The checkpoint is a **full pickled `segmentation_models_pytorch` FPN**, so
`segmentation-models-pytorch` (and a matching `torchvision`) must be importable
to load it. If `torch.load` fails to unpickle, the installed
`segmentation-models-pytorch` version differs from the one used to save the
model — pin a compatible version (0.3.3 is known to work).

The weights are **downloaded automatically on first use** from the Hugging Face
Hub ([luohwu/UltraBones100k_segmentation](https://huggingface.co/luohwu/UltraBones100k_segmentation))
to `checkpoint_path` if it does not already exist — see
`reconstruction/models/README.md`.

## Usage

The raw dataset is read from `dataset_root` (default `/mnt/UltraBonesHip`, the
devcontainer mount). The checkpoint (`reconstruction/models/epoch_30_leave_12_out.pth`) is
fetched automatically if missing, so you can just run:

```bash
# Full run over all configured specimens/anatomies
python -m reconstruction --config reconstruction/conf/reconstruction.yaml

# Bare host (data not at /mnt): override the dataset root
python -m reconstruction --dataset-root /path/to/UltraBonesHip --specimen specimen02

# Fast read-only validation (one anatomy of specimen02, coarse stride):
python -m reconstruction --smoke --dataset-root /path/to/UltraBonesHip
```

`--smoke` loads the model, reconstructs `specimen02` left femur with a coarse
frame stride into `reconstruction/_smoke_output/`, and reports the mean surface
distance to the CT-segmentation femur as a sanity check (PASS/WARN). On the
devcontainer (`/mnt/UltraBonesHip` present) the `--dataset-root` override is
unnecessary.

## Configuration highlights (`conf/reconstruction.yaml`)

| Key | Meaning |
|---|---|
| `dataset_root` | Raw data root (default `/mnt/UltraBonesHip`; override with `--dataset-root`). |
| `checkpoint_path` | Pickled segmentation FPN (default `reconstruction/models/epoch_30_leave_12_out.pth`). |
| `specimen_to_preset_map` | Maps each specimen to a calibration preset (specimen00–04 → `SpineUS`). |
| `pose_csv_name` | Per-record pose CSV (`poses.csv`, raw poses). |
| `seg_batch_size` / `seg_num_workers` | Inference throughput: frames per GPU batch / parallel image-loading processes (`seg_num_workers: null` → auto from CPU count). See *Performance* below. |
| `seg_amp` | fp16 autocast for the segmentation forward pass (CUDA-only; faster, masks differ by a few edge pixels). Off by default. |
| `postprocess` | `skeleton_only` (default) or `icp_clean` (border + largest-component cleanup). |
| `merge_map` | Output anatomy → source sweep folders to union. |
| `denoise_enable` / `denoise_dbscan_eps` / `denoise_keep_frac` / `sor_*` | CT-free denoiser (DBSCAN cluster-keep + statistical outlier removal) applied per merged anatomy. |
| `raw_downsample` / `merge_downsample` | Point caps per record / per merged anatomy. |
| `strict_image_size` | Skip frames whose native size ≠ the preset size. |
| `intraoperative_dir` / `preoperative_dir` | Output dirs for the `.xyz` clouds and copied `.stl` preop meshes. |
| `export_preoperative` | Also copy `CT_bone_segmentations/<anatomy>.stl` as the preop model. |
| `overwrite` | Existing output files are preserved unless this is `true`. |

## Performance

The dominant cost is **decoding ultrasound frames from disk**, not the GPU
(profiling a 1249-frame record: ~80 % image I/O, ~20 % GPU at batch size 1).
Segmentation therefore loads and preprocesses frames in parallel worker processes
and runs the network on **batches** (`seg_num_workers` / `seg_batch_size`), each
image read from disk exactly once. On an RTX 5080 this takes a full record from
~14 to ~240 frames/s (**~17× faster**) with no change to the reconstructed cloud.
Batched GPU math flips a handful of mask edge pixels vs. the per-frame path
(~0.0005 % of pixels), well
below the surface-accuracy floor. Tuning: raise `seg_batch_size` until VRAM-bound,
raise `seg_num_workers` until disk/CPU-bound; `seg_amp: true` adds fp16 for a
further GPU speedup.

## Module map

| File | Responsibility |
|---|---|
| `config.py` | Load YAML → `ReconstructionConfig`; path resolution; merge-map helpers. |
| `calibration.py` | `construct_ultrasound_calibration_matrix` (image-plane-centred). |
| `segmentation.py` | `BoneSegmenter`: load the pickled FPN, predict native-size masks. |
| `projection.py` | Mask post-processing + `pixels_to_world`. |
| `filtering.py` | CT-free denoiser (DBSCAN cluster-keep + statistical outlier removal), downsample. |
| `record.py` | `reconstruct_one_record`. |
| `merge.py` | Union per-record clouds → `.xyz`; copy CT mesh → preop `.stl`. |
| `pipeline.py` | Orchestrate specimens × sweeps × records → merge. |
| `__main__.py` | CLI and `--smoke` validation. |

Reuses `utilities.converter.vectorToMatrix` from the repo (identical Euler
convention); the CUDA Chamfer extension is **not** used here.
