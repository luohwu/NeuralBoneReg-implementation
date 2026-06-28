# `data/`

This directory is **empty in the repository** except for this file — datasets are
large and are not committed (`.gitignore` keeps only `data/README.md`). It holds
the processed, `main_UltraBonesHip.py`-ready data, produced from the raw dataset.

## Layout

Data is grouped per dataset, so multiple datasets can coexist (e.g. a future
`data/Other_dataset/` alongside `data/UltraBonesHip/`):

```
data/
  UltraBonesHip/
    preoperative/    specimenNN_<anatomy>.stl   # CT bone meshes  (copied from the raw CT segmentations)
    intraoperative/  specimenNN_<anatomy>.xyz   # reconstructed US point clouds
```

`main_UltraBonesHip.py` reads these via `preoperative_data_dir: ./data/UltraBonesHip/preoperative` and
`intraoperative_data_dir: ./data/UltraBonesHip/intraoperative` (see `configs/UltraBonesHip.yaml`).
Both sides are in the **CT-segmentation frame**, so the pair registers correctly.

## How it gets populated

1. The **raw** UltraBonesHip dataset is bind-mounted at **`/mnt/UltraBonesHip`**
   by the devcontainer (set the host path via `ULTRABONESHIP_DIR`; see
   [`.devcontainer/devcontainer.json`](../.devcontainer/devcontainer.json)). On a
   bare host, pass `--dataset-root` instead. Expected raw layout:

   ```
   /mnt/UltraBonesHip/specimenNN/
     ultrasound_records/<anatomy>_<axial|coronal>/record*/{UltrasoundImages/, poses.csv}
     CT_bone_segmentations/{left_femur,right_femur,left_pelvis,right_pelvis,pelvis,all}.stl
   ```

2. The reconstruction package fills `data/UltraBonesHip/preoperative` and `data/UltraBonesHip/intraoperative`:

   ```bash
   python -m reconstruction --config reconstruction/conf/reconstruction.yaml
   ```

   See [`reconstruction/README.md`](../reconstruction/README.md).
