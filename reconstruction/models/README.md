# Segmentation model

`epoch_30_leave_12_out.pth` is the pretrained 2D bone-segmentation network used by the
`reconstruction` package to segment ultrasound frames before projecting them to
3D (see [`../README.md`](../README.md)).

## Provenance

- **Source:** https://github.com/luohwu/UltraBones100k
- **Architecture:** a `segmentation_models_pytorch` FPN with a ResNet encoder,
  saved as a full pickled `nn.Module` (loaded with `torch.load(weights_only=False)`,
  so `segmentation-models-pytorch` must be importable — see `reconstruction/requirements.txt`).
- **Training scheme:** leave-one-specimen-out. **This checkpoint leaves
  specimen12 out** — specimen12 has the fewest frames, so excluding it from
  training sacrifices the least data while still providing a held-out specimen.

## Storage

The weights are **not committed** (git-ignored). They are hosted on the
Hugging Face Hub and **downloaded automatically on first use** to
`reconstruction/models/epoch_30_leave_12_out.pth`:

- Hub repo: https://huggingface.co/luohwu/UltraBones100k_segmentation
- File: https://huggingface.co/luohwu/UltraBones100k_segmentation/blob/main/epoch_30_leave_12_out.pth

`reconstruction/segmentation.py::ensure_checkpoint` fetches it (via
`huggingface_hub`, falling back to a direct download) when `checkpoint_path` is
missing. To install manually, just drop `epoch_30_leave_12_out.pth` into this folder.
