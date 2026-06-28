"""Bone segmentation inference.

Wraps the pretrained network used to produce the 2D bone masks. The checkpoint
is a *full pickled model object* (a ``segmentation_models_pytorch`` FPN with a
ResNet encoder), so it is loaded with ``torch.load(weights_only=False)`` and
``segmentation_models_pytorch`` + ``torchvision`` must be importable for the
pickled classes to resolve. Preprocessing matches the original
``predict_on_images.py`` exactly (grayscale, resize-to-256, fixed mean/std
normalisation, sigmoid > threshold, then bilinear resize back to native size).
"""

import os
import shutil

# Normalisation constants from the segmentation training pipeline (do not change;
# they must match what the checkpoint was trained with).
_NORM_MEAN = 0.17475835978984833
_NORM_STD = 0.16475939750671387
_INPUT_SIZE = 256


def _preprocess_image(path):
    """Load and preprocess one image to a normalised ``[1, 256, 256]`` tensor.

    Returns ``(tensor, height, width)`` or ``None`` if the image is unreadable.
    This is the single, exact-match preprocessing used by both ``predict_mask``
    and the batched loader: one disk read (PIL, grayscale), to-tensor, resize to
    256 (bilinear), fixed mean/std normalisation -- matching ``predict_on_images.py``.
    Imports are local so this runs inside DataLoader worker processes.
    """
    import torchvision.transforms.functional as TF
    from PIL import Image
    from torchvision.transforms.functional import InterpolationMode

    try:
        img = Image.open(path).convert("L")
    except (OSError, ValueError, SyntaxError):
        return None
    width, height = img.size  # PIL gives (W, H); native size for the resize-back
    image = TF.to_tensor(img)
    image = TF.resize(
        image, [_INPUT_SIZE, _INPUT_SIZE], interpolation=InterpolationMode.BILINEAR
    )
    image = TF.normalize(image, mean=_NORM_MEAN, std=_NORM_STD)
    return image, height, width


class _FrameDataset:
    """Map-style dataset over image paths (no torch subclassing needed).

    Each item is the preprocessed tensor plus native size, so disk reads and CPU
    preprocessing are parallelised across DataLoader worker processes and
    prefetched while the GPU is busy with the previous batch. Unreadable frames
    yield ``None`` and are dropped by the collate function.
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        result = _preprocess_image(self.paths[i])
        if result is None:
            return i, None, 0, 0
        tensor, height, width = result
        return i, tensor, height, width


def _collate_frames(batch):
    """Stack readable frames in a batch; drop unreadable ones. ``None`` if all bad."""
    import torch

    items = [item for item in batch if item[1] is not None]
    if not items:
        return None
    indices = [item[0] for item in items]
    images = torch.stack([item[1] for item in items], dim=0)
    sizes = [(item[2], item[3]) for item in items]
    return indices, images, sizes


class _FrameDatasetNative:
    """Like ``_FrameDataset`` but also returns the native grayscale image (uint8).

    Used by the shadow denoiser, which needs native image intensities to measure
    the acoustic shadow below each bone pixel.
    """

    def __init__(self, paths):
        self.paths = list(paths)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        import numpy as np
        from PIL import Image

        result = _preprocess_image(self.paths[i])
        if result is None:
            return i, None, None, 0, 0
        tensor, height, width = result
        try:
            native = np.asarray(Image.open(self.paths[i]).convert("L"), dtype=np.uint8)
        except (OSError, ValueError, SyntaxError):
            return i, None, None, 0, 0
        return i, tensor, native, height, width


def _collate_frames_with_native(batch):
    """Collate for :meth:`BoneSegmenter.predict_masks_images`."""
    import torch

    items = [item for item in batch if item[1] is not None]
    if not items:
        return None
    indices = [item[0] for item in items]
    images = torch.stack([item[1] for item in items], dim=0)
    natives = [item[2] for item in items]
    sizes = [(item[3], item[4]) for item in items]
    return indices, images, natives, sizes


# Pretrained checkpoint hosted on the Hugging Face Hub (auto-downloaded if absent).
HF_REPO_ID = "luohwu/UltraBones100k_segmentation"
HF_FILENAME = "epoch_30_leave_12_out.pth"
HF_RESOLVE_URL = f"https://huggingface.co/{HF_REPO_ID}/resolve/main/{HF_FILENAME}"


def _download_url(url, dst, log=print):
    """Stream-download ``url`` to ``dst`` (atomic via a .part temp file)."""
    import urllib.request

    tmp = dst + ".part"
    with urllib.request.urlopen(url) as resp:  # follows redirects (HF -> CDN)
        total = int(resp.headers.get("Content-Length", 0))
        done = 0
        next_report = 16 * 1024 * 1024
        with open(tmp, "wb") as f:
            while True:
                chunk = resp.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    pct = f" ({100 * done / total:.0f}%)" if total else ""
                    log(f"  downloaded {done / 1e6:.0f} MB{pct}")
                    next_report += 16 * 1024 * 1024
    os.replace(tmp, dst)


def ensure_checkpoint(checkpoint_path, log=print):
    """Return ``checkpoint_path``, downloading it from Hugging Face if missing.

    Tries ``huggingface_hub`` first (caching/integrity/resume), then falls back to
    a direct download from the resolve URL.
    """
    if os.path.isfile(checkpoint_path):
        return checkpoint_path

    log(
        f"Segmentation checkpoint not found at {checkpoint_path}; "
        f"downloading from Hugging Face ({HF_REPO_ID}) ..."
    )
    os.makedirs(os.path.dirname(os.path.abspath(checkpoint_path)), exist_ok=True)

    try:
        from huggingface_hub import hf_hub_download

        cached = hf_hub_download(repo_id=HF_REPO_ID, filename=HF_FILENAME)
        if os.path.abspath(cached) != os.path.abspath(checkpoint_path):
            shutil.copyfile(cached, checkpoint_path)
    except Exception as exc:  # noqa: BLE001 - fall back to a plain HTTP download
        log(f"  huggingface_hub unavailable/failed ({exc}); using direct download.")
        try:
            _download_url(HF_RESOLVE_URL, checkpoint_path, log=log)
        except Exception as exc2:  # noqa: BLE001
            raise FileNotFoundError(
                f"Could not obtain the segmentation checkpoint. Download it manually "
                f"from {HF_RESOLVE_URL} and place it at {checkpoint_path}. "
                f"Error: {exc2}"
            ) from exc2

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint download did not produce a file at {checkpoint_path}."
        )
    log(f"  checkpoint ready: {checkpoint_path}")
    return checkpoint_path


class BoneSegmenter:
    """Loads the pretrained segmentation model once and predicts masks per image."""

    def __init__(
        self,
        checkpoint_path,
        device=None,
        threshold=0.5,
        batch_size=16,
        num_workers=8,
        amp=False,
    ):
        try:
            import cv2
            import torch
            import torchvision  # noqa: F401  (registers torchvision.* classes for unpickling)
            import torchvision.transforms.functional as TF
            import segmentation_models_pytorch  # noqa: F401  (registers smp.* classes for unpickling)
            from PIL import Image
            from torchvision.transforms.functional import InterpolationMode
        except ImportError as exc:  # fail fast with an actionable message
            raise ImportError(
                "BoneSegmenter needs torch, torchvision, segmentation-models-pytorch, "
                "opencv-python and pillow. Install them (see reconstruction/README.md). "
                f"Import error: {exc}"
            ) from exc

        ensure_checkpoint(checkpoint_path)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.threshold = float(threshold)
        self.batch_size = max(1, int(batch_size))
        self.num_workers = max(0, int(num_workers))
        # AMP (fp16) only helps on CUDA; off by default so masks match fp32 exactly.
        self.amp = bool(amp) and self.device.type == "cuda"

        # Stash module handles so per-frame inference does not re-import.
        self._torch = torch
        self._cv2 = cv2
        self._tf = TF
        self._image = Image
        self._interp = InterpolationMode

        # Input size is fixed at 256x256, so let cuDNN pick the fastest kernels once.
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

        try:
            model = torch.load(
                checkpoint_path, map_location=self.device, weights_only=False
            )
        except Exception as exc:  # pragma: no cover - environment dependent
            raise RuntimeError(
                "Failed to unpickle the segmentation checkpoint. This usually means "
                "the installed segmentation-models-pytorch / torchvision version "
                "differs from the one used to save it. Pin a compatible "
                "segmentation-models-pytorch version (see reconstruction/README.md). "
                f"Original error: {exc}"
            ) from exc

        self.model = model.to(self.device).eval()

    def _forward_threshold(self, images):
        """Run the model on a ``[B, 1, 256, 256]`` batch -> ``[B, 256, 256]`` uint8.

        Values are ``{0, 255}`` (sigmoid > threshold). One device transfer per
        batch, so this is where batching pays off.
        """
        torch = self._torch
        images = images.to(self.device, non_blocking=True)
        with torch.no_grad(), torch.autocast(
            self.device.type, dtype=torch.float16, enabled=self.amp
        ):
            outputs = torch.sigmoid(self.model(images))
        pred = (outputs[:, 0] > self.threshold).to(torch.uint8).mul_(255)
        return pred.cpu().numpy()

    def predict_mask(self, image_path):
        """Return a native-resolution ``uint8`` mask, or ``None`` if unreadable.

        The mask is the sigmoid>threshold prediction resized back to the original
        image size with bilinear interpolation (matching ``predict_on_images.py``),
        so it may contain intermediate values along edges; downstream code
        binarises with ``> 0``. Prefer :meth:`predict_masks` for whole records --
        it batches the GPU forward and loads images in parallel.
        """
        result = _preprocess_image(image_path)
        if result is None:
            return None
        image, height, width = result
        pred = self._forward_threshold(image.unsqueeze(0))[0]
        # Resize back to native size (bilinear, as in predict_on_images.py).
        return self._cv2.resize(pred, (width, height))

    def predict_masks(self, image_paths, batch_size=None, num_workers=None):
        """Yield ``(index, native_uint8_mask)`` for each readable image in order.

        Images are loaded and preprocessed in parallel worker processes and the
        GPU forward pass is batched, so this is far faster than calling
        :meth:`predict_mask` per frame on an I/O-bound dataset. ``index`` is the
        position in ``image_paths``; unreadable images are skipped (not yielded).
        """
        torch = self._torch
        cv2 = self._cv2
        from torch.utils.data import DataLoader

        paths = list(image_paths)
        if not paths:
            return
        batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        num_workers = (
            self.num_workers if num_workers is None else max(0, int(num_workers))
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_frames,
            pin_memory=(self.device.type == "cuda"),
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = False
        loader = DataLoader(_FrameDataset(paths), **loader_kwargs)

        for batch in loader:
            if batch is None:
                continue
            indices, images, sizes = batch
            preds = self._forward_threshold(images)
            for k, idx in enumerate(indices):
                height, width = sizes[k]
                yield idx, cv2.resize(preds[k], (width, height))

    def predict_masks_images(self, image_paths, batch_size=None, num_workers=None):
        """Like :meth:`predict_masks` but also yields the native grayscale image.

        Yields ``(index, native_uint8_mask, native_uint8_gray)`` per readable image.
        The native image is needed for image-based denoising cues (e.g. the
        acoustic-shadow feature: mean intensity in a window *below* each bone
        pixel). Used only when the shadow denoiser is enabled; the default
        pipeline keeps using the lighter :meth:`predict_masks`.
        """
        torch = self._torch
        cv2 = self._cv2
        import numpy as np
        from torch.utils.data import DataLoader

        paths = list(image_paths)
        if not paths:
            return
        batch_size = self.batch_size if batch_size is None else max(1, int(batch_size))
        num_workers = (
            self.num_workers if num_workers is None else max(0, int(num_workers))
        )

        loader_kwargs = dict(
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=_collate_frames_with_native,
            pin_memory=(self.device.type == "cuda"),
        )
        if num_workers > 0:
            loader_kwargs["prefetch_factor"] = 4
            loader_kwargs["persistent_workers"] = False
        loader = DataLoader(_FrameDatasetNative(paths), **loader_kwargs)

        for batch in loader:
            if batch is None:
                continue
            indices, images, natives, sizes = batch
            preds = self._forward_threshold(images)
            for k, idx in enumerate(indices):
                height, width = sizes[k]
                mask = cv2.resize(preds[k], (width, height))
                native = natives[k]
                if native.shape != (height, width):
                    native = cv2.resize(native, (width, height))
                yield idx, mask, native

    def image_size(self, image_path):
        """Return ``(height, width)`` of an image without decoding it fully."""
        try:
            with self._image.open(image_path) as img:
                width, height = img.size
        except (OSError, ValueError, SyntaxError):
            return None
        return height, width
