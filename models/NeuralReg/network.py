from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np

from utilities.OwnPytorch3d import quaternion_to_matrix, quaternion_multiply


FPS_CACHE_ROOT_CANDIDATE_PATHS = (
    Path(__file__).resolve().parents[0] / "result_fps",
)


def _load_cached_fps_quaternions(k):
    file_name = f"heads_{k}.pt"
    for root in FPS_CACHE_ROOT_CANDIDATE_PATHS:
        cache_path = root / file_name
        if cache_path.exists():
            payload = torch.load(cache_path, map_location="cpu")
            return F.normalize(payload.float(), dim=1)
    return None


def _save_cached_fps_quaternions(k, quaternions):
    file_name = f"heads_{k}.pt"
    for root in FPS_CACHE_ROOT_CANDIDATE_PATHS:
        try:
            root.mkdir(parents=True, exist_ok=True)
            cache_path = root / file_name
            torch.save(quaternions.detach().cpu(), cache_path)
            return
        except Exception:
            continue


def sample_deterministic_quaternions_so3(k):
    """Build a deterministic, near-uniform set of k unit quaternions on SO(3).

    Uses farthest-point sampling (FPS) over a dense normalized quaternion pool,
    optimized for SO(3) distance via antipodal-invariant |dot(q1, q2)|.
    Results are cached on disk and reused in subsequent runs.
    """
    if k <= 0:
        raise ValueError("k must be > 0")

    cached = _load_cached_fps_quaternions(k)
    if cached is not None:
        return cached

    generator = torch.Generator(device="cpu")
    generator.manual_seed(2025)

    # Draw a large candidate pool once, then greedily keep the next quaternion
    # that is most dissimilar from all previously selected ones.
    num_candidates = max(20000, 1000 * min(k,20000))
    candidates = torch.randn((num_candidates, 4), generator=generator)
    candidates = F.normalize(candidates, dim=1)

    selected = torch.empty((k,), dtype=torch.long)
    selected[0] = 0

    dots = torch.abs(candidates @ candidates[selected[0]].unsqueeze(1)).squeeze(1)
    min_dissimilarity = 1.0 - dots

    for idx in range(1, k):
        # if idx%10==0:
        #     print(idx)
        next_index = torch.argmax(min_dissimilarity)
        selected[idx] = next_index
        dots = torch.abs(candidates @ candidates[next_index].unsqueeze(1)).squeeze(1)
        min_dissimilarity = torch.minimum(min_dissimilarity, 1.0 - dots)

    q = F.normalize(candidates[selected], dim=1)
    _save_cached_fps_quaternions(k, q)
    return q




class NeuralReg(nn.Module):
    def __init__(
        self,
        udf_network,
        num_heads=10,
        gt=np.eye(4),
        use_amp=True,
        amp_dtype=torch.float16,
        use_coarse_refine=True,
        refine_topk_heads=64,
        score_mode="mean",
        trim_keep=1.0,
    ):
        super(NeuralReg, self).__init__()
        self.num_heads = num_heads
        self.udf_network = udf_network
        self.use_amp = use_amp
        self.amp_dtype = amp_dtype
        self.use_coarse_refine = use_coarse_refine
        self.refine_topk_heads = refine_topk_heads
        # Per-head score aggregation over points. "mean" (default) reproduces the
        # original plain mean-UDF score. "trimmed" keeps only the closest
        # ``trim_keep`` fraction of points per head and averages those, so a
        # residual off-bone fragment (high UDF) cannot bias the pose -- the
        # registration-time analogue of keep-largest, applied uniformly.
        self.score_mode = score_mode
        self.trim_keep = float(trim_keep)

        # This module starts from a learnable/global latent seed and optimizes a
        # large set of pose hypotheses against the frozen UDF network.
        self.feature_initialization = torch.tanh(torch.randn((1, 128), requires_grad=False)).cuda()

        self.t_head = nn.Linear(128, num_heads * 3)
        self.q_head = nn.Linear(128, num_heads * 4)

        # Initialize pose heads by FPS
        init_quaternions = sample_deterministic_quaternions_so3(self.num_heads)
        self.init_quaternions = nn.Parameter(init_quaternions, requires_grad=True)  # [num_heads, 4]

        # Weight init
        init.kaiming_normal_(self.q_head.weight)
        init.kaiming_normal_(self.t_head.weight)
        init.constant_(self.t_head.bias, 0.01)
        init.constant_(self.q_head.bias, 0.0)
        self.encoder = nn.Sequential(
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
            nn.Linear(128, 128), nn.Tanh(),
        )

    def _aggregate(self, udf_per_point):
        """Aggregate per-head per-point UDF ``[num_heads, M]`` -> per-head score
        ``[num_heads]`` (lower = better fit). Modes:
          mean     - original equal-weight average.
          trimmed  - mean of the closest ``trim_keep`` fraction (ignore off-bone tail).
          median   - robust middle (insensitive to both tails).
          max      - worst point (minimax; forces all points onto the surface).
          meanmax  - 0.5*mean + 0.5*max (average fit + worst-point penalty).
        """
        mode = self.score_mode
        if mode == "trimmed" and self.trim_keep < 1.0:
            m = udf_per_point.shape[1]
            k = max(1, int(round(m * self.trim_keep)))
            closest, _ = torch.topk(udf_per_point, k, dim=1, largest=False)
            return closest.mean(1)
        if mode == "median":
            return udf_per_point.median(dim=1).values
        if mode == "max":
            return udf_per_point.max(dim=1).values
        if mode == "meanmax":
            return 0.5 * udf_per_point.mean(1) + 0.5 * udf_per_point.max(dim=1).values
        return udf_per_point.mean(1)

    def forward(self, x, x_downsampled):
        pcd_US_moved_tensor = x  # [1, 3, N]
        pcd_US_moved_tensor_downsampled = x_downsampled  # [1, 3, M]
        use_amp = self.use_amp and pcd_US_moved_tensor.is_cuda

        with torch.autocast(device_type="cuda", dtype=self.amp_dtype, enabled=use_amp):
            # Produce one translation and one quaternion residual per head.
            feature_initialization = self.encoder(self.feature_initialization)
            t = self.t_head(feature_initialization).view(-1, self.num_heads, 3).squeeze(0)  # [num_heads, 3]
            q_pred = self.q_head(feature_initialization).view(-1, self.num_heads, 4).squeeze(0)  # [num_heads, 4]

            # Bound translations to a compact normalized range and keep
            # quaternions on the unit sphere.
            t = torch.tanh(t)
            q_pred = F.normalize(torch.tanh(q_pred), dim=-1)
            q = quaternion_multiply(q_pred, self.init_quaternions)

            rotation_matrices = quaternion_to_matrix(q)  # [num_heads, 3, 3]

            if self.use_coarse_refine:
                # First evaluate all heads on a cheaper downsampled cloud.
                transformed_US_pcd_coarse = (
                    torch.matmul(rotation_matrices, pcd_US_moved_tensor_downsampled) + t.unsqueeze(-1)
                ).transpose(1, 2)
                transformed_US_pcd_udf = self._aggregate(self.udf_network.udf(transformed_US_pcd_coarse).squeeze(-1))  # [num_heads]

                refine_k = self.refine_topk_heads
                if refine_k is not None and 0 < refine_k < self.num_heads:
                    # Re-score only the most promising heads on the full point
                    # cloud to save UDF evaluations without losing accuracy.
                    refine_idx = transformed_US_pcd_udf.topk(k=refine_k, largest=False).indices
                    transformed_US_pcd_refined = (
                        torch.matmul(rotation_matrices[refine_idx], pcd_US_moved_tensor) + t[refine_idx].unsqueeze(-1)
                    ).transpose(1, 2)
                    refined_scores = self._aggregate(self.udf_network.udf(transformed_US_pcd_refined).squeeze(-1))
                    transformed_US_pcd_udf = transformed_US_pcd_udf.clone()
                    transformed_US_pcd_udf[refine_idx] = refined_scores
            else:
                transformed_US_pcd = (
                    torch.matmul(rotation_matrices, pcd_US_moved_tensor) + t.unsqueeze(-1)
                ).transpose(1, 2)
                transformed_US_pcd_udf = self._aggregate(self.udf_network.udf(transformed_US_pcd).squeeze(-1))  # [num_heads]


        udf_best, best_indices = transformed_US_pcd_udf.min(dim=0)
        t_best = t[best_indices]
        q_best = q[best_indices]

        return transformed_US_pcd_udf, t, q, t_best, q_best
