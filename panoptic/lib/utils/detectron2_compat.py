"""Small, API-compatible fallbacks for the Detectron2 pieces used here.

The Exchanger Mask2Former port only needs GroupNorm construction and the two
PointRend sampling helpers below.  Keeping these functions local avoids a
compiled Detectron2 dependency while preserving its coordinate convention and
sampling algorithm.
"""

from __future__ import annotations

from collections.abc import Callable

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def get_norm(norm: str | Callable[[int], nn.Module] | None, out_channels: int):
    if norm in (None, ""):
        return None
    if callable(norm):
        return norm(out_channels)
    if norm == "GN":
        return nn.GroupNorm(32, out_channels)
    if norm == "BN":
        return nn.BatchNorm2d(out_channels)
    if norm == "SyncBN":
        return nn.SyncBatchNorm(out_channels)
    raise KeyError(f"Unsupported normalization type: {norm}")


def point_sample(
    input: Tensor,
    point_coords: Tensor,
    **kwargs,
) -> Tensor:
    """Equivalent to Detectron2 PointRend's normalized point sampler."""

    add_dimension = False
    if point_coords.dim() == 3:
        add_dimension = True
        point_coords = point_coords.unsqueeze(2)
    output = F.grid_sample(input, 2.0 * point_coords - 1.0, **kwargs)
    if add_dimension:
        output = output.squeeze(3)
    return output


def get_uncertain_point_coords_with_randomness(
    coarse_logits: Tensor,
    uncertainty_func: Callable[[Tensor], Tensor],
    num_points: int,
    oversample_ratio: float,
    importance_sample_ratio: float,
) -> Tensor:
    """Equivalent to Detectron2's uncertainty-guided PointRend sampler."""

    if oversample_ratio < 1:
        raise ValueError("oversample_ratio must be at least 1")
    if not 0 <= importance_sample_ratio <= 1:
        raise ValueError("importance_sample_ratio must be in [0, 1]")

    num_boxes = coarse_logits.shape[0]
    num_sampled = int(num_points * oversample_ratio)
    point_coords = torch.rand(
        num_boxes,
        num_sampled,
        2,
        device=coarse_logits.device,
    )
    point_logits = point_sample(
        coarse_logits,
        point_coords,
        align_corners=False,
    )
    point_uncertainties = uncertainty_func(point_logits)

    num_uncertain_points = int(importance_sample_ratio * num_points)
    num_random_points = num_points - num_uncertain_points
    idx = torch.topk(
        point_uncertainties[:, 0, :],
        k=num_uncertain_points,
        dim=1,
    ).indices
    shift = num_sampled * torch.arange(
        num_boxes,
        dtype=torch.long,
        device=coarse_logits.device,
    )
    idx = idx + shift[:, None]
    point_coords = point_coords.reshape(-1, 2)[idx.reshape(-1), :].reshape(
        num_boxes,
        num_uncertain_points,
        2,
    )
    if num_random_points > 0:
        point_coords = torch.cat(
            (
                point_coords,
                torch.rand(
                    num_boxes,
                    num_random_points,
                    2,
                    device=coarse_logits.device,
                ),
            ),
            dim=1,
        )
    return point_coords
