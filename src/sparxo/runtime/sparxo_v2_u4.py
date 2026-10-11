"""SPARXO v2.U.4: conservative boundary-closed-loop S2 segmentation."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .optical import (
    SPARXOOpticalUNetBase,
    SPARXOV2UNetDecoder,
    UNetDoubleConv,
    _group_count,
)


def _shift4(feature: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Return north, south, west and east neighbours with zero image borders."""

    return (
        F.pad(feature[:, :, :-1], (0, 0, 1, 0)),
        F.pad(feature[:, :, 1:], (0, 0, 0, 1)),
        F.pad(feature[:, :, :, :-1], (1, 0, 0, 0)),
        F.pad(feature[:, :, :, 1:], (0, 1, 0, 0)),
    )


def _symmetric_affinity_gates(logits: Tensor) -> Tensor:
    """Convert directed logits into undirected four-neighbour probabilities."""

    gates = logits.sigmoid()
    vertical = 0.5 * (gates[:, 0:1, 1:] + gates[:, 1:2, :-1])
    horizontal = 0.5 * (gates[:, 2:3, :, 1:] + gates[:, 3:4, :, :-1])
    north = F.pad(vertical, (0, 0, 1, 0))
    south = F.pad(vertical, (0, 0, 0, 1))
    west = F.pad(horizontal, (1, 0, 0, 0))
    east = F.pad(horizontal, (0, 1, 0, 0))
    return torch.cat((north, south, west, east), dim=1)


class BoundaryClosedLoopUNetDecoder(SPARXOV2UNetDecoder):
    """U-Net whose shape predictions actively refine semantic predictions.

    The strong v2.U.1 U-Net remains the semantic trunk. A shallow full-
    resolution shape stream combines its final feature with a direct projection
    of the temporally aggregated S2 feature. Supervised same-region affinities
    gate one local propagation step, while boundary and uncertainty gates limit
    the class residual to pixels that need high-frequency correction.
    """

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        dropout: float,
        shape_depth: int = 2,
    ) -> None:
        super().__init__(in_channels, num_classes, base_channels, dropout)
        if shape_depth < 1:
            raise ValueError("shape_depth must be positive")

        self.temporal_detail = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, 1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.GELU(),
        )
        shape_blocks: list[nn.Module] = [
            UNetDoubleConv(2 * base_channels, base_channels, dropout),
        ]
        shape_blocks.extend(
            UNetDoubleConv(base_channels, base_channels, dropout)
            for _ in range(shape_depth - 1)
        )
        self.shape_stream = nn.Sequential(*shape_blocks)
        self.shape_gate = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels, 1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.Sigmoid(),
        )
        self.affinity_head = nn.Conv2d(base_channels, 4, 3, padding=1)
        self.affinity_message = nn.Sequential(
            nn.Conv2d(base_channels, base_channels, 1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels, 1),
        )
        self.propagation_scale = nn.Parameter(torch.tensor(math.atanh(0.10)))
        self.semantic_refiner = nn.Sequential(
            nn.Conv2d(2 * base_channels, base_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(base_channels), base_channels),
            nn.GELU(),
            nn.Conv2d(base_channels, num_classes, 1),
        )
        self.refinement_scale = nn.Parameter(torch.tensor(math.atanh(0.10)))

    def _decode_feature(self, feature: Tensor) -> Tensor:
        skips: list[Tensor] = []
        x = feature
        for level, encoder in enumerate(self.encoders):
            x = encoder(x)
            skips.append(x)
            if level + 1 < len(self.encoders):
                x = self.pool(x)
        for index, (projection, decoder) in enumerate(
            zip(self.up_projections, self.decoders),
        ):
            skip = skips[-2 - index]
            x = F.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = decoder(torch.cat((skip, projection(x)), dim=1))
        return x

    def forward(self, feature: Tensor) -> dict[str, Tensor]:
        semantic = self._decode_feature(feature)
        temporal_detail = self.temporal_detail(feature)
        shape = self.shape_stream(torch.cat((semantic, temporal_detail), dim=1))
        shape = shape * (
            1.0 + self.shape_gate(torch.cat((semantic, shape), dim=1))
        )

        boundary_logits = self.boundary_head(shape)
        affinity_logits = self.affinity_head(shape)
        affinity = _symmetric_affinity_gates(affinity_logits)
        neighbours = _shift4(semantic)
        normalizer = affinity.sum(dim=1, keepdim=True)
        message = sum(
            affinity[:, index : index + 1] * neighbour
            for index, neighbour in enumerate(neighbours)
        ) / normalizer.clamp_min(1.0e-3)
        has_neighbour = (normalizer > 0).to(semantic.dtype)
        semantic = semantic + torch.tanh(self.propagation_scale) * has_neighbour * (
            self.affinity_message(message - semantic)
        )

        base_logits = self.classifier(semantic)
        with torch.no_grad():
            probabilities = torch.softmax(base_logits.float(), dim=1)
            top_two = probabilities.topk(2, dim=1).values
            uncertainty = (1.0 - top_two[:, 0:1] + top_two[:, 1:2]).clamp(0.0, 1.0)
            boundary_gate = boundary_logits.float().sigmoid()
            refinement_gate = torch.maximum(uncertainty, boundary_gate)
        correction = self.semantic_refiner(torch.cat((semantic, shape), dim=1))
        logits = base_logits + torch.tanh(self.refinement_scale) * (
            refinement_gate.to(correction.dtype) * correction
        )
        return {
            "logits": logits,
            "base_logits": base_logits,
            "boundary_logits": boundary_logits,
            "affinity_logits": affinity_logits,
            "decoder_feature": semantic,
            "shape_feature": shape,
            "refinement_gate": refinement_gate.to(logits.dtype),
        }


class SPARXOv2U4(SPARXOOpticalUNetBase):
    """Exact v2 temporal encoder with a boundary-closed-loop spatial decoder."""

    def __init__(
        self,
        *args: Any,
        unet_base_channels: int = 64,
        boundary_shape_depth: int = 2,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, unet_base_channels=unet_base_channels, **kwargs)
        self.decoder = BoundaryClosedLoopUNetDecoder(
            int(kwargs.get("dim", 128)),
            int(kwargs.get("num_classes", 19)),
            int(unet_base_channels),
            float(kwargs.get("dropout", 0.1)),
            int(boundary_shape_depth),
        )

    def forward(self, batch: dict[str, Tensor], return_aux: bool = False) -> dict[str, Any]:
        feature, details = self.encode_s2(batch)
        output: dict[str, Any] = dict(self.decoder(feature))
        if return_aux:
            output.update(details)
        return output
