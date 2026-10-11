"""Shared optical backbone used by the SPARXO/SPARX v2.U.4 models."""

from __future__ import annotations

from typing import Any

import torch
from torch import Tensor, nn

from .layers import (
    FrameStem,
    SemanticDecoder,
    SensorTemporalAdapter,
    build_temporal_encoder,
)

def _group_count(channels: int) -> int:
    """Largest practical GroupNorm group count for small DDP microbatches."""

    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


class UNetDoubleConv(nn.Module):
    """The standard two-convolution U-Net unit with batch-size-safe norms."""

    def __init__(self, in_channels: int, out_channels: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
            nn.Dropout2d(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(out_channels), out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SPARXOV2UNetDecoder(nn.Module):
    """Four-level U-Net operating only after v2 temporal aggregation."""

    def __init__(
        self,
        in_channels: int,
        num_classes: int,
        base_channels: int,
        dropout: float,
    ) -> None:
        super().__init__()
        widths = tuple(base_channels * (2**level) for level in range(4))
        self.encoders = nn.ModuleList()
        current = in_channels
        for width in widths:
            self.encoders.append(UNetDoubleConv(current, width, dropout))
            current = width
        self.pool = nn.MaxPool2d(2)
        self.up_projections = nn.ModuleList(
            [
                nn.Conv2d(widths[level], widths[level - 1], 1)
                for level in range(3, 0, -1)
            ],
        )
        self.decoders = nn.ModuleList(
            [
                UNetDoubleConv(2 * widths[level - 1], widths[level - 1], dropout)
                for level in range(3, 0, -1)
            ],
        )
        self.classifier = nn.Conv2d(widths[0], num_classes, 1)
        self.boundary_head = nn.Sequential(
            nn.Conv2d(widths[0], widths[0], 3, padding=1, bias=False),
            nn.GroupNorm(_group_count(widths[0]), widths[0]),
            nn.GELU(),
            nn.Conv2d(widths[0], 1, 1),
        )

    def forward(self, feature: Tensor) -> tuple[Tensor, Tensor]:
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
            x = torch.nn.functional.interpolate(
                x,
                size=skip.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            x = decoder(torch.cat((skip, projection(x)), dim=1))
        return self.classifier(x), self.boundary_head(x)


class SPARXOTemporalBase(nn.Module):
    """The SPARXO temporal backbone made into a standalone model."""

    def __init__(
        self,
        num_classes: int = 19,
        input_channels: int = 10,
        dim: int = 128,
        state_dim: int = 64,
        temporal_heads: int = 4,
        temporal_layers: int = 2,
        decoder_depth: int = 6,
        dropout: float = 0.1,
        complex_state: bool = True,
        physical_time: bool = True,
        trapezoidal: bool = True,
        adaptive_pool: bool = True,
        temporal_mask_probability: float = 0.1,
        calendar_features: int = 4,
    ) -> None:
        super().__init__()
        if not 0.0 <= temporal_mask_probability < 1.0:
            raise ValueError("temporal_mask_probability must be in [0,1)")
        self.temporal_mask_probability = temporal_mask_probability
        if input_channels <= 0:
            raise ValueError("input_channels must be positive")
        self.s2_stem = FrameStem(input_channels, dim, dropout)
        self.temporal_encoder = build_temporal_encoder(
            backend="calendar_ssm",
            dim=dim,
            state_dim=state_dim,
            heads=temporal_heads,
            layers=temporal_layers,
            dropout=dropout,
            complex_state=complex_state,
            physical_time=physical_time,
            trapezoidal=trapezoidal,
            adaptive_pool=adaptive_pool,
            calendar_features=calendar_features,
        )
        self.adapter = SensorTemporalAdapter(dim, dropout)
        self.decoder = SemanticDecoder(dim, num_classes, decoder_depth, dropout)

    def _mask_validity(self, valid: Tensor) -> Tensor:
        if not self.training or self.temporal_mask_probability == 0:
            return valid
        keep = valid & (
            torch.rand(valid.shape, device=valid.device) >= self.temporal_mask_probability
        )
        empty = ~keep.any(dim=1)
        if empty.any():
            first_valid = valid.to(torch.int64).argmax(dim=1)
            keep = keep.clone()
            keep[empty, first_valid[empty]] = True
        return keep

    def encode_s2(self, batch: dict[str, Tensor]) -> tuple[Tensor, dict[str, Tensor]]:
        """Expose the exact pre-decoder v2 representation for protected fusion."""

        frames = self.adapter.adapt_input(self.s2_stem(batch["s2"]))
        valid = self._mask_validity(batch["s2_valid"])
        feature, attention = self.temporal_encoder(frames, batch["s2_dates"], valid)
        feature = self.adapter.adapt_output(feature)
        return feature, {
            "s2_attention": attention,
            "s2_encoding_valid": valid,
        }

    def forward(self, batch: dict[str, Tensor], return_aux: bool = False) -> dict[str, Any]:
        feature, details = self.encode_s2(batch)
        logits, boundary_logits = self.decoder(feature)
        output: dict[str, Any] = {
            "logits": logits,
            "boundary_logits": boundary_logits,
        }
        if return_aux:
            output.update(details)
        return output


class SPARXOOpticalUNetBase(SPARXOTemporalBase):
    """Calendar-SSM optical backbone followed by the established U-Net trunk."""

    def __init__(
        self,
        *args: Any,
        unet_base_channels: int = 64,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        # Preserve every pre-decoder v2 parameter and its state-dict name. Only
        # the weak same-resolution dilated decoder is replaced.
        dim = int(kwargs.get("dim", 128))
        num_classes = int(kwargs.get("num_classes", 19))
        dropout = float(kwargs.get("dropout", 0.1))
        self.decoder = SPARXOV2UNetDecoder(
            dim,
            num_classes,
            int(unet_base_channels),
            dropout,
        )
