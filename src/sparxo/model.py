"""Public tensor API backed by the checkpoint-compatible complete main model."""
from __future__ import annotations
from torch import Tensor, nn
from .runtime.layers import CalendarPhaseSSM, count_parameters
from .runtime.sparxo_v2_u4 import SPARXOv2U4

class SPARXO(SPARXOv2U4):
    """SPARX-O semantic model; dates are elapsed days since 2018-01-01.

    Inputs: S2 [B,T,C,H,W], dates [B,T], valid [B,T] (boolean).
    A training batch dictionary with s2/s2_dates/s2_valid is also accepted.
    """
    def __init__(self, num_classes: int = 20, input_channels: int = 10,
                 dim: int = 160, state_dim: int = 64, temporal_heads: int = 8,
                 temporal_layers: int = 2, unet_base_channels: int = 64,
                 boundary_shape_depth: int = 2, dropout: float = 0.1,
                 temporal_mask_probability: float = 0.1, calendar_features: int = 4,
                 **kwargs) -> None:
        super().__init__(num_classes=num_classes, input_channels=input_channels,
                         dim=dim, state_dim=state_dim, temporal_heads=temporal_heads,
                         temporal_layers=temporal_layers, unet_base_channels=unet_base_channels,
                         boundary_shape_depth=boundary_shape_depth, dropout=dropout,
                         temporal_mask_probability=temporal_mask_probability,
                         calendar_features=calendar_features, **kwargs)

    def encode_temporal(self, s2: Tensor, dates: Tensor, valid: Tensor):
        if s2.ndim != 5 or s2.shape[:2] != dates.shape or dates.shape != valid.shape:
            raise ValueError("expected S2 [B,T,C,H,W] and dates/valid [B,T]")
        if str(valid.dtype) != "torch.bool":
            raise TypeError("valid must be boolean")
        if not valid.any(dim=1).all():
            raise ValueError("each sample must contain a valid acquisition")
        feature, details = self.encode_s2({"s2": s2, "s2_dates": dates, "s2_valid": valid})
        details["attention"] = details["s2_attention"]
        details["encoding_valid"] = details["s2_encoding_valid"]
        return feature, details

    def forward(self, s2, dates=None, valid=None, return_aux=False):
        if isinstance(s2, dict):
            return super().forward(s2, return_aux=return_aux)
        feature, details = self.encode_temporal(s2, dates, valid)
        output = self.decoder(feature)
        if return_aux:
            output.update(details)
            output["temporal_feature"] = feature
        return output
