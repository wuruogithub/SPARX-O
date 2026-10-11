"""SPARXO-v2 temporal encoder with the official Exchanger U-Net + PaPs head."""

from typing import Dict, List, Tuple

import torch
from torch import Tensor, nn

from modules import MaskFormer, PaPs, UNet
from sparxo.runtime.layers import FrameStem, SensorTemporalAdapter, build_temporal_encoder

from .pano_seg import PanopticSegmentor


class SPARXOOpticalTemporalEncoder(nn.Module):
    """Exact SPARXO-v2 S2 encoder adapted to Exchanger's dense batch format."""

    def __init__(self, config) -> None:
        super().__init__()
        spec = config.CALENDAR_SSM
        in_channels = int(config.DATASET.INPUT_DIM[0])
        dim = int(spec.EMBED_DIM)
        dropout = float(spec.DROPOUT)
        self.s2_stem = FrameStem(in_channels, dim, dropout)
        self.temporal_encoder = build_temporal_encoder(
            backend='calendar_ssm',
            dim=dim,
            state_dim=int(spec.STATE_DIM),
            heads=int(spec.NUM_HEADS),
            layers=int(spec.NUM_LAYERS),
            dropout=dropout,
            complex_state=bool(spec.COMPLEX_STATE),
            physical_time=bool(spec.PHYSICAL_TIME),
            trapezoidal=bool(spec.TRAPEZOIDAL),
            adaptive_pool=bool(spec.ADAPTIVE_POOL),
        )
        self.adapter = SensorTemporalAdapter(dim, dropout)
        self.out_dim = dim
        self.date_offset = float(spec.DATE_OFFSET)

    def forward(
        self,
        data: Tensor,
        date_positions: Tensor,
        temporal_mask: Tensor,
    ) -> Tensor:
        # Official dense batches are [B,C,T,H,W]; SPARXO uses [B,T,C,H,W].
        frames = data.permute(0, 2, 1, 3, 4).contiguous()
        frames = self.adapter.adapt_input(self.s2_stem(frames))
        valid = ~temporal_mask.bool()
        if not torch.all(valid.any(dim=1)):
            raise RuntimeError('Every sample must retain at least one S2 acquisition')
        dates = date_positions.to(dtype=frames.dtype) + self.date_offset
        pooled, _ = self.temporal_encoder(frames, dates, valid)
        return self.adapter.adapt_output(pooled)


class SPARXOPanopticSegmentor(PanopticSegmentor):
    """Official U-Net/PaPs pipeline with only its temporal encoder replaced."""

    def __init__(self, config, **kwargs) -> None:
        # Build the official model first so parsing, losses and evaluation stay
        # byte-for-byte identical, then replace its temporal and spatial stack.
        super().__init__(config, **kwargs)
        self.temp_pos_encode = nn.Identity()
        self.temp_encoder = SPARXOOpticalTemporalEncoder(config)
        self.space_encoder = UNet(config, in_channels=self.temp_encoder.out_dim)
        space_enc_out_dims = self.space_encoder.out_dims
        self.paps_head = PaPs(
            config,
            enc_dim=space_enc_out_dims[0],
            stack_dim=sum(space_enc_out_dims),
        )

    def forward(
        self,
        inputs: Dict[str, Tensor],
        output_aux: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        del output_aux, kwargs
        data, _, date_pos, temporal_mask, labels = self.parse_inputs(
            inputs,
            self.modality,
        )
        x = self.temp_encoder(data, date_pos, temporal_mask)
        space_outs: List[Tensor] = self.space_encoder(x)
        high_res_level = space_outs[-1]
        feat_maps = space_outs[::-1]
        zones = labels[:, 2, ...] if self.training else None
        outputs = self.paps_head(high_res_level, feat_maps, zones=zones)
        outputs['labels'] = labels
        losses = self.losses(outputs, labels) if self.mode != 'test' else {}
        return outputs, losses


class SPARXOMask2FormerPanopticSegmentor(PanopticSegmentor):
    """SPARXO-v2.U.4 temporal features with official Mask2Former spatial heads.

    The transferred checkpoint supplies only FrameStem, Calendar-SSM and the
    temporal adapter.  PVTv2, FPN and the masked query decoder remain exactly
    the official Exchanger implementation and are initialized from scratch.
    """

    def __init__(self, config, **kwargs) -> None:
        super().__init__(config, **kwargs)
        self.temp_pos_encode = nn.Identity()
        self.temp_encoder = SPARXOOpticalTemporalEncoder(config)
        if MaskFormer is None:
            raise ImportError('MaskFormer dependencies could not be imported')
        self.space_encoder = MaskFormer(
            config,
            in_channels=self.temp_encoder.out_dim,
        )
        self.space_encoder_type = 'maskformer'

    def forward(
        self,
        inputs: Dict[str, Tensor],
        output_aux: bool = False,
        **kwargs,
    ) -> Tuple[Dict[str, Tensor], Dict[str, Tensor]]:
        del output_aux, kwargs
        data, _, date_pos, temporal_mask, labels = self.parse_inputs(
            inputs,
            self.modality,
        )
        height, width = data.shape[-2:]
        temporal_feature = self.temp_encoder(data, date_pos, temporal_mask)
        outputs = self.space_encoder(temporal_feature, (height, width))
        outputs['labels'] = labels
        losses = self.losses(outputs, labels) if self.mode != 'test' else {}
        return outputs, losses
