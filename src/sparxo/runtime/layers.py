"""Checkpoint-compatible SPARX-O temporal primitives."""

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


class LayerNorm2d(nn.Module):
    """Channel-wise LayerNorm for BCHW tensors."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)

    def forward(self, x: Tensor) -> Tensor:
        return self.norm(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.0, dilation: int = 1) -> None:
        super().__init__()
        self.depthwise = nn.Conv2d(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dim,
        )
        self.norm = LayerNorm2d(dim)
        self.pointwise = nn.Sequential(
            nn.Conv2d(dim, 4 * dim, 1),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv2d(4 * dim, dim, 1),
        )
        self.scale = nn.Parameter(torch.full((dim,), 1e-4))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale.view(1, -1, 1, 1) * self.pointwise(self.norm(self.depthwise(x)))


class FrameStem(nn.Module):
    """Small spatial encoder applied independently to each acquisition."""

    def __init__(self, in_channels: int, dim: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, padding=1),
            nn.GroupNorm(8, dim),
            nn.GELU(),
            ConvNeXtBlock(dim, dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        batch, time, channels, height, width = x.shape
        x = self.net(x.reshape(batch * time, channels, height, width))
        return x.reshape(batch, time, -1, height, width)


class ResidualAdapter(nn.Module):
    """Low-rank sensor-specific residual adapter on the channel dimension."""

    def __init__(self, dim: int, dropout: float, ratio: int = 4) -> None:
        super().__init__()
        hidden = max(16, dim // ratio)
        self.norm = nn.LayerNorm(dim)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.scale = nn.Parameter(torch.full((dim,), 1e-3))

    def forward(self, x: Tensor) -> Tensor:
        return x + self.scale * self.net(self.norm(x))


class SensorTemporalAdapter(nn.Module):
    """Sensor-specific input (B-like) and output (C-like) adaptation."""

    def __init__(self, dim: int, dropout: float) -> None:
        super().__init__()
        self.input_adapter = ResidualAdapter(dim, dropout)
        self.output_adapter = ResidualAdapter(dim, dropout)

    def adapt_input(self, frames: Tensor) -> Tensor:
        frames = frames.permute(0, 1, 3, 4, 2)
        return self.input_adapter(frames).permute(0, 1, 4, 2, 3)

    def adapt_output(self, pooled: Tensor) -> Tensor:
        pooled = pooled.permute(0, 2, 3, 1)
        return self.output_adapter(pooled).permute(0, 3, 1, 2)


class CalendarPhaseSSM(nn.Module):
    """Shallow diagonal continuous-time complex SSM.

    The recurrence uses physical date gaps, exponential state propagation and
    trapezoidal integration of the previous/current input. Complex transition
    pairs implement calendar phase. It is intentionally compact and auditable.
    """

    def __init__(
        self,
        dim: int,
        state_dim: int,
        dropout: float = 0.1,
        complex_state: bool = True,
        physical_time: bool = True,
        trapezoidal: bool = True,
        selective_decay_physical_phase: bool = False,
        zero_init_time_gate: bool = False,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim > dim:
            raise ValueError("state_dim must be positive and no larger than dim")
        self.dim = dim
        self.state_dim = state_dim
        self.complex_state = complex_state
        self.physical_time = physical_time
        self.trapezoidal = trapezoidal
        self.selective_decay_physical_phase = selective_decay_physical_phase

        # Project through a narrow state bottleneck; this makes state_dim a real
        # capacity/efficiency control rather than an unused label.
        self.to_state = nn.Linear(dim, state_dim)
        self.from_state = nn.Linear(state_dim, dim)
        self.input_gate = nn.Linear(dim, state_dim)
        self.time_gate = nn.Linear(dim, 1)
        if zero_init_time_gate:
            # Start from the well-conditioned physical-time recurrence.
            # The gate then learns a smooth, content-selective decay update.
            nn.init.zeros_(self.time_gate.weight)
            nn.init.zeros_(self.time_gate.bias)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

        # Positive decay rate per year. Initial memory half-lives cover roughly
        # one month to two years.
        half_lives = torch.logspace(
            math.log10(1.0 / 12.0),
            math.log10(2.0),
            state_dim,
        )
        decay_rate = math.log(2.0) / half_lives
        self.log_decay = nn.Parameter(torch.log(torch.expm1(decay_rate)))

        # Cycles/year: annual, semiannual, quarterly and slow trend-like phase.
        base_frequencies = torch.tensor([1.0, 2.0, 4.0, 0.5])
        frequencies = base_frequencies.repeat(math.ceil(state_dim / 4))[:state_dim]
        self.log_frequency = nn.Parameter(torch.log(frequencies))
        self.input_scale = nn.Parameter(torch.ones(state_dim))
        self.output_scale = nn.Parameter(torch.ones(state_dim))

    def _time_steps(self, dates: Tensor, valid: Tensor) -> Tensor:
        # The sequence can contain holes introduced by model-side token
        # masking. Measure every transition from the previous *valid*
        # observation, because invalid positions do not update the SSM state.
        delta = torch.zeros_like(dates)
        previous_valid_date = torch.zeros(
            dates.shape[0],
            dtype=dates.dtype,
            device=dates.device,
        )
        seen_valid = torch.zeros(
            dates.shape[0],
            dtype=torch.bool,
            device=dates.device,
        )
        if not self.physical_time:
            lengths = valid.sum(dim=1).clamp_min(2).to(dates.dtype)
            uniform = 1.0 / (lengths - 1.0)
        for step in range(dates.shape[1]):
            current_valid = valid[:, step]
            if self.physical_time:
                elapsed = (
                    (dates[:, step] - previous_valid_date).clamp_min(0.0)
                    / 365.2425
                )
            else:
                elapsed = uniform
            delta[:, step] = torch.where(
                current_valid & seen_valid,
                elapsed,
                torch.zeros_like(elapsed),
            )
            previous_valid_date = torch.where(
                current_valid,
                dates[:, step],
                previous_valid_date,
            )
            seen_valid = seen_valid | current_valid
        return delta

    def forward(self, x: Tensor, dates: Tensor, valid: Tensor) -> Tensor:
        """Encode ``x`` shaped ``[N,T,D]`` with masks shaped ``[N,T]``."""
        if x.ndim != 3 or dates.shape != valid.shape or x.shape[:2] != dates.shape:
            raise ValueError("CalendarPhaseSSM received inconsistent tensor shapes")
        delta = self._time_steps(dates, valid)
        decay_rate = F.softplus(self.log_decay).to(x.dtype)
        frequency = self.log_frequency.exp().to(x.dtype)
        real = x.new_zeros((x.shape[0], self.state_dim))
        imag = x.new_zeros((x.shape[0], self.state_dim))
        previous_input = x.new_zeros((x.shape[0], self.state_dim))
        outputs = []

        for step in range(x.shape[1]):
            x_t = x[:, step]
            mask_t = valid[:, step].unsqueeze(-1)
            correction = 0.75 + 0.5 * torch.sigmoid(self.time_gate(x_t))
            raw_dt = delta[:, step].unsqueeze(-1)
            if self.selective_decay_physical_phase:
                # In selective-decay mode, timestamps determine integration and phase exactly;
                # content selection changes only the real (memory) dynamics.
                # This keeps one coherent calendar-frequency basis while still
                # allowing informative/noisy observations to hold/forget at
                # different rates.
                dt = raw_dt.clamp(0.0, 1.5)
                decay = torch.exp(-dt * decay_rate * correction)
            else:
                # In the baseline mode, the learned correction rescales
                # the complete discretization step, including phase.
                dt = (raw_dt * correction).clamp(0.0, 1.5)
                decay = torch.exp(-dt * decay_rate)
            if self.complex_state:
                angle = 2.0 * math.pi * dt * frequency
                cosine, sine = torch.cos(angle), torch.sin(angle)
            else:
                cosine, sine = torch.ones_like(decay), torch.zeros_like(decay)

            propagated_real = decay * (real * cosine - imag * sine)
            propagated_imag = decay * (real * sine + imag * cosine)
            current_input = (
                self.to_state(x_t)
                * torch.sigmoid(self.input_gate(x_t))
                * self.input_scale
            )
            if self.trapezoidal:
                previous_real = decay * cosine * previous_input
                previous_imag = decay * sine * previous_input
                input_real = 0.5 * dt * (previous_real + current_input)
                input_imag = 0.5 * dt * previous_imag
            else:
                input_real = dt * current_input
                input_imag = torch.zeros_like(input_real)

            next_real = propagated_real + input_real
            next_imag = propagated_imag + input_imag
            real = torch.where(mask_t, next_real, real)
            imag = torch.where(mask_t, next_imag, imag)
            previous_input = torch.where(mask_t, current_input, previous_input)
            state_output = self.from_state(real * self.output_scale)
            outputs.append(torch.where(mask_t, x_t + state_output, torch.zeros_like(x_t)))

        y = torch.stack(outputs, dim=1)
        return y + self.ffn(self.norm(y))


class TemporalPool(nn.Module):
    def __init__(self, dim: int, heads: int, dropout: float, adaptive: bool = True) -> None:
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by temporal_heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.adaptive = adaptive
        self.score = nn.Linear(dim, heads)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.Dropout(dropout))

    def forward(self, x: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        values = x.reshape(x.shape[0], x.shape[1], self.heads, self.head_dim)
        if self.adaptive:
            logits = self.score(x).masked_fill(~valid.unsqueeze(-1), -torch.inf)
            weights = torch.softmax(logits, dim=1)
        else:
            weights = valid.to(x.dtype)
            weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1.0)
            weights = weights.unsqueeze(-1).expand(-1, -1, self.heads)
        pooled = (values * weights.unsqueeze(-1)).sum(dim=1).reshape(x.shape[0], -1)
        return self.proj(pooled), weights


class CalendarTemporalEncoder(nn.Module):
    def __init__(
        self,
        dim: int,
        state_dim: int,
        heads: int,
        layers: int,
        dropout: float,
        complex_state: bool,
        physical_time: bool,
        trapezoidal: bool,
        adaptive_pool: bool,
        calendar_features: int = 4,
    ) -> None:
        super().__init__()
        self.date_encoding = CalendarEncoding(dim, calendar_features)
        self.layers = nn.ModuleList(
            [
                CalendarPhaseSSM(
                    dim=dim,
                    state_dim=state_dim,
                    dropout=dropout,
                    complex_state=complex_state,
                    physical_time=physical_time,
                    trapezoidal=trapezoidal,
                )
                for _ in range(layers)
            ],
        )
        self.pool = TemporalPool(dim, heads, dropout, adaptive_pool)

    def forward(
        self,
        frames: Tensor,
        dates: Tensor,
        valid: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch, time, dim, height, width = frames.shape
        sequence = frames.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, dim)
        pixel_dates = dates.repeat_interleave(height * width, dim=0)
        pixel_valid = valid.repeat_interleave(height * width, dim=0)
        # Absolute DOY anchors crop phenology; the recurrence still uses true
        # inter-acquisition gaps for continuous-time propagation.
        sequence = (sequence + self.date_encoding(pixel_dates)) * pixel_valid.unsqueeze(-1)
        for layer in self.layers:
            sequence = layer(sequence, pixel_dates, pixel_valid)
        pooled, weights = self.pool(sequence, pixel_valid)
        pooled = pooled.reshape(batch, height, width, dim).permute(0, 3, 1, 2)
        weights = weights.reshape(batch, height, width, time, -1)
        return pooled, weights


class CalendarEncoding(nn.Module):
    """Annual and semiannual Fourier features from absolute acquisition dates."""

    def __init__(self, dim: int, features: int = 4) -> None:
        super().__init__()
        if features not in (2, 4):
            raise ValueError("calendar_features must be 2 or 4")
        self.features = features
        self.projection = nn.Linear(features, dim)

    def forward(self, dates: Tensor) -> Tensor:
        annual = 2.0 * math.pi * dates / 365.2425
        values = (annual.sin(), annual.cos())
        if self.features == 4:
            values += ((2.0 * annual).sin(), (2.0 * annual).cos())
        features = torch.stack(values, dim=-1)
        return self.projection(features)


class TemporalConvBlock(nn.Module):
    def __init__(self, dim: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.depthwise = nn.Conv1d(
            dim,
            dim,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
            groups=dim,
        )
        self.pointwise = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor, valid: Tensor) -> Tensor:
        residual = self.norm(x).transpose(1, 2)
        residual = self.depthwise(residual).transpose(1, 2)
        x = x + self.pointwise(residual)
        return x * valid.unsqueeze(-1)


class TCNTemporalEncoder(nn.Module):
    """Strong short-sequence convolutional control."""

    def __init__(self, dim: int, heads: int, layers: int, dropout: float, adaptive_pool: bool) -> None:
        super().__init__()
        self.date_encoding = CalendarEncoding(dim)
        self.layers = nn.ModuleList(
            [TemporalConvBlock(dim, 2 ** (index % 4), dropout) for index in range(max(1, layers))],
        )
        self.pool = TemporalPool(dim, heads, dropout, adaptive_pool)

    def forward(self, frames: Tensor, dates: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, dim, height, width = frames.shape
        sequence = frames.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, dim)
        pixel_dates = dates.repeat_interleave(height * width, dim=0)
        pixel_valid = valid.repeat_interleave(height * width, dim=0)
        sequence = (sequence + self.date_encoding(pixel_dates)) * pixel_valid.unsqueeze(-1)
        for layer in self.layers:
            sequence = layer(sequence, pixel_valid)
        pooled, weights = self.pool(sequence, pixel_valid)
        pooled = pooled.reshape(batch, height, width, dim).permute(0, 3, 1, 2)
        return pooled, weights.reshape(batch, height, width, time, -1)


class TransformerTemporalEncoder(nn.Module):
    """Same-budget attention control with explicit calendar encoding."""

    def __init__(self, dim: int, heads: int, layers: int, dropout: float, adaptive_pool: bool) -> None:
        super().__init__()
        self.date_encoding = CalendarEncoding(dim)
        layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=heads,
            dim_feedforward=4 * dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=max(1, layers), enable_nested_tensor=False)
        self.pool = TemporalPool(dim, heads, dropout, adaptive_pool)

    def forward(self, frames: Tensor, dates: Tensor, valid: Tensor) -> tuple[Tensor, Tensor]:
        batch, time, dim, height, width = frames.shape
        sequence = frames.permute(0, 3, 4, 1, 2).reshape(batch * height * width, time, dim)
        pixel_dates = dates.repeat_interleave(height * width, dim=0)
        pixel_valid = valid.repeat_interleave(height * width, dim=0)
        sequence = sequence + self.date_encoding(pixel_dates)
        sequence = self.encoder(sequence, src_key_padding_mask=~pixel_valid)
        sequence = sequence * pixel_valid.unsqueeze(-1)
        pooled, weights = self.pool(sequence, pixel_valid)
        pooled = pooled.reshape(batch, height, width, dim).permute(0, 3, 1, 2)
        return pooled, weights.reshape(batch, height, width, time, -1)


def build_temporal_encoder(
    backend: str,
    dim: int,
    state_dim: int,
    heads: int,
    layers: int,
    dropout: float,
    complex_state: bool,
    physical_time: bool,
    trapezoidal: bool,
    adaptive_pool: bool,
    calendar_features: int = 4,
) -> nn.Module:
    if backend == "calendar_ssm":
        return CalendarTemporalEncoder(
            dim=dim,
            state_dim=state_dim,
            heads=heads,
            layers=layers,
            dropout=dropout,
            complex_state=complex_state,
            physical_time=physical_time,
            trapezoidal=trapezoidal,
            adaptive_pool=adaptive_pool,
            calendar_features=calendar_features,
        )
    if backend == "tcn":
        return TCNTemporalEncoder(dim, heads, layers, dropout, adaptive_pool)
    if backend == "transformer":
        return TransformerTemporalEncoder(dim, heads, layers, dropout, adaptive_pool)
    raise ValueError(f"Unknown temporal_backend: {backend}")


class SemanticDecoder(nn.Module):
    def __init__(self, dim: int, num_classes: int, depth: int, dropout: float) -> None:
        super().__init__()
        # Cover field interiors and neighbouring parcels on the 10 m grid.
        # The former [1,2,3,1] cycle underused crops larger than 32 pixels.
        dilations = [1, 2, 4, 8, 4, 2]
        self.context = nn.Sequential(
            *[
                ConvNeXtBlock(dim, dropout, dilation=dilations[index % len(dilations)])
                for index in range(depth)
            ],
        )
        self.classifier_weight = nn.Parameter(torch.empty(num_classes, dim))
        self.classifier_bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.classifier_weight, a=math.sqrt(5))
        self.boundary_features = nn.Sequential(
            nn.Conv2d(dim, dim // 2, 3, padding=1),
            nn.GELU(),
        )
        self.boundary_weight = nn.Parameter(torch.empty(dim // 2))
        self.boundary_bias = nn.Parameter(torch.zeros(()))
        nn.init.normal_(self.boundary_weight, std=math.sqrt(2.0 / dim))

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.context(x)
        boundary_features = self.boundary_features(x)
        boundary = torch.einsum("bchw,c->bhw", boundary_features, self.boundary_weight)
        logits = torch.einsum("bchw,kc->bkhw", x, self.classifier_weight)
        logits = logits + self.classifier_bias.view(1, -1, 1, 1)
        return logits, boundary.unsqueeze(1) + self.boundary_bias


class PixelClassifier(nn.Module):
    """Per-pixel linear classifier without singleton Conv2d gradient strides."""

    def __init__(self, dim: int, num_classes: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, dim))
        self.bias = nn.Parameter(torch.zeros(num_classes))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x: Tensor) -> Tensor:
        return (
            torch.einsum("bchw,kc->bkhw", x, self.weight)
            + self.bias.view(1, -1, 1, 1)
        )




def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
