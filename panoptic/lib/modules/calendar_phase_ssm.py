"""Calendar-Phase SSM temporal encoder used by SPARXO + PaPs.

The wrapper deliberately keeps Exchanger's two ``LinearLayer`` spectral
embeddings and returns a per-timestep sequence.  Temporal aggregation remains
the caller's original masked mean, so the only substantive replacement is the
two CUD/GP blocks with Calendar-Phase SSM blocks.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from layers import LinearLayer


class CalendarEncoding(nn.Module):
    """Annual and semiannual Fourier features from physical acquisition dates."""

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(4, dim)

    def forward(self, dates: Tensor) -> Tensor:
        annual = 2.0 * math.pi * dates / 365.2425
        features = torch.stack(
            (
                annual.sin(),
                annual.cos(),
                (2.0 * annual).sin(),
                (2.0 * annual).cos(),
            ),
            dim=-1,
        )
        return self.projection(features)


class CalendarPhaseSSM(nn.Module):
    """Diagonal continuous-time complex SSM used by the SPARXO v2 core."""

    def __init__(
        self,
        dim: int,
        state_dim: int,
        dropout: float = 0.1,
        complex_state: bool = True,
        physical_time: bool = True,
        trapezoidal: bool = True,
    ) -> None:
        super().__init__()
        if state_dim <= 0 or state_dim > dim:
            raise ValueError("state_dim must be positive and no larger than dim")
        self.dim = dim
        self.state_dim = state_dim
        self.complex_state = complex_state
        self.physical_time = physical_time
        self.trapezoidal = trapezoidal

        self.to_state = nn.Linear(dim, state_dim)
        self.from_state = nn.Linear(state_dim, dim)
        self.input_gate = nn.Linear(dim, state_dim)
        self.time_gate = nn.Linear(dim, 1)
        self.norm = nn.LayerNorm(dim)
        self.ffn = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * dim, dim),
            nn.Dropout(dropout),
        )

        half_lives = torch.logspace(
            math.log10(1.0 / 12.0),
            math.log10(2.0),
            state_dim,
        )
        decay_rate = math.log(2.0) / half_lives
        self.log_decay = nn.Parameter(torch.log(torch.expm1(decay_rate)))

        base_frequencies = torch.tensor([1.0, 2.0, 4.0, 0.5])
        frequencies = base_frequencies.repeat(math.ceil(state_dim / 4))[:state_dim]
        self.log_frequency = nn.Parameter(torch.log(frequencies))
        self.input_scale = nn.Parameter(torch.ones(state_dim))
        self.output_scale = nn.Parameter(torch.ones(state_dim))

    def _time_steps(self, dates: Tensor, valid: Tensor) -> Tensor:
        if not dates.is_floating_point():
            raise TypeError("CalendarPhaseSSM dates must be floating point")
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
            dt = (delta[:, step].unsqueeze(-1) * correction).clamp(0.0, 1.5)
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
            outputs.append(
                torch.where(mask_t, x_t + state_output, torch.zeros_like(x_t)),
            )

        y = torch.stack(outputs, dim=1)
        return y + self.ffn(self.norm(y))


class CalendarPhaseTemporalEncoder(nn.Module):
    """Two-stage Exchanger-shaped encoder with CPS in place of each GP block."""

    def __init__(self, config, **kwargs) -> None:
        super().__init__()
        spec = {key.lower(): value for key, value in dict(config.CALENDAR_SSM).items()}
        in_dim = int(kwargs["in_channels"])
        dim = int(spec["embed_dim"])
        state_dim = int(spec["state_dim"])
        num_layers = int(spec["num_layers"])
        if num_layers <= 0:
            raise ValueError("CALENDAR_SSM.NUM_LAYERS must be positive")

        self.mlp_layers = nn.ModuleList(
            [
                LinearLayer(
                    in_dim=in_dim if index == 0 else dim,
                    out_dim=dim,
                    norm_type=spec["mlp_norm"],
                    act_type=spec["mlp_act"],
                    bias=False,
                )
                for index in range(num_layers)
            ],
        )
        self.calendar_encoding = CalendarEncoding(dim)
        self.ssm_layers = nn.ModuleList(
            [
                CalendarPhaseSSM(
                    dim=dim,
                    state_dim=state_dim,
                    dropout=float(spec["dropout"]),
                    complex_state=bool(spec["complex_state"]),
                    physical_time=bool(spec["physical_time"]),
                    trapezoidal=bool(spec["trapezoidal"]),
                )
                for _ in range(num_layers)
            ],
        )
        self.num_stages = num_layers
        self.out_dim = dim

    def forward(
        self,
        x: Tensor,
        date_positions: Tensor,
        temp_mask: Optional[Tensor] = None,
        return_attn: bool = False,
        **kwargs,
    ):
        if x.ndim != 3:
            raise ValueError(f"expected [B,T,C], received {tuple(x.shape)}")
        if date_positions.shape != x.shape[:2]:
            raise ValueError("date positions must have shape [B,T]")
        invalid = (
            torch.zeros_like(date_positions, dtype=torch.bool)
            if temp_mask is None
            else temp_mask.bool()
        )
        valid = ~invalid
        # Exchanger stores date positions as integer days.  Convert before the
        # continuous-time recurrence so sub-year gaps are never truncated.
        dates = date_positions.to(dtype=x.dtype)

        latent_features = []
        for index, (embedding, ssm) in enumerate(zip(self.mlp_layers, self.ssm_layers)):
            x = embedding(x)
            if index == 0:
                x = x + self.calendar_encoding(dates)
            x = x * valid.unsqueeze(-1)
            x = ssm(x, dates, valid)
            x = x * valid.unsqueeze(-1)
            if return_attn:
                latent_features.append(x)

        latent = torch.stack(latent_features, dim=1) if return_attn else None
        return x, None, latent
