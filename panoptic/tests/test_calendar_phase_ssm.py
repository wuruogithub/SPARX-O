from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[1]
    / "lib"
    / "modules"
    / "calendar_phase_ssm.py"
)
LIB_ROOT = MODULE_PATH.parents[1]
if str(LIB_ROOT) not in sys.path:
    sys.path.insert(0, str(LIB_ROOT))
SPEC = importlib.util.spec_from_file_location("calendar_phase_ssm_under_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
CalendarPhaseTemporalEncoder = MODULE.CalendarPhaseTemporalEncoder


def _config():
    return SimpleNamespace(
        CALENDAR_SSM={
            "EMBED_DIM": 128,
            "STATE_DIM": 64,
            "NUM_LAYERS": 2,
            "DROPOUT": 0.1,
            "COMPLEX_STATE": True,
            "PHYSICAL_TIME": True,
            "TRAPEZOIDAL": True,
            "MLP_NORM": "batchnorm",
            "MLP_ACT": "gelu",
        },
    )


def _masked_mean(x: torch.Tensor, invalid: torch.Tensor) -> torch.Tensor:
    valid = ~invalid
    return (x * valid.unsqueeze(-1)).sum(dim=1) / valid.sum(
        dim=1,
        keepdim=True,
    ).clamp_min(1)


def test_calendar_encoder_preserves_exchanger_sequence_contract() -> None:
    torch.manual_seed(42)
    model = CalendarPhaseTemporalEncoder(_config(), in_channels=10).eval()
    inputs = torch.randn(6, 7, 10)
    integer_dates = torch.tensor([[1, 18, 37, 64, 91, 130, 177]]).repeat(6, 1)
    invalid = torch.zeros(6, 7, dtype=torch.bool)
    invalid[0, -2:] = True

    with torch.no_grad():
        outputs, attention, latent = model(
            inputs,
            integer_dates,
            invalid,
            return_attn=True,
        )

    assert outputs.shape == (6, 7, 128)
    assert attention is None
    assert latent.shape == (6, 2, 7, 128)
    assert torch.equal(outputs[0, -2:], torch.zeros_like(outputs[0, -2:]))
    pooled = _masked_mean(outputs, invalid)
    assert pooled.shape == (6, 128)
    assert torch.isfinite(pooled).all()


def test_physical_dates_change_calendar_ssm_output() -> None:
    torch.manual_seed(7)
    model = CalendarPhaseTemporalEncoder(_config(), in_channels=10).eval()
    inputs = torch.randn(2, 6, 10)
    invalid = torch.zeros(2, 6, dtype=torch.bool)
    regular_dates = torch.tensor([[1, 31, 61, 91, 121, 151]]).repeat(2, 1)
    stretched_dates = torch.tensor([[1, 61, 121, 181, 241, 301]]).repeat(2, 1)

    with torch.no_grad():
        regular = model(inputs, regular_dates, invalid)[0]
        stretched = model(inputs, stretched_dates, invalid)[0]

    assert not torch.allclose(regular, stretched)
