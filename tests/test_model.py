from __future__ import annotations

import torch

from sparxo import CalendarPhaseSSM, SPARXO, count_parameters


def test_paper_parameter_count() -> None:
    model = SPARXO(num_classes=20)
    assert count_parameters(model) == 8_354_153


def test_forward_shapes_and_finiteness() -> None:
    model = SPARXO(
        num_classes=20,
        dim=32,
        state_dim=16,
        temporal_heads=4,
        unet_base_channels=8,
        dropout=0.0,
        temporal_mask_probability=0.0,
    ).eval()
    s2 = torch.randn(1, 3, 10, 16, 16)
    dates = torch.tensor([[20.0, 46.0, 103.0]])
    valid = torch.tensor([[True, False, True]])
    with torch.inference_mode():
        output = model(s2, dates, valid, return_aux=True)
    assert output["logits"].shape == (1, 20, 16, 16)
    assert output["boundary_logits"].shape == (1, 1, 16, 16)
    assert output["affinity_logits"].shape == (1, 4, 16, 16)
    assert output["temporal_feature"].shape == (1, 32, 16, 16)
    assert torch.isfinite(output["logits"]).all()


def test_elapsed_time_skips_invalid_positions() -> None:
    dates = torch.tensor([[10.0, 40.0, 100.0, 130.0]])
    valid = torch.tensor([[True, False, True, True]])
    delta = CalendarPhaseSSM(dim=8, state_dim=4)._time_steps(dates, valid)
    expected = torch.tensor([[0.0, 0.0, 90.0 / 365.2425, 30.0 / 365.2425]])
    torch.testing.assert_close(delta, expected)


def test_invalid_observation_does_not_change_valid_result() -> None:
    torch.manual_seed(7)
    layer = CalendarPhaseSSM(dim=8, state_dim=4, dropout=0.0).eval()
    first = torch.randn(1, 1, 8)
    invalid = torch.randn(1, 1, 8) * 100.0
    last = torch.randn(1, 1, 8)
    x_with_hole = torch.cat((first, invalid, last), dim=1)
    dates_with_hole = torch.tensor([[10.0, 60.0, 100.0]])
    valid_with_hole = torch.tensor([[True, False, True]])
    x_compact = torch.cat((first, last), dim=1)
    dates_compact = torch.tensor([[10.0, 100.0]])
    valid_compact = torch.tensor([[True, True]])
    with torch.inference_mode():
        with_hole = layer(x_with_hole, dates_with_hole, valid_with_hole)
        compact = layer(x_compact, dates_compact, valid_compact)
    torch.testing.assert_close(with_hole[:, -1], compact[:, -1])
