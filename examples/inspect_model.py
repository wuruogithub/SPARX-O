"""Construct the paper model and run a synthetic inspection pass."""

from __future__ import annotations

import torch

from sparxo import SPARXO, count_parameters


def main() -> None:
    model = SPARXO(num_classes=20).eval()
    print(f"Trainable parameters: {count_parameters(model):,}")

    s2 = torch.randn(1, 4, 10, 32, 32)
    dates = torch.tensor([[18.0, 46.0, 103.0, 166.0]])
    valid = torch.tensor([[True, True, False, True]])
    with torch.inference_mode():
        output = model(s2, dates, valid, return_aux=True)

    for name in ("logits", "boundary_logits", "affinity_logits", "temporal_feature"):
        print(f"{name:>18}: {tuple(output[name].shape)}")


if __name__ == "__main__":
    main()
