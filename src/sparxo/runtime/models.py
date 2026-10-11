"""Optical-only model factory; no multimodal SPARX models are included."""
from typing import Any
from torch import nn
from .layers import count_parameters
from .sparxo_v2_u4 import SPARXOv2U4

def build_model(config: dict[str, Any]) -> nn.Module:
    architecture = str(config.get("architecture", "sparxo"))
    if architecture not in {"sparxo", "sparxo_v2_u4"}:
        raise ValueError(f"Unsupported optical architecture: {architecture}")
    kwargs = {key: value for key, value in config.items() if key != "architecture"}
    return SPARXOv2U4(**kwargs)
