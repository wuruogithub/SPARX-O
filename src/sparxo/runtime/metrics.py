"""Streaming semantic and boundary metrics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor

from .losses import boundary_target

CLASS_NAMES = [
    "Background",
    "Meadow",
    "Soft winter wheat",
    "Corn",
    "Winter barley",
    "Winter rapeseed",
    "Spring barley",
    "Sunflower",
    "Grapevine",
    "Beet",
    "Winter triticale",
    "Winter durum wheat",
    "Fruits, vegetables, flowers",
    "Potatoes",
    "Leguminous fodder",
    "Soybeans",
    "Orchard",
    "Mixed cereal",
    "Sorghum",
]


@dataclass
class MetricState:
    confusion: Tensor
    boundary_tp: Tensor
    boundary_fp: Tensor
    boundary_fn: Tensor


class SegmentationMetrics:
    def __init__(
        self,
        num_classes: int = 19,
        ignore_index: int = 19,
        device: str = "cpu",
        class_names: list[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.class_names = tuple(class_names) if class_names is not None else tuple(CLASS_NAMES)
        evaluated_indices = [
            index for index in range(self.num_classes) if index != self.ignore_index
        ]
        if evaluated_indices and len(self.class_names) <= max(evaluated_indices):
            raise ValueError("class_names must cover every evaluated class")
        self.state = MetricState(
            confusion=torch.zeros(num_classes, num_classes, dtype=torch.int64, device=device),
            boundary_tp=torch.zeros((), dtype=torch.int64, device=device),
            boundary_fp=torch.zeros((), dtype=torch.int64, device=device),
            boundary_fn=torch.zeros((), dtype=torch.int64, device=device),
        )

    @torch.no_grad()
    def update(self, logits: Tensor, labels: Tensor, boundary_logits: Tensor | None = None) -> None:
        predictions = logits.argmax(dim=1)
        valid = labels != self.ignore_index
        encoded = labels[valid] * self.num_classes + predictions[valid]
        self.state.confusion += torch.bincount(
            encoded,
            minlength=self.num_classes**2,
        ).reshape(self.num_classes, self.num_classes)

        if boundary_logits is not None:
            true_edges = boundary_target(labels, self.ignore_index).bool()
            pred_edges = torch.sigmoid(boundary_logits) >= 0.5
            valid_edges = valid.unsqueeze(1)
            self.state.boundary_tp += (pred_edges & true_edges & valid_edges).sum()
            self.state.boundary_fp += (pred_edges & ~true_edges & valid_edges).sum()
            self.state.boundary_fn += (~pred_edges & true_edges & valid_edges).sum()

    def synchronize(self) -> None:
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            return
        for value in (
            self.state.confusion,
            self.state.boundary_tp,
            self.state.boundary_fp,
            self.state.boundary_fn,
        ):
            torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)

    def compute(self) -> dict[str, Any]:
        cm = self.state.confusion.to(torch.float64)
        true_positive = cm.diag()
        false_positive = cm.sum(0) - true_positive
        false_negative = cm.sum(1) - true_positive
        union = true_positive + false_positive + false_negative
        iou = true_positive / union.clamp_min(1.0)
        f1 = 2.0 * true_positive / (2.0 * true_positive + false_positive + false_negative).clamp_min(1.0)
        evaluated = torch.ones(self.num_classes, dtype=torch.bool, device=cm.device)
        if 0 <= self.ignore_index < self.num_classes:
            evaluated[self.ignore_index] = False
        present = (union > 0) & evaluated
        total = cm.sum()
        boundary_denominator = (
            2 * self.state.boundary_tp + self.state.boundary_fp + self.state.boundary_fn
        ).clamp_min(1)
        result = {
            "miou": float(iou[present].mean()),
            "macro_f1": float(f1[present].mean()),
            "overall_accuracy": float(true_positive.sum() / total.clamp_min(1.0)),
            "boundary_f1": float(2 * self.state.boundary_tp / boundary_denominator),
            "per_class_iou": {
                self.class_names[index] if index < len(self.class_names) else str(index): (
                    float(iou[index]) if present[index] else None
                )
                for index in range(self.num_classes)
                if index != self.ignore_index
            },
            "confusion_matrix": self.state.confusion.cpu().tolist(),
        }
        return result
