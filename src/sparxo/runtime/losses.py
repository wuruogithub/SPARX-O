"""Losses for semantic and boundary supervision."""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def boundary_target(labels: Tensor, ignore_index: int = 19) -> Tensor:
    """One-pixel symmetric boundary map from a categorical label image."""
    if labels.ndim != 3:
        raise ValueError("labels must have shape [B,H,W]")
    valid = labels != ignore_index
    boundary = torch.zeros_like(labels, dtype=torch.bool)

    vertical = (labels[:, 1:] != labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    boundary[:, 1:] |= vertical
    boundary[:, :-1] |= vertical
    horizontal = (labels[:, :, 1:] != labels[:, :, :-1]) & valid[:, :, 1:] & valid[:, :, :-1]
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    return boundary.unsqueeze(1).to(torch.float32)


def parcel_boundary_target(parcels: Tensor) -> Tensor:
    """Optional boundary target from PASTIS parcel IDs."""
    if parcels.ndim != 3:
        raise ValueError("parcels must have shape [B,H,W]")
    boundary = torch.zeros_like(parcels, dtype=torch.bool)
    vertical = parcels[:, 1:] != parcels[:, :-1]
    horizontal = parcels[:, :, 1:] != parcels[:, :, :-1]
    boundary[:, 1:] |= vertical
    boundary[:, :-1] |= vertical
    boundary[:, :, 1:] |= horizontal
    boundary[:, :, :-1] |= horizontal
    return boundary.unsqueeze(1).to(torch.float32)


def semantic_boundary_consistency_loss(
    logits: Tensor,
    boundary_logits: Tensor,
    valid: Tensor,
) -> Tensor:
    """Match the learned boundary map to differentiable semantic variation."""

    probabilities = torch.softmax(logits.float(), dim=1)
    vertical = 0.5 * (
        probabilities[:, :, 1:] - probabilities[:, :, :-1]
    ).abs().sum(dim=1, keepdim=True)
    horizontal = 0.5 * (
        probabilities[:, :, :, 1:] - probabilities[:, :, :, :-1]
    ).abs().sum(dim=1, keepdim=True)
    vertical_edges = torch.maximum(
        F.pad(vertical, (0, 0, 1, 0)),
        F.pad(vertical, (0, 0, 0, 1)),
    )
    horizontal_edges = torch.maximum(
        F.pad(horizontal, (1, 0, 0, 0)),
        F.pad(horizontal, (0, 1, 0, 0)),
    )
    semantic_edges = torch.maximum(vertical_edges, horizontal_edges)
    difference = F.smooth_l1_loss(
        boundary_logits.float().sigmoid(),
        semantic_edges.clamp(0.0, 1.0),
        reduction="none",
        beta=0.1,
    )
    valid_float = valid.to(difference.dtype)
    return (
        (difference * valid_float).sum() / valid_float.sum().clamp_min(1.0)
    ).to(logits.dtype)


def directional_affinity_target(
    labels: Tensor,
    ignore_index: int = 19,
) -> tuple[Tensor, Tensor]:
    """Four directional same-region targets in north/south/west/east order."""
    valid = labels != ignore_index
    target = torch.zeros(
        (labels.shape[0], 4, labels.shape[1], labels.shape[2]),
        dtype=torch.float32,
        device=labels.device,
    )
    pair_valid = torch.zeros_like(target, dtype=torch.bool)
    vertical_same = (labels[:, 1:] == labels[:, :-1]) & valid[:, 1:] & valid[:, :-1]
    target[:, 0, 1:] = vertical_same
    target[:, 1, :-1] = vertical_same
    pair_valid[:, 0, 1:] = valid[:, 1:] & valid[:, :-1]
    pair_valid[:, 1, :-1] = valid[:, 1:] & valid[:, :-1]
    horizontal_same = (
        (labels[:, :, 1:] == labels[:, :, :-1])
        & valid[:, :, 1:]
        & valid[:, :, :-1]
    )
    target[:, 2, :, 1:] = horizontal_same
    target[:, 3, :, :-1] = horizontal_same
    pair_valid[:, 2, :, 1:] = valid[:, :, 1:] & valid[:, :, :-1]
    pair_valid[:, 3, :, :-1] = valid[:, :, 1:] & valid[:, :, :-1]
    return target, pair_valid


def soft_dice_loss(logits: Tensor, labels: Tensor, ignore_index: int = 19) -> Tensor:
    """Per-sample macro Dice so accumulation/DDP partitioning is invariant."""
    num_classes = logits.shape[1]
    valid = labels != ignore_index
    safe_labels = labels.masked_fill(~valid, 0)
    target = F.one_hot(safe_labels, num_classes=num_classes).permute(0, 3, 1, 2)
    target = target.to(logits.dtype) * valid.unsqueeze(1)
    probabilities = torch.softmax(logits, dim=1) * valid.unsqueeze(1)
    intersection = (probabilities * target).sum(dim=(2, 3))
    denominator = probabilities.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    present = target.sum(dim=(2, 3)) > 0
    dice = (2.0 * intersection + 1.0) / (denominator + 1.0)
    per_sample = [
        1.0 - dice[index, present[index]].mean()
        for index in range(logits.shape[0])
        if present[index].any()
    ]
    return torch.stack(per_sample).mean() if per_sample else logits.sum() * 0.0


def lovasz_gradient(sorted_foreground: Tensor) -> Tensor:
    """Gradient of the discrete Jaccard loss for sorted binary labels."""
    total = sorted_foreground.sum()
    intersection = total - sorted_foreground.float().cumsum(0)
    union = total + (1.0 - sorted_foreground).float().cumsum(0)
    gradient = 1.0 - intersection / union.clamp_min(1.0)
    if sorted_foreground.numel() > 1:
        gradient = torch.cat((gradient[:1], gradient[1:] - gradient[:-1]))
    return gradient


def lovasz_softmax_loss(logits: Tensor, labels: Tensor, ignore_index: int = 19) -> Tensor:
    """Per-sample, present-class Lovasz-Softmax surrogate for mean IoU."""
    probabilities = torch.softmax(logits.float(), dim=1)
    losses = []
    for probability, label in zip(probabilities, labels):
        valid = label != ignore_index
        if not valid.any():
            continue
        flat_label = label[valid]
        flat_probability = probability.permute(1, 2, 0)[valid]
        class_losses = []
        for class_index in flat_label.unique():
            foreground = (flat_label == class_index).float()
            errors = (foreground - flat_probability[:, class_index]).abs()
            errors, permutation = torch.sort(errors, descending=True)
            foreground = foreground[permutation]
            class_losses.append(torch.dot(errors, lovasz_gradient(foreground)))
        if class_losses:
            losses.append(torch.stack(class_losses).mean())
    return torch.stack(losses).mean().to(logits.dtype) if losses else logits.sum() * 0.0


def per_sample_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    ignore_index: int,
    label_smoothing: float,
) -> Tensor:
    pixels = F.cross_entropy(
        logits,
        labels,
        weight=class_weights,
        ignore_index=ignore_index,
        label_smoothing=label_smoothing,
        reduction="none",
    )
    valid = labels != ignore_index
    reduce_dims = tuple(range(1, labels.ndim))
    per_sample = (pixels * valid).sum(dim=reduce_dims)
    return (per_sample / valid.sum(dim=reduce_dims).clamp_min(1)).mean()


def focal_cross_entropy(
    logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    ignore_index: int,
    gamma: float,
) -> Tensor:
    """Multiclass focal CE used by Exchanger's MTLCC segmentation recipe."""
    valid = labels != ignore_index
    if not valid.any():
        return logits.sum() * 0.0
    safe_labels = labels.masked_fill(~valid, 0)
    log_probability = F.log_softmax(logits.float(), dim=1)
    target_log_probability = log_probability.gather(1, safe_labels.unsqueeze(1)).squeeze(1)
    target_probability = target_log_probability.exp()
    loss = -((1.0 - target_probability) ** gamma) * target_log_probability
    if class_weights is not None:
        loss = loss * class_weights[safe_labels]
    return loss[valid].mean().to(logits.dtype)


def balanced_boundary_bce(logits: Tensor, target: Tensor, valid: Tensor | None = None) -> Tensor:
    """Per-sample dynamically balanced BCE.

    Computing each sample independently makes the result invariant to
    micro-batch grouping and DDP world size.
    """
    if valid is None:
        valid = torch.ones_like(target, dtype=torch.bool)
    reduce_dims = tuple(range(1, target.ndim))
    positives = (target.bool() & valid).sum(dim=reduce_dims).to(logits.dtype)
    negatives = ((~target.bool()) & valid).sum(dim=reduce_dims).to(logits.dtype)
    pos_weight = (negatives / positives.clamp_min(1.0)).clamp(1.0, 20.0)
    positive_loss = F.softplus(-logits) * target * pos_weight.view(-1, 1, 1, 1)
    negative_loss = F.softplus(logits) * (1.0 - target)
    loss = (positive_loss + negative_loss) * valid
    valid_count = valid.sum(dim=reduce_dims).clamp_min(1).to(logits.dtype)
    return (loss.sum(dim=reduce_dims) / valid_count).mean()


def symmetric_balanced_bce(logits: Tensor, target: Tensor, valid: Tensor) -> Tensor:
    """Give positive and negative affinity pairs equal total influence."""
    reduce_dims = tuple(range(1, target.ndim))
    positive = target.bool() & valid
    negative = (~target.bool()) & valid
    positive_count = positive.sum(dim=reduce_dims).clamp_min(1).to(logits.dtype)
    negative_count = negative.sum(dim=reduce_dims).clamp_min(1).to(logits.dtype)
    positive_loss = (F.softplus(-logits) * positive).sum(dim=reduce_dims)
    negative_loss = (F.softplus(logits) * negative).sum(dim=reduce_dims)
    return (0.5 * positive_loss / positive_count + 0.5 * negative_loss / negative_count).mean()


def resize_labels(labels: Tensor, size: tuple[int, int]) -> Tensor:
    return F.interpolate(
        labels.unsqueeze(1).to(torch.float32),
        size=size,
        mode="nearest",
    ).squeeze(1).to(torch.long)


def region_losses(
    logits: Tensor,
    feature: Tensor,
    labels: Tensor,
    regions: Tensor,
    class_weights: Tensor | None,
) -> tuple[Tensor, Tensor]:
    """Classify target-derived components and compact their feature embeddings."""
    classification_losses = []
    compactness_losses = []
    for sample_index in range(logits.shape[0]):
        region_ids = regions[sample_index].reshape(-1)
        flat_labels = labels[sample_index].reshape(-1)
        valid = (
            (region_ids > 0)
            & (flat_labels >= 0)
            & (flat_labels < logits.shape[1])
        )
        if not valid.any():
            continue
        # Parcel IDs come from the full source tile and can be large, sparse
        # integers.  Using max(ID) as an allocation size makes a small crop
        # consume gigabytes.  Compact the IDs that are actually present in
        # this sample to [0, number_of_regions) first.
        _, ids = torch.unique(region_ids[valid], sorted=False, return_inverse=True)
        count = int(ids.max()) + 1
        region_counts = torch.zeros(count, dtype=logits.dtype, device=logits.device)
        region_counts.index_add_(0, ids, torch.ones_like(ids, dtype=logits.dtype))

        flat_logits = logits[sample_index].permute(1, 2, 0).reshape(-1, logits.shape[1])
        logit_sums = torch.zeros(
            (count, logits.shape[1]),
            dtype=logits.dtype,
            device=logits.device,
        )
        logit_sums.index_add_(0, ids, flat_logits[valid])
        region_logits = logit_sums / region_counts[:, None].clamp_min(1.0)

        label_votes = torch.zeros(
            (count, logits.shape[1]),
            dtype=logits.dtype,
            device=logits.device,
        )
        label_votes.index_add_(
            0,
            ids,
            F.one_hot(flat_labels[valid], num_classes=logits.shape[1]).to(logits.dtype),
        )
        region_labels = label_votes.argmax(dim=1)
        classification_losses.append(
            F.cross_entropy(region_logits, region_labels, weight=class_weights),
        )

        flat_feature = feature[sample_index].permute(1, 2, 0).reshape(-1, feature.shape[1])
        flat_feature = F.normalize(flat_feature, dim=1)
        feature_sums = torch.zeros(
            (count, feature.shape[1]),
            dtype=feature.dtype,
            device=feature.device,
        )
        feature_sums.index_add_(0, ids, flat_feature[valid])
        region_means = feature_sums / region_counts[:, None].clamp_min(1.0)
        deviations = (flat_feature[valid] - region_means[ids]).square().mean(dim=1)
        compactness_losses.append(deviations.mean())
    zero = logits.sum() * 0.0
    region = torch.stack(classification_losses).mean() if classification_losses else zero
    compactness = torch.stack(compactness_losses).mean() if compactness_losses else zero
    return region, compactness


def temporal_prototype_contrastive_loss(
    phase_features: Tensor,
    semantic_logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    ignore_index: int,
    temperature: float,
    max_pixels_per_class: int,
    prototype_fraction: float,
    hard_fraction: float,
) -> Tensor:
    """Separate crop classes by their complete Sentinel-2 phase trajectory.

    Prototypes are estimated from the most confidently classified pixels of
    each class in the distributed micro-batch.  The contrastive anchors come
    from the lower-confidence (but not only the single worst) part of the same
    classes.  This focuses the extra supervision on semantic confusions while
    avoiding an inference-time projection head or memory bank.
    """

    if phase_features.ndim != 5:
        raise ValueError("phase_features must have shape [B,K,C,H,W]")
    if temperature <= 0:
        raise ValueError("temporal prototype temperature must be positive")
    if max_pixels_per_class < 1:
        raise ValueError("max_pixels_per_class must be positive")
    if not 0.0 < prototype_fraction <= 1.0:
        raise ValueError("prototype_fraction must lie in (0,1]")
    if not 0.0 < hard_fraction <= 1.0:
        raise ValueError("hard_fraction must lie in (0,1]")

    spatial_size = phase_features.shape[-2:]
    if labels.shape[-2:] != spatial_size:
        labels = resize_labels(labels, spatial_size)
    if semantic_logits.shape[-2:] != spatial_size:
        semantic_logits = F.interpolate(
            semantic_logits,
            size=spatial_size,
            mode="bilinear",
            align_corners=False,
        )

    # A pixel descriptor contains its ordered calendar-phase trajectory.  The
    # calculation is kept in fp32 because the cosine margins are small.
    descriptors = phase_features.float().permute(0, 3, 4, 1, 2).flatten(3)
    descriptors = F.normalize(descriptors.reshape(-1, descriptors.shape[-1]), dim=1)
    flat_labels = labels.reshape(-1)
    valid = flat_labels != ignore_index
    if not valid.any():
        return phase_features.sum() * 0.0

    with torch.no_grad():
        probabilities = torch.softmax(semantic_logits.float(), dim=1)
        probabilities = probabilities.permute(0, 2, 3, 1).reshape(
            -1,
            semantic_logits.shape[1],
        )
        safe_labels = flat_labels.clamp(0, semantic_logits.shape[1] - 1)
        true_confidence = probabilities.gather(1, safe_labels[:, None]).squeeze(1)

    present_classes = flat_labels[valid].unique(sorted=True)
    if present_classes.numel() < 2:
        return phase_features.sum() * 0.0

    prototypes = []
    anchor_indices = []
    anchor_targets = []
    for prototype_index, class_index in enumerate(present_classes):
        indices = torch.nonzero(valid & (flat_labels == class_index), as_tuple=False).squeeze(1)
        confidence = true_confidence[indices]
        order = confidence.argsort()
        prototype_count = max(1, int(round(indices.numel() * prototype_fraction)))
        prototype_indices = indices[order[-prototype_count:]]
        prototypes.append(F.normalize(descriptors[prototype_indices].mean(0), dim=0))

        # Use a broad semi-hard pool.  It includes semantic mistakes but caps
        # every class equally, so background cannot dominate the objective.
        hard_count = min(
            max_pixels_per_class,
            max(1, int(round(indices.numel() * hard_fraction))),
        )
        selected = indices[order[:hard_count]]
        anchor_indices.append(selected)
        anchor_targets.append(
            torch.full(
                (selected.numel(),),
                prototype_index,
                dtype=torch.long,
                device=labels.device,
            ),
        )

    prototype_tensor = torch.stack(prototypes)
    query_tensor = descriptors[torch.cat(anchor_indices)]
    targets = torch.cat(anchor_targets)
    contrastive_logits = query_tensor @ prototype_tensor.transpose(0, 1)
    present_weights = (
        None
        if class_weights is None
        else class_weights[present_classes].float()
    )
    return F.cross_entropy(
        contrastive_logits / temperature,
        targets,
        weight=present_weights,
    ).to(phase_features.dtype)


def confidence_weighted_distillation(
    student_logits: Tensor,
    teacher_logits: Tensor,
    labels: Tensor,
    class_weights: Tensor | None,
    ignore_index: int,
    temperature: float,
    confidence_threshold: float,
    correct_only: bool = False,
    target_classes: list[int] | tuple[int, ...] | None = None,
) -> Tensor:
    """Train-only soft-target KL with auditable teacher-reliability filters."""
    if teacher_logits.shape != student_logits.shape:
        raise ValueError("Teacher and student logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("Distillation temperature must be positive")
    teacher_probabilities = torch.softmax(
        teacher_logits.float() / temperature,
        dim=1,
    )
    student_log_probabilities = torch.log_softmax(
        student_logits.float() / temperature,
        dim=1,
    )
    divergence = F.kl_div(
        student_log_probabilities,
        teacher_probabilities,
        reduction="none",
    ).sum(dim=1) * temperature**2
    confidence, prediction = teacher_probabilities.max(dim=1)
    valid = (labels != ignore_index) & (confidence >= confidence_threshold)
    if correct_only:
        valid = valid & (prediction == labels)
    if target_classes is not None:
        target_mask = torch.zeros_like(valid)
        for class_index in target_classes:
            if not 0 <= int(class_index) < student_logits.shape[1]:
                raise ValueError(
                    f"Distillation target class {class_index} is outside "
                    f"[0, {student_logits.shape[1]})",
                )
            target_mask |= labels == int(class_index)
        valid = valid & target_mask
    if not valid.any():
        return student_logits.sum() * 0.0
    safe_labels = labels.masked_fill(~valid, 0)
    weights = torch.ones_like(divergence)
    if class_weights is not None:
        weights = class_weights[safe_labels].to(divergence.dtype)
    weights = weights * valid
    reduce_dims = tuple(range(1, labels.ndim))
    per_sample = (divergence * weights).sum(dim=reduce_dims)
    normalization = weights.sum(dim=reduce_dims).clamp_min(1.0)
    return (per_sample / normalization).mean().to(student_logits.dtype)


def label_guided_inheritance_distillation(
    student_logits: Tensor,
    inherited_logits: Tensor,
    labels: Tensor,
    ignore_index: int,
    temperature: float,
    confidence_threshold: float,
) -> Tensor:
    """Preserve inherited decisions only where they are correct and trustworthy.

    This keeps the residual branch plastic on pixels the inherited model gets
    wrong, while directly penalizing collateral drift on its reliable pixels.
    """
    if inherited_logits.shape != student_logits.shape:
        raise ValueError("Inherited and student logits must have identical shapes")
    if temperature <= 0:
        raise ValueError("Inheritance temperature must be positive")
    inherited_probabilities = torch.softmax(
        inherited_logits.float() / temperature,
        dim=1,
    )
    student_log_probabilities = torch.log_softmax(
        student_logits.float() / temperature,
        dim=1,
    )
    divergence = F.kl_div(
        student_log_probabilities,
        inherited_probabilities,
        reduction="none",
    ).sum(dim=1) * temperature**2
    confidence, prediction = inherited_probabilities.max(dim=1)
    valid = (
        (labels != ignore_index)
        & (prediction == labels)
        & (confidence >= confidence_threshold)
    )
    if not valid.any():
        return student_logits.sum() * 0.0
    reduce_dims = tuple(range(1, labels.ndim))
    weights = valid.to(divergence.dtype)
    per_sample = (divergence * weights).sum(dim=reduce_dims)
    normalization = weights.sum(dim=reduce_dims).clamp_min(1.0)
    return (per_sample / normalization).mean().to(student_logits.dtype)


class SPARXLoss(nn.Module):
    def __init__(
        self,
        ce_weight: float = 1.0,
        dice_weight: float = 0.5,
        lovasz_weight: float = 0.0,
        boundary_weight: float = 0.2,
        boundary_consistency_weight: float = 0.0,
        affinity_weight: float = 0.0,
        deep_supervision_weight: float = 0.0,
        auxiliary_weight: float = 0.0,
        auxiliary_dice_weight: float = 0.25,
        anchor_weight: float = 0.0,
        anchor_dice_weight: float = 0.20,
        anchor_lovasz_weight: float = 0.30,
        anchor_boundary_weight: float = 0.10,
        anchor_class_weights: list[float] | None = None,
        parcel_boundary_weight: float = 0.0,
        region_weight: float = 0.0,
        region_compactness_weight: float = 0.0,
        distillation_weight: float = 0.0,
        distillation_temperature: float = 2.0,
        distillation_confidence_threshold: float = 0.0,
        distillation_correct_only: bool = False,
        distillation_target_classes: list[int] | None = None,
        inheritance_weight: float = 0.0,
        inheritance_temperature: float = 1.0,
        inheritance_confidence_threshold: float = 0.6,
        temporal_reconstruction_weight: float = 0.0,
        temporal_reconstruction_position_power: float = 1.0,
        temporal_reconstruction_dynamic_balance: bool = False,
        temporal_prototype_weight: float = 0.0,
        temporal_prototype_temperature: float = 0.12,
        temporal_prototype_max_pixels_per_class: int = 96,
        temporal_prototype_fraction: float = 0.50,
        temporal_prototype_hard_fraction: float = 0.35,
        phase_offset_magnitude_weight: float = 0.0,
        phase_offset_smoothness_weight: float = 0.0,
        mask2former_weight: float = 0.0,
        mask2former_class_weight: float = 2.0,
        mask2former_mask_weight: float = 5.0,
        mask2former_dice_weight: float = 5.0,
        mask2former_no_object_weight: float = 0.1,
        focal_gamma: float = 0.0,
        label_smoothing: float = 0.05,
        class_weights: list[float] | None = None,
        ignore_index: int = 19,
    ) -> None:
        super().__init__()
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.lovasz_weight = lovasz_weight
        self.boundary_weight = boundary_weight
        self.boundary_consistency_weight = boundary_consistency_weight
        self.affinity_weight = affinity_weight
        self.deep_supervision_weight = deep_supervision_weight
        self.auxiliary_weight = auxiliary_weight
        self.auxiliary_dice_weight = auxiliary_dice_weight
        self.anchor_weight = anchor_weight
        self.anchor_dice_weight = anchor_dice_weight
        self.anchor_lovasz_weight = anchor_lovasz_weight
        self.anchor_boundary_weight = anchor_boundary_weight
        self.parcel_boundary_weight = parcel_boundary_weight
        self.region_weight = region_weight
        self.region_compactness_weight = region_compactness_weight
        self.distillation_weight = distillation_weight
        self.distillation_temperature = distillation_temperature
        self.distillation_confidence_threshold = distillation_confidence_threshold
        self.distillation_correct_only = distillation_correct_only
        self.distillation_target_classes = distillation_target_classes
        self.inheritance_weight = inheritance_weight
        self.inheritance_temperature = inheritance_temperature
        self.inheritance_confidence_threshold = inheritance_confidence_threshold
        self.temporal_reconstruction_weight = temporal_reconstruction_weight
        self.temporal_reconstruction_position_power = (
            temporal_reconstruction_position_power
        )
        self.temporal_reconstruction_dynamic_balance = bool(
            temporal_reconstruction_dynamic_balance
        )
        self.temporal_prototype_weight = temporal_prototype_weight
        self.temporal_prototype_temperature = temporal_prototype_temperature
        self.temporal_prototype_max_pixels_per_class = (
            temporal_prototype_max_pixels_per_class
        )
        self.temporal_prototype_fraction = temporal_prototype_fraction
        self.temporal_prototype_hard_fraction = temporal_prototype_hard_fraction
        self.phase_offset_magnitude_weight = phase_offset_magnitude_weight
        self.phase_offset_smoothness_weight = phase_offset_smoothness_weight
        self.mask2former_weight = mask2former_weight
        self.mask2former_class_weight = mask2former_class_weight
        self.mask2former_mask_weight = mask2former_mask_weight
        self.mask2former_dice_weight = mask2former_dice_weight
        self.mask2former_no_object_weight = mask2former_no_object_weight
        if focal_gamma < 0:
            raise ValueError("focal_gamma must be non-negative")
        self.focal_gamma = float(focal_gamma)
        self.label_smoothing = label_smoothing
        self.ignore_index = ignore_index
        weights = None if class_weights is None else torch.tensor(class_weights, dtype=torch.float32)
        if weights is not None and weights.numel() == 0:
            raise ValueError("class_weights cannot be empty")
        self.register_buffer("class_weights", weights)
        anchor_weights = (
            None
            if anchor_class_weights is None
            else torch.tensor(anchor_class_weights, dtype=torch.float32)
        )
        if anchor_weights is not None and anchor_weights.numel() == 0:
            raise ValueError("anchor_class_weights cannot be empty")
        self.register_buffer("anchor_class_weights", anchor_weights)

    def forward(
        self,
        output: dict[str, Tensor],
        labels: Tensor,
        parcels: Tensor | None = None,
        regions: Tensor | None = None,
        teacher_logits: Tensor | None = None,
    ) -> dict[str, Tensor]:
        logits = output["logits"]
        if self.class_weights is not None and self.class_weights.numel() != logits.shape[1]:
            raise ValueError("class_weights length must equal the number of logits classes")
        ce = (
            focal_cross_entropy(
                logits,
                labels,
                self.class_weights,
                self.ignore_index,
                self.focal_gamma,
            )
            if self.focal_gamma > 0
            else per_sample_cross_entropy(
                logits,
                labels,
                self.class_weights,
                self.ignore_index,
                self.label_smoothing,
            )
        )
        dice = soft_dice_loss(logits, labels, self.ignore_index)
        lovasz = (
            lovasz_softmax_loss(logits, labels, self.ignore_index)
            if self.lovasz_weight > 0
            else logits.sum() * 0.0
        )
        valid = (labels != self.ignore_index).unsqueeze(1)
        semantic_edges = boundary_target(labels, self.ignore_index)
        boundary = balanced_boundary_bce(output["boundary_logits"], semantic_edges, valid)
        boundary_consistency = logits.sum() * 0.0
        if self.boundary_consistency_weight > 0:
            boundary_consistency = semantic_boundary_consistency_loss(
                logits,
                output["boundary_logits"],
                valid,
            )
        affinity = logits.sum() * 0.0
        if self.affinity_weight > 0:
            if "affinity_logits" not in output:
                raise ValueError("affinity_weight > 0 requires affinity_logits")
            affinity_target, affinity_valid = directional_affinity_target(
                labels,
                self.ignore_index,
            )
            affinity = symmetric_balanced_bce(
                output["affinity_logits"],
                affinity_target,
                affinity_valid,
            )
        deep_supervision = logits.sum() * 0.0
        if self.deep_supervision_weight > 0:
            if "deep_logits" not in output:
                raise ValueError("deep_supervision_weight > 0 requires deep_logits")
            deep_losses = []
            for deep_logits in output["deep_logits"]:
                deep_labels = resize_labels(labels, deep_logits.shape[-2:])
                deep_losses.append(
                    per_sample_cross_entropy(
                        deep_logits,
                        deep_labels,
                        self.class_weights,
                        self.ignore_index,
                        self.label_smoothing,
                    ),
                )
            deep_supervision = torch.stack(deep_losses).mean()
        parcel_boundary = logits.sum() * 0.0
        if self.parcel_boundary_weight > 0:
            if parcels is None:
                raise ValueError("parcel_boundary_weight > 0 requires parcel tensors")
            parcel_boundary = balanced_boundary_bce(
                output["boundary_logits"],
                parcel_boundary_target(parcels),
                valid,
            )
        auxiliary = logits.sum() * 0.0
        # Curriculum schedules may deliberately anneal the auxiliary loss to
        # exactly zero while the model still emits auxiliary logits.  Retain a
        # zero-valued autograd dependency on every auxiliary output so DDP sees
        # those head parameters as participating in the graph.  This changes
        # neither the forward value nor any gradient, and avoids a reduction
        # failure at the first zero-weight epoch without the overhead of global
        # unused-parameter discovery.
        inactive_auxiliary_logits = output.get("aux_logits")
        if inactive_auxiliary_logits is not None:
            inactive_streams = (
                inactive_auxiliary_logits.unbind(dim=1)
                if isinstance(inactive_auxiliary_logits, Tensor)
                else inactive_auxiliary_logits
            )
            for inactive_logits in inactive_streams:
                auxiliary = auxiliary + inactive_logits.sum() * 0.0
        if self.auxiliary_weight > 0:
            if "aux_logits" not in output:
                raise ValueError("auxiliary_weight > 0 requires aux_logits from the model")
            stream_losses = []
            auxiliary_logits = output["aux_logits"]
            streams = (
                auxiliary_logits.unbind(dim=1)
                if isinstance(auxiliary_logits, Tensor)
                else auxiliary_logits
            )
            # Newer models can expose an availability tensor already aligned
            # with aux_logits.  This avoids assuming that the general modality
            # vector and the auxiliary-head list have the same order.
            aux_availability = output.get("aux_availability")
            availability = output.get("availability")
            if aux_availability is not None:
                if (
                    aux_availability.ndim != 2
                    or aux_availability.shape[0] != labels.shape[0]
                    or aux_availability.shape[1] != len(streams)
                ):
                    raise ValueError(
                        "aux_availability must have shape [B,num_aux_streams]",
                    )
            elif availability is not None:
                if availability.ndim != 2 or availability.shape[0] != labels.shape[0]:
                    raise ValueError("availability must have shape [B,1+num_aux_streams]")
                if availability.shape[1] != len(streams) + 1:
                    raise ValueError(
                        "availability must contain geometry followed by every auxiliary stream",
                    )
            for stream_index, stream_logits in enumerate(streams):
                stream_available = (
                    torch.ones(labels.shape[0], dtype=torch.bool, device=labels.device)
                    if aux_availability is None and availability is None
                    else (
                        aux_availability[:, stream_index].bool()
                        if aux_availability is not None
                        else availability[:, stream_index + 1].bool()
                    )
                )
                # Keep the zero-valued dependency so DDP still sees every
                # auxiliary head on ranks where an entire stream was dropped.
                zero = stream_logits.sum() * 0.0
                if not stream_available.any():
                    stream_losses.append(zero)
                    continue
                stream_ce = per_sample_cross_entropy(
                    stream_logits[stream_available],
                    labels[stream_available],
                    self.class_weights,
                    self.ignore_index,
                    self.label_smoothing,
                )
                stream_dice = soft_dice_loss(
                    stream_logits[stream_available],
                    labels[stream_available],
                    self.ignore_index,
                )
                stream_losses.append(
                    zero + stream_ce + self.auxiliary_dice_weight * stream_dice,
                )
            auxiliary = torch.stack(stream_losses).mean()
        anchor = logits.sum() * 0.0
        if self.anchor_weight > 0:
            anchor_logits = output.get("anchor_logits")
            if anchor_logits is None:
                raise ValueError("anchor_weight > 0 requires anchor_logits")
            anchor_ce = per_sample_cross_entropy(
                anchor_logits,
                labels,
                self.anchor_class_weights,
                self.ignore_index,
                self.label_smoothing,
            )
            anchor_dice = soft_dice_loss(
                anchor_logits,
                labels,
                self.ignore_index,
            )
            anchor_lovasz = (
                lovasz_softmax_loss(anchor_logits, labels, self.ignore_index)
                if self.anchor_lovasz_weight > 0
                else anchor_logits.sum() * 0.0
            )
            anchor_boundary = anchor_logits.sum() * 0.0
            if self.anchor_boundary_weight > 0:
                anchor_boundary_logits = output.get("anchor_boundary_logits")
                if anchor_boundary_logits is None:
                    raise ValueError(
                        "positive anchor_boundary_weight requires "
                        "anchor_boundary_logits",
                    )
                anchor_boundary = balanced_boundary_bce(
                    anchor_boundary_logits,
                    semantic_edges,
                    valid,
                )
            anchor = (
                anchor_ce
                + self.anchor_dice_weight * anchor_dice
                + self.anchor_lovasz_weight * anchor_lovasz
                + self.anchor_boundary_weight * anchor_boundary
            )
        region = logits.sum() * 0.0
        region_compactness = logits.sum() * 0.0
        if self.region_weight > 0 or self.region_compactness_weight > 0:
            if regions is None or "decoder_feature" not in output:
                raise ValueError("region losses require regions and decoder_feature")
            region, region_compactness = region_losses(
                logits,
                output["decoder_feature"],
                labels,
                regions,
                self.class_weights,
            )
        distillation = logits.sum() * 0.0
        if self.distillation_weight > 0 and teacher_logits is not None:
            distillation = confidence_weighted_distillation(
                logits,
                teacher_logits,
                labels,
                self.class_weights,
                self.ignore_index,
                self.distillation_temperature,
                self.distillation_confidence_threshold,
                self.distillation_correct_only,
                self.distillation_target_classes,
            )
        inheritance = logits.sum() * 0.0
        if self.inheritance_weight > 0:
            inherited_logits = output.get("legacy_logits")
            if inherited_logits is None:
                raise ValueError("inheritance_weight > 0 requires legacy_logits")
            inheritance = label_guided_inheritance_distillation(
                logits,
                inherited_logits,
                labels,
                self.ignore_index,
                self.inheritance_temperature,
                self.inheritance_confidence_threshold,
            )
        temporal_reconstruction = logits.sum() * 0.0
        inactive_reconstruction = output.get("s2_phase_reconstruction")
        if inactive_reconstruction is not None:
            # Keep the reconstruction head in the DDP autograd graph after the
            # late curriculum anneals its coefficient exactly to zero.
            temporal_reconstruction = (
                temporal_reconstruction + inactive_reconstruction.sum() * 0.0
            )
        if self.temporal_reconstruction_weight > 0:
            prediction = output.get("s2_phase_reconstruction")
            target = output.get("s2_phase_target")
            if prediction is None or target is None:
                raise ValueError(
                    "temporal_reconstruction_weight > 0 requires S2 phase "
                    "reconstruction outputs",
                )
            if prediction.shape != target.shape or prediction.ndim != 5:
                raise ValueError(
                    "S2 phase prediction and target must have matching "
                    "[B,K,C,H,W] shapes",
                )
            phase_count = prediction.shape[1]
            position_weights = torch.linspace(
                1.0 / phase_count,
                1.0,
                phase_count,
                device=prediction.device,
                dtype=torch.float32,
            ).pow(self.temporal_reconstruction_position_power)
            position_weights = position_weights / position_weights.mean()
            phase_loss = F.smooth_l1_loss(
                prediction.float(),
                target.float(),
                beta=0.1,
                reduction="none",
            ).mean(dim=(2, 3, 4))
            reconstruction_valid = output.get("s2_reconstruction_valid")
            if reconstruction_valid is None:
                temporal_reconstruction = (
                    phase_loss * position_weights.unsqueeze(0)
                ).mean().to(logits.dtype)
            else:
                if reconstruction_valid.shape != phase_loss.shape:
                    raise ValueError(
                        "s2_reconstruction_valid must match [B,K] reconstruction loss",
                    )
                weights = (
                    position_weights.unsqueeze(0)
                    * reconstruction_valid.to(position_weights.dtype)
                )
                temporal_reconstruction = (
                    (phase_loss * weights).sum() / weights.sum().clamp_min(1.0)
                ).to(logits.dtype)
        temporal_prototype = logits.sum() * 0.0
        if self.temporal_prototype_weight > 0:
            phase_features = output.get("s2_phase_features")
            if phase_features is None:
                raise ValueError(
                    "temporal_prototype_weight > 0 requires s2_phase_features",
                )
            temporal_prototype = temporal_prototype_contrastive_loss(
                phase_features,
                logits,
                labels,
                self.class_weights,
                self.ignore_index,
                self.temporal_prototype_temperature,
                self.temporal_prototype_max_pixels_per_class,
                self.temporal_prototype_fraction,
                self.temporal_prototype_hard_fraction,
            )
        phase_offset_magnitude = logits.sum() * 0.0
        phase_offset_smoothness = logits.sum() * 0.0
        if (
            self.phase_offset_magnitude_weight > 0
            or self.phase_offset_smoothness_weight > 0
        ):
            phase_offset = output.get("s2_phase_offset_fraction")
            if phase_offset is None:
                raise ValueError(
                    "phase offset regularization requires "
                    "s2_phase_offset_fraction",
                )
            if phase_offset.ndim != 4 or phase_offset.shape[1] != 1:
                raise ValueError(
                    "s2_phase_offset_fraction must have shape [B,1,H,W]",
                )
            phase_offset_magnitude = phase_offset.float().square().mean().to(logits.dtype)
            differences = []
            if phase_offset.shape[-2] > 1:
                differences.append(
                    (phase_offset[..., 1:, :] - phase_offset[..., :-1, :]).abs().mean(),
                )
            if phase_offset.shape[-1] > 1:
                differences.append(
                    (phase_offset[..., :, 1:] - phase_offset[..., :, :-1]).abs().mean(),
                )
            if differences:
                phase_offset_smoothness = torch.stack(differences).mean().to(logits.dtype)
        mask2former_class = logits.sum() * 0.0
        mask2former_mask = logits.sum() * 0.0
        mask2former_dice = logits.sum() * 0.0
        if self.mask2former_weight > 0:
            query_classes = output.get("query_class_logits")
            query_masks = output.get("query_mask_logits")
            if query_classes is None or query_masks is None:
                raise ValueError(
                    "mask2former_weight > 0 requires query class and mask logits",
                )
            valid_classes = logits.shape[1] - 1
            if (
                query_classes.ndim != 3
                or query_masks.ndim != 4
                or query_classes.shape[:2] != query_masks.shape[:2]
                or query_masks.shape[1] != valid_classes
                or query_classes.shape[-1] != valid_classes + 1
            ):
                raise ValueError(
                    "semantic Mask2Former requires one query per evaluated class",
                )
            if query_masks.shape[-2:] != labels.shape[-2:]:
                query_masks = F.interpolate(
                    query_masks,
                    size=labels.shape[-2:],
                    mode="bilinear",
                    align_corners=False,
                )
            class_ids = torch.arange(valid_classes, device=labels.device)
            target_masks = labels[:, None] == class_ids[None, :, None, None]
            present = target_masks.flatten(2).any(dim=-1)
            no_object = torch.full_like(class_ids, valid_classes)[None].expand_as(present)
            class_targets = torch.where(
                present,
                class_ids[None].expand_as(present),
                no_object,
            )
            class_weights = torch.ones(
                valid_classes + 1,
                dtype=query_classes.dtype,
                device=query_classes.device,
            )
            class_weights[-1] = self.mask2former_no_object_weight
            mask2former_class = F.cross_entropy(
                query_classes.flatten(0, 1),
                class_targets.flatten(),
                weight=class_weights,
            )
            valid_pixels = (labels != self.ignore_index)[:, None]
            target_float = target_masks.to(query_masks.dtype)
            mask_bce = F.binary_cross_entropy_with_logits(
                query_masks,
                target_float,
                reduction="none",
            )
            mask_bce = (mask_bce * valid_pixels).flatten(2).sum(dim=-1)
            mask_bce = mask_bce / valid_pixels.flatten(2).sum(dim=-1).clamp_min(1)
            probabilities = query_masks.sigmoid() * valid_pixels
            targets = target_float * valid_pixels
            intersection = (probabilities * targets).flatten(2).sum(dim=-1)
            denominator = probabilities.flatten(2).sum(dim=-1) + targets.flatten(2).sum(dim=-1)
            mask_dice = 1.0 - (2.0 * intersection + 1.0) / (denominator + 1.0)
            if present.any():
                mask2former_mask = mask_bce[present].mean()
                mask2former_dice = mask_dice[present].mean()
            else:
                mask2former_mask = query_masks.sum() * 0.0
                mask2former_dice = query_masks.sum() * 0.0
        mask2former = (
            self.mask2former_class_weight * mask2former_class
            + self.mask2former_mask_weight * mask2former_mask
            + self.mask2former_dice_weight * mask2former_dice
        )
        reconstruction_scale = logits.new_ones(())
        if self.temporal_reconstruction_dynamic_balance and self.temporal_reconstruction_weight > 0:
            semantic_reference = (
                self.ce_weight * ce.detach()
                + self.dice_weight * dice.detach()
                + self.lovasz_weight * lovasz.detach()
                + self.boundary_weight * boundary.detach()
            ).clamp_min(1e-6)
            reconstruction_scale = (
                semantic_reference / temporal_reconstruction.detach().clamp_min(1e-6)
            ).clamp(0.1, 10.0)
        total = (
            self.ce_weight * ce
            + self.dice_weight * dice
            + self.lovasz_weight * lovasz
            + self.boundary_weight * boundary
            + self.boundary_consistency_weight * boundary_consistency
            + self.affinity_weight * affinity
            + self.deep_supervision_weight * deep_supervision
            + self.parcel_boundary_weight * parcel_boundary
            + self.auxiliary_weight * auxiliary
            + self.anchor_weight * anchor
            + self.region_weight * region
            + self.region_compactness_weight * region_compactness
            + self.distillation_weight * distillation
            + self.inheritance_weight * inheritance
            + self.temporal_reconstruction_weight
            * reconstruction_scale
            * temporal_reconstruction
            + self.temporal_prototype_weight * temporal_prototype
            + self.phase_offset_magnitude_weight * phase_offset_magnitude
            + self.phase_offset_smoothness_weight * phase_offset_smoothness
            + self.mask2former_weight * mask2former
        )
        return {
            "total": total,
            "ce": ce.detach(),
            "dice": dice.detach(),
            "lovasz": lovasz.detach(),
            "boundary": boundary.detach(),
            "boundary_consistency": boundary_consistency.detach(),
            "affinity": affinity.detach(),
            "deep_supervision": deep_supervision.detach(),
            "parcel_boundary": parcel_boundary.detach(),
            "auxiliary": auxiliary.detach(),
            "anchor": anchor.detach(),
            "region": region.detach(),
            "region_compactness": region_compactness.detach(),
            "distillation": distillation.detach(),
            "inheritance": inheritance.detach(),
            "temporal_reconstruction": temporal_reconstruction.detach(),
            "temporal_prototype": temporal_prototype.detach(),
            "phase_offset_magnitude": phase_offset_magnitude.detach(),
            "phase_offset_smoothness": phase_offset_smoothness.detach(),
            "mask2former": mask2former.detach(),
            "mask2former_class": mask2former_class.detach(),
            "mask2former_mask": mask2former_mask.detach(),
            "mask2former_dice": mask2former_dice.detach(),
        }


def build_loss(config: dict[str, Any], ignore_index: int) -> nn.Module:
    """Build the semantic segmentation loss."""
    return SPARXLoss(ignore_index=ignore_index, **config)
