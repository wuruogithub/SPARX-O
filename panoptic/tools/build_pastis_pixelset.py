#!/usr/bin/env python3
"""Build Exchanger's PASTIS-R PixelSet layout from the local PASTIS-R tiles.

Each non-void parcel instance becomes one sample.  Its full Sentinel-2 time
series is stored as [time, band, parcel_pixel]; the Exchanger reader performs
the stochastic 16-bin pixel averaging at training time.

The builder is restartable: parcel files are written atomically and existing
valid files in the ``.building`` directory are reused.  Dataset-level metadata
is only published after a complete audit succeeds.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np


VALID_LABELS = frozenset(range(1, 19))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--progress-every", type=int, default=25)
    parser.add_argument(
        "--keep-building-dir",
        action="store_true",
        help="Do not atomically rename OUTPUT.building to OUTPUT at completion.",
    )
    return parser.parse_args()


def load_features(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        collection = json.load(handle)
    features = collection.get("features", [])
    if not features:
        raise RuntimeError(f"No features found in {path}")
    features.sort(key=lambda f: int(f["properties"]["ID_PATCH"]))
    return features


def atomic_save_npy(path: Path, array: np.ndarray) -> bool:
    """Save an NPY atomically; return False when a valid file already exists."""
    if path.exists():
        try:
            old = np.load(path, mmap_mode="r")
            if old.shape == array.shape and old.dtype == array.dtype:
                return False
        except Exception:
            pass
    tmp = path.with_suffix(path.suffix + f".tmp.{os.getpid()}")
    with tmp.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    return True


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    args = parse_args()
    source = args.source.resolve()
    output = args.output.resolve()
    building = output.with_name(output.name + ".building")

    required = (
        source / "metadata.geojson",
        source / "DATA_S2",
        source / "ANNOTATIONS",
        source / "INSTANCE_ANNOTATIONS",
    )
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source assets: " + ", ".join(missing))
    if output.exists():
        raise FileExistsError(
            f"Final output already exists: {output}. Refusing to overwrite it."
        )
    if output.parent != source.parent:
        raise RuntimeError(
            "For safe atomic publication, source and output must share a parent."
        )

    data_out = building / "DATA_S2"
    data_out.mkdir(parents=True, exist_ok=True)
    features = load_features(source / "metadata.geojson")

    # One accumulator per fold and band.  Values are computed over the exact
    # clipped parcel values that the Exchanger reader will normalize.
    sums = {fold: np.zeros(10, dtype=np.float64) for fold in range(1, 6)}
    sums_sq = {fold: np.zeros(10, dtype=np.float64) for fold in range(1, 6)}
    counts = {fold: np.zeros(10, dtype=np.int64) for fold in range(1, 6)}
    class_counts = {fold: {label: 0 for label in VALID_LABELS} for fold in range(1, 6)}
    patch_counts = {fold: 0 for fold in range(1, 6)}
    rows: list[dict] = []
    excluded_void = 0
    reused_files = 0
    written_files = 0
    non_pure_instances = 0
    total_pixel_observations = 0
    started = time.time()

    for patch_index, feature in enumerate(features, start=1):
        properties = feature["properties"]
        patch_id = int(properties["ID_PATCH"])
        fold = int(properties["Fold"])
        if fold not in patch_counts:
            raise ValueError(f"Invalid fold {fold} for patch {patch_id}")
        patch_counts[fold] += 1

        s2_path = source / "DATA_S2" / f"S2_{patch_id}.npy"
        instance_path = source / "INSTANCE_ANNOTATIONS" / f"INSTANCES_{patch_id}.npy"
        target_path = source / "ANNOTATIONS" / f"TARGET_{patch_id}.npy"
        if not (s2_path.exists() and instance_path.exists() and target_path.exists()):
            raise FileNotFoundError(f"Incomplete source tile for patch {patch_id}")

        s2 = np.load(s2_path, mmap_mode="r")
        instances = np.asarray(np.load(instance_path)).squeeze()
        target = np.asarray(np.load(target_path, mmap_mode="r"))[0]
        if s2.ndim != 4 or s2.shape[1] != 10:
            raise ValueError(f"Unexpected S2 shape {s2.shape} for patch {patch_id}")
        if tuple(instances.shape) != tuple(s2.shape[-2:]) or target.shape != instances.shape:
            raise ValueError(f"Spatial shape mismatch for patch {patch_id}")

        local_ids = np.unique(instances)
        local_ids = local_ids[local_ids != 0]
        if local_ids.size and int(local_ids.max()) >= 1000:
            raise ValueError(
                f"Patch {patch_id} has local instance id >=1000; global id scheme unsafe"
            )

        for local_id_np in local_ids:
            local_id = int(local_id_np)
            mask = instances == local_id
            labels, label_counts = np.unique(target[mask], return_counts=True)
            majority_index = int(np.argmax(label_counts))
            label = int(labels[majority_index])
            purity = float(label_counts[majority_index] / label_counts.sum())
            if purity < 1.0:
                non_pure_instances += 1
            if label not in VALID_LABELS:
                excluded_void += 1
                continue

            parcel_id = patch_id * 1000 + local_id
            parcel = np.asarray(s2[:, :, mask])  # [T, C, N]
            if parcel.ndim != 3 or parcel.shape[2] < 1:
                raise RuntimeError(f"Empty parcel {parcel_id}")
            parcel = np.ascontiguousarray(parcel)
            parcel_path = data_out / f"S2_{parcel_id}.npy"
            if atomic_save_npy(parcel_path, parcel):
                written_files += 1
            else:
                reused_files += 1

            # Match PASTISPixelSetReader's clipping before z-normalization.
            values = np.clip(parcel, 0, 65535).astype(np.float64, copy=False)
            sums[fold] += values.sum(axis=(0, 2))
            sums_sq[fold] += np.square(values).sum(axis=(0, 2))
            n_per_band = values.shape[0] * values.shape[2]
            counts[fold] += n_per_band
            total_pixel_observations += n_per_band
            class_counts[fold][label] += 1
            rows.append(
                {
                    "ID_PARCEL": parcel_id,
                    "ID_PATCH": patch_id,
                    "LOCAL_INSTANCE_ID": local_id,
                    "Label": label,
                    "Fold": fold,
                    "N_PIXELS": int(parcel.shape[2]),
                    "LABEL_PURITY": f"{purity:.8f}",
                }
            )

        if patch_index % args.progress_every == 0 or patch_index == len(features):
            elapsed = time.time() - started
            rate = patch_index / max(elapsed, 1e-9)
            eta = (len(features) - patch_index) / max(rate, 1e-9)
            print(
                f"[{patch_index:4d}/{len(features)}] parcels={len(rows):6d} "
                f"written={written_files:6d} reused={reused_files:6d} "
                f"elapsed={elapsed/60:.1f}m eta={eta/60:.1f}m",
                flush=True,
            )

    rows.sort(key=lambda row: int(row["ID_PARCEL"]))
    parcel_ids = [int(row["ID_PARCEL"]) for row in rows]
    if len(parcel_ids) != len(set(parcel_ids)):
        raise RuntimeError("Global parcel IDs are not unique")
    if not rows:
        raise RuntimeError("No valid parcel was constructed")

    metadata_tmp = building / "metadata_parcel.csv.tmp"
    metadata_path = building / "metadata_parcel.csv"
    fieldnames = list(rows[0].keys())
    with metadata_tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(metadata_tmp, metadata_path)

    norm = {}
    for fold in range(1, 6):
        if np.any(counts[fold] == 0):
            raise RuntimeError(f"Fold {fold} has an empty normalization band")
        mean = sums[fold] / counts[fold]
        variance = sums_sq[fold] / counts[fold] - np.square(mean)
        std = np.sqrt(np.maximum(variance, 0.0))
        if np.any(~np.isfinite(std)) or np.any(std <= 0):
            raise RuntimeError(f"Fold {fold} produced invalid standard deviations")
        norm[f"Fold_{fold}"] = {
            "mean": mean.tolist(),
            "std": std.tolist(),
        }
    norm_tmp = building / "NORM_PARCEL_S2_set.json.tmp"
    norm_path = building / "NORM_PARCEL_S2_set.json"
    with norm_tmp.open("w", encoding="utf-8") as handle:
        json.dump(norm, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(norm_tmp, norm_path)

    shutil.copy2(source / "metadata.geojson", building / "metadata.geojson")

    # Final structural audit, including one deterministic sample from each fold.
    samples = {}
    for fold in range(1, 6):
        fold_rows = [row for row in rows if int(row["Fold"]) == fold]
        if not fold_rows:
            raise RuntimeError(f"Fold {fold} has no parcel samples")
        row = fold_rows[len(fold_rows) // 2]
        sample_path = data_out / f"S2_{row['ID_PARCEL']}.npy"
        sample = np.load(sample_path, mmap_mode="r")
        if sample.ndim != 3 or sample.shape[1] != 10 or sample.shape[2] < 1:
            raise RuntimeError(f"Invalid audit sample {sample_path}: {sample.shape}")
        samples[str(fold)] = {
            "parcel_id": int(row["ID_PARCEL"]),
            "shape": list(sample.shape),
            "sha256": file_sha256(sample_path),
        }

    report = {
        "source": str(source),
        "output": str(output),
        "patches": len(features),
        "patches_per_fold": patch_counts,
        "parcels": len(rows),
        "parcels_per_fold": {
            fold: sum(1 for row in rows if int(row["Fold"]) == fold)
            for fold in range(1, 6)
        },
        "parcels_per_fold_and_class": class_counts,
        "excluded_void_parcels": excluded_void,
        "non_pure_instances": non_pure_instances,
        "written_files": written_files,
        "reused_files": reused_files,
        "s2_scalar_observations": total_pixel_observations * 10,
        "metadata_sha256": file_sha256(metadata_path),
        "audit_samples": samples,
        "normalization": norm,
        "elapsed_seconds": time.time() - started,
    }
    report_tmp = building / "pixelset_build_report.json.tmp"
    report_path = building / "pixelset_build_report.json"
    with report_tmp.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(report_tmp, report_path)

    if not args.keep_building_dir:
        os.replace(building, output)
        report_path = output / report_path.name
    print(f"PixelSet complete: {len(rows)} parcels; report={report_path}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr, flush=True)
        raise
