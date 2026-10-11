#!/usr/bin/env python3
"""Convert the complete official MTLCC 240-grid fold-0 data to fast memmaps."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from numpy.lib.format import open_memmap
from tfrecord.reader import tfrecord_loader


SPLITS = {
    "train": "train_fold0.tileids",
    "val": "eval.tileids",
    "test": "test_fold0.tileids",
}
ARRAY_SPECS = {
    "x10": (np.int16, (46, 24, 24, 4)),
    "x20": (np.int16, (46, 12, 12, 6)),
    "doy": (np.int16, (46,)),
    "labels": (np.uint8, (24, 24)),
    "lengths": (np.uint8, ()),
}


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def normalize_id(value: str) -> str:
    name = Path(value.strip()).name
    if name.endswith(".tfrecord.gz"):
        return name[: -len(".tfrecord.gz")]
    return Path(name).stem


def decode_record(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.uint8]:
    record = next(iter(tfrecord_loader(str(path), None, compression_type="gzip")))

    def shaped(prefix: str, shape_key: str | None = None) -> np.ndarray:
        key = shape_key or prefix
        shape = tuple(int(value) for value in record[f"{key}/shape"])
        return np.frombuffer(record[f"{prefix}/data"], dtype=np.int64).reshape(shape)

    x10 = shaped("x10")
    x20 = shaped("x20")
    labels = shaped("labels")
    doy_shape = tuple(int(value) for value in record["dates/shape"])
    doy = np.frombuffer(record["dates/doy"], dtype=np.int64).reshape(doy_shape)
    if x10.shape != ARRAY_SPECS["x10"][1] or x20.shape != ARRAY_SPECS["x20"][1]:
        raise RuntimeError(f"Unexpected band shapes in {path}: x10={x10.shape}, x20={x20.shape}")
    if doy.shape != ARRAY_SPECS["doy"][1] or labels.shape != (46, 24, 24):
        raise RuntimeError(f"Unexpected date/label shapes in {path}: doy={doy.shape}, labels={labels.shape}")
    int16 = np.iinfo(np.int16)
    if x10.min() < int16.min or x20.min() < int16.min or x10.max() > int16.max or x20.max() > int16.max:
        raise RuntimeError(f"Reflectance outside int16 range in {path}")
    if doy.min() < int16.min or doy.max() > int16.max or labels.min() < -1 or labels.max() > 255:
        raise RuntimeError(f"Date or label outside cache range in {path}")
    valid = doy > 0  # Matches the official DeepSatModels conversion exactly.
    length = int(valid.sum())
    if length <= 0 or length > 46:
        raise RuntimeError(f"Invalid acquisition count {length} in {path}")
    valid_labels = labels[valid]
    if not np.array_equal(valid_labels, np.broadcast_to(valid_labels[0], valid_labels.shape)):
        raise RuntimeError(f"Semantic target changes over time in {path}")
    packed_x10 = np.full(ARRAY_SPECS["x10"][1], -1, dtype=np.int16)
    packed_x20 = np.full(ARRAY_SPECS["x20"][1], -1, dtype=np.int16)
    packed_doy = np.full(ARRAY_SPECS["doy"][1], -1, dtype=np.int16)
    packed_x10[:length] = x10[valid].astype(np.int16)
    packed_x20[:length] = x20[valid].astype(np.int16)
    packed_doy[:length] = doy[valid].astype(np.int16)
    return (
        packed_x10,
        packed_x20,
        packed_doy,
        valid_labels[0].astype(np.uint8),
        np.uint8(length),
    )


def convert_chunk(
    stage_dir: str,
    items: list[tuple[int, str]],
) -> dict[str, Any]:
    root = Path(stage_dir)
    arrays = {name: np.load(root / f"{name}.npy", mmap_mode="r+") for name in ARRAY_SPECS}
    sums = np.zeros(10, dtype=np.float64)
    squares = np.zeros(10, dtype=np.float64)
    counts = np.zeros(10, dtype=np.int64)
    label_counts = np.zeros(256, dtype=np.int64)
    length_min = 46
    length_max = 0
    for index, raw_path in items:
        x10, x20, doy, labels, length_value = decode_record(Path(raw_path))
        length = int(length_value)
        arrays["x10"][index] = x10
        arrays["x20"][index] = x20
        arrays["doy"][index] = doy
        arrays["labels"][index] = labels
        arrays["lengths"][index] = length_value
        x10_float = x10[:length].astype(np.float64)
        x20_float = x20[:length].astype(np.float64)
        sums[:4] += x10_float.sum(axis=(0, 1, 2))
        sums[4:] += x20_float.sum(axis=(0, 1, 2))
        squares[:4] += np.square(x10_float).sum(axis=(0, 1, 2))
        squares[4:] += np.square(x20_float).sum(axis=(0, 1, 2))
        counts[:4] += np.prod(x10_float.shape[:3], dtype=np.int64)
        counts[4:] += np.prod(x20_float.shape[:3], dtype=np.int64)
        label_counts += np.bincount(labels.reshape(-1), minlength=256)
        length_min = min(length_min, length)
        length_max = max(length_max, length)
    for array in arrays.values():
        array.flush()
    return {
        "samples": len(items),
        "sum": sums.tolist(),
        "square_sum": squares.tolist(),
        "pixel_count": counts.tolist(),
        "label_counts": label_counts.tolist(),
        "length_min": length_min,
        "length_max": length_max,
    }


def prepare_stage(
    raw_root: Path,
    output_root: Path,
    stage: str,
    workers: int,
    chunk_size: int,
) -> dict[str, Any]:
    split_path = raw_root / "tileids" / SPLITS[stage]
    listed = [normalize_id(line) for line in split_path.read_text().splitlines() if line.strip()]
    if len(set(listed)) != len(listed):
        raise RuntimeError(f"Duplicate sample IDs in {split_path}")
    data_dir = raw_root / "data16"
    present = [sample_id for sample_id in listed if (data_dir / f"{sample_id}.tfrecord.gz").is_file()]
    present_set = set(present)
    missing = [sample_id for sample_id in listed if sample_id not in present_set]
    if not present:
        raise RuntimeError(f"No records from {split_path} exist in {data_dir}")

    stage_dir = output_root / stage
    stage_dir.mkdir(parents=True, exist_ok=True)
    ids_path = stage_dir / "ids.txt"
    ids_text = "".join(f"{sample_id}\n" for sample_id in present)
    if ids_path.exists() and ids_path.read_text() != ids_text:
        raise RuntimeError(f"Existing cache IDs disagree with {split_path}")
    ids_path.write_text(ids_text)
    for name, (dtype, sample_shape) in ARRAY_SPECS.items():
        array_path = stage_dir / f"{name}.npy"
        expected_shape = (len(present), *sample_shape)
        if array_path.exists():
            array = np.load(array_path, mmap_mode="r")
            if array.shape != expected_shape or array.dtype != dtype:
                raise RuntimeError(f"Existing cache array disagrees: {array_path}")
        else:
            array = open_memmap(array_path, mode="w+", dtype=dtype, shape=expected_shape)
            array.flush()
        del array

    progress_path = stage_dir / "progress.json"
    progress = (
        json.loads(progress_path.read_text())
        if progress_path.exists()
        else {
            "completed_chunks": [],
            "samples": 0,
            "sum": [0.0] * 10,
            "square_sum": [0.0] * 10,
            "pixel_count": [0] * 10,
            "label_counts": [0] * 256,
            "length_min": 46,
            "length_max": 0,
        }
    )
    completed = {int(value) for value in progress["completed_chunks"]}
    chunks = [
        [
            (index, str(data_dir / f"{present[index]}.tfrecord.gz"))
            for index in range(start, min(len(present), start + chunk_size))
        ]
        for start in range(0, len(present), chunk_size)
    ]
    pending = [(chunk_index, chunk) for chunk_index, chunk in enumerate(chunks) if chunk_index not in completed]
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(convert_chunk, str(stage_dir), chunk): chunk_index
            for chunk_index, chunk in pending
        }
        for ordinal, future in enumerate(as_completed(futures), start=1):
            chunk_index = futures[future]
            result = future.result()
            progress["samples"] += int(result["samples"])
            for key in ("sum", "square_sum", "pixel_count", "label_counts"):
                progress[key] = (np.asarray(progress[key]) + np.asarray(result[key])).tolist()
            progress["completed_chunks"].append(chunk_index)
            progress["length_min"] = min(int(progress["length_min"]), int(result["length_min"]))
            progress["length_max"] = max(int(progress["length_max"]), int(result["length_max"]))
            progress["completed_chunks"].sort()
            atomic_json(progress_path, progress)
            if ordinal % 10 == 0 or ordinal == len(pending):
                print(
                    f"[{stage}] chunks={len(progress['completed_chunks'])}/{len(chunks)} "
                    f"samples={progress['samples']}/{len(present)}",
                    flush=True,
                )
    if set(progress["completed_chunks"]) != set(range(len(chunks))) or progress["samples"] != len(present):
        raise RuntimeError(f"Incomplete conversion for {stage}")
    metadata = {
        "stage": stage,
        "samples": len(present),
        "listed_samples": len(listed),
        "missing_samples": len(missing),
        "missing_ids": missing,
        "split_file": SPLITS[stage],
        "split_sha256": hashlib.sha256(split_path.read_bytes()).hexdigest(),
        "ids_sha256": hashlib.sha256(ids_text.encode()).hexdigest(),
        "sequence_length_min": int(progress["length_min"]),
        "sequence_length_max": int(progress["length_max"]),
        "tile_size": 24,
        "raw_label_counts": progress["label_counts"],
        "sum": progress["sum"],
        "square_sum": progress["square_sum"],
        "pixel_count": progress["pixel_count"],
    }
    atomic_json(stage_dir / "metadata.json", metadata)
    return metadata


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--chunk-size", type=int, default=128)
    args = parser.parse_args()
    raw_root = args.raw_root.resolve()
    output_root = args.output_root.resolve()
    if not (raw_root / "data16").is_dir() or not (raw_root / "tileids").is_dir():
        raise RuntimeError(f"Not an official MTLCC full/240 root: {raw_root}")
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.json"
    atomic_json(
        manifest_path,
        {
            "format": "mtlcc_memmap_v1",
            "complete": False,
            "raw_root": str(raw_root),
            "year": 2016,
            "resolution_grid": 240,
            "bands": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"],
            "split_files": SPLITS,
        },
    )
    metadata = {
        stage: prepare_stage(raw_root, output_root, stage, args.workers, args.chunk_size)
        for stage in ("train", "val", "test")
    }
    train = metadata["train"]
    sums = np.asarray(train["sum"], dtype=np.float64)
    squares = np.asarray(train["square_sum"], dtype=np.float64)
    counts = np.asarray(train["pixel_count"], dtype=np.float64)
    mean = sums / counts
    variance = np.maximum(0.0, squares / counts - np.square(mean))
    stats = {
        "mean": mean.tolist(),
        "std": np.sqrt(variance).tolist(),
        "pixel_count": counts.astype(np.int64).tolist(),
        "source": "official MTLCC full/240 data16 train_fold0 only",
        "band_order": ["B02", "B03", "B04", "B08", "B05", "B06", "B07", "B8A", "B11", "B12"],
        "train_ids_sha256": train["ids_sha256"],
    }
    atomic_json(output_root / "stats_10band.json", stats)
    atomic_json(
        manifest_path,
        {
            "format": "mtlcc_memmap_v1",
            "complete": True,
            "raw_root": str(raw_root),
            "year": 2016,
            "resolution_grid": 240,
            "bands": stats["band_order"],
            "split_files": SPLITS,
            "stages": {stage: metadata[stage]["samples"] for stage in metadata},
            "stats_file": "stats_10band.json",
        },
    )
    print(json.dumps({"complete": True, "stages": {s: metadata[s]["samples"] for s in metadata}}), flush=True)


if __name__ == "__main__":
    main()
