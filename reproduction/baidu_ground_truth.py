"""Recover author-order Baidu evaluation from locally cached official poses.

This module performs no network access and reads no image pixels.  Raw XYZ poses
define positives; rounded XY coordinates are used solely for the fully audited
image-name mapping.  The second, incompatible local image group is excluded.
"""

from __future__ import annotations

import binascii
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _simple_images(directory: Path) -> tuple[dict[tuple[float, float], list[Path]], Counter]:
    by_xy: dict[tuple[float, float], list[Path]] = defaultdict(list)
    counts: Counter = Counter()
    for path in sorted(directory.glob("*.jpg")):
        fields = path.name.split("@")
        if len(fields) == 16 and all(not value for value in fields[3:-1]):
            try:
                key = (float(fields[1]), float(fields[2]))
            except ValueError:
                counts["invalid_simple_coordinates"] += 1
                continue
            by_xy[key].append(path.resolve())
            counts["simple"] += 1
        elif len(fields) == 16 and fields[3] == "_":
            counts["excluded_extended"] += 1
        else:
            counts["excluded_unrecognized"] += 1
    return dict(by_xy), counts


def build_baidu_author_manifest(repo_root: Path, metadata_root: Path) -> dict[str, Any]:
    """Return a ready/blocked manifest after integrity and bijection checks.

    Comparability remains conditional: poses and author image IDs are verified,
    but the local postprocessed image pixels and unpublished Baidu-specific
    filename formatter have not been independently reproduced.
    """
    from scipy.spatial import cKDTree

    repo_root = Path(repo_root).resolve()
    metadata_root = Path(metadata_root).resolve()
    dataset_root = repo_root / "datasets/baidu"
    protocol: dict[str, Any] = {
        "name": "VLAD-BuFF Baidu author XYZ ground truth with audited local image mapping",
        "source": "dataloaders/val/BaiduDataset.py; author NPY arrays; original IDL_dataset_cvpr17_3852.zip camera metadata",
        "positive_definition": "Euclidean distance between original unrounded camera XYZ coordinates <= 10m, with no angular filtering.",
        "comparability": "conditional",
        "notes": [
            "References and queries preserve the exact author NPY order and original image IDs.",
            "Only the XY-only 689-reference/2292-query group is eligible; the extended coordinate group is excluded.",
            "Three-decimal XY matching is used only to map image names. Every author camera must map uniquely, with no reused or leftover simple image.",
            "The three-decimal name rule is verified empirically on the entire dataset; a Baidu-specific official formatter has not been found.",
            "Local image pixel identity and preprocessing equivalence to the author images are unverified; scores must be labeled conditional.",
            "Queries with no 10m positive remain in the author query list and recall denominator.",
        ],
    }
    audit: dict[str, Any] = {
        "expected_references": 689,
        "expected_queries": 2292,
        "missing_references": 689,
        "missing_queries": 2292,
        "metadata_root": str(metadata_root),
        "mapping_rule": "numeric filename XY == camera XY rounded to 3 decimal places; exact, role-specific, one-to-one matching",
        "official_baidu_formatter_verified": False,
        "pixel_identity_verified": False,
    }
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "dataset": "baidu",
        "status": "blocked",
        "references": [],
        "queries": [],
        "positives": None,
        "protocol": protocol,
        "audit": audit,
    }
    errors: list[str] = []
    required = ("provenance.json", "camera_index.json", "zip_index.json")
    missing_metadata = [name for name in required if not (metadata_root / name).is_file()]
    if missing_metadata:
        audit["blocked_reasons"] = [f"Missing metadata: {name}" for name in missing_metadata]
        return manifest
    provenance = _json(metadata_root / "provenance.json")
    camera_index = {item["name"]: item for item in _json(metadata_root / "camera_index.json")}
    archive_images = {item["name"] for item in _json(metadata_root / "zip_index.json")}
    audit["provenance_sha256"] = _sha256(metadata_root / "provenance.json")
    audit["source_metadata_sha256"] = dict(provenance.get("source_hashes", {}))
    protocol["source_metadata"] = str(metadata_root / "provenance.json")
    protocol["source_archive_url"] = provenance.get("source_archive_landing_url")
    for filename, expected_hash in provenance.get("source_hashes", {}).items():
        path = metadata_root / filename
        if not path.is_file() or _sha256(path) != expected_hash:
            errors.append(f"Metadata hash mismatch: {filename}")
    if len(camera_index) != 2981:
        errors.append(f"Expected 2981 indexed cameras, got {len(camera_index)}")

    coordinates: dict[str, np.ndarray] = {}
    for role, image_npy, camera_npy, original_group, native_group in (
        ("references", "baidu_dbImages.npy", "baidu_dbgImages_gt.npy", "training_gt", "database"),
        ("queries", "baidu_qImages.npy", "baidu_qImages_gt.npy", "query_gt", "queries"),
    ):
        ids = np.load(dataset_root / image_npy, allow_pickle=False)
        camera_ids = np.load(dataset_root / camera_npy, allow_pickle=False)
        audit[f"expected_{role}"] = len(ids)
        if len(ids) != len(camera_ids):
            errors.append(f"{role}: image and camera ID counts differ")
            continue
        for filename in (image_npy, camera_npy):
            relative = "datasets/baidu/" + filename
            expected_hash = provenance.get("author_metadata_sha256", {}).get(relative)
            actual_hash = _sha256(dataset_root / filename)
            if not expected_hash or actual_hash != expected_hash:
                errors.append(f"Author NPY hash mismatch: {filename}")
        by_xy, groups = _simple_images(dataset_root / "images/test" / native_group)
        records: list[dict[str, Any]] = []
        proof: list[dict[str, Any]] = []
        xyz_rows: list[np.ndarray] = []
        used: Counter[str] = Counter()
        missing_cameras: list[str] = []
        invalid_cameras: list[str] = []
        unmatched: list[str] = []
        ambiguous: list[str] = []
        original_names_missing: list[str] = []
        max_quantization_error = 0.0
        for image_id, camera_id in zip(ids, camera_ids):
            image_id, camera_id = str(image_id), str(camera_id)
            if Path(image_id).stem != Path(camera_id).stem:
                errors.append(f"{role}: author image/camera stems disagree: {image_id}")
                continue
            archive_image_group = "training_images_undistort" if role == "references" else "query_images_undistort"
            if archive_image_group + "/" + Path(image_id).name not in archive_images:
                original_names_missing.append(image_id)
            source_name = original_group + "/" + Path(camera_id).name
            camera_path = metadata_root / "official_camera" / source_name
            entry = camera_index.get(source_name)
            if entry is None or not camera_path.is_file():
                missing_cameras.append(camera_id)
                continue
            payload = camera_path.read_bytes()
            camera_hash = hashlib.sha256(payload).hexdigest()
            if (
                camera_hash != entry.get("sha256")
                or (binascii.crc32(payload) & 0xFFFFFFFF) != entry.get("crc")
                or len(payload) != entry.get("uncompressed_size")
            ):
                invalid_cameras.append(camera_id)
                continue
            try:
                xyz = np.fromstring(payload.decode("utf-8").splitlines()[-2], sep=" ")
            except (UnicodeError, IndexError, ValueError):
                invalid_cameras.append(camera_id)
                continue
            if xyz.shape != (3,) or not np.isfinite(xyz).all():
                invalid_cameras.append(camera_id)
                continue
            key = tuple(float(f"{float(value):.3f}") for value in xyz[:2])
            candidates = by_xy.get(key, [])
            if len(candidates) != 1:
                (unmatched if not candidates else ambiguous).append(image_id)
                continue
            image_path = candidates[0]
            used[str(image_path)] += 1
            max_quantization_error = max(max_quantization_error, float(np.max(np.abs(xyz[:2] - np.asarray(key)))))
            records.append({"id": image_id, "path": str(image_path)})
            xyz_rows.append(xyz)
            proof.append({
                "id": image_id,
                "camera_id": camera_id,
                "camera_path": str(camera_path),
                "camera_sha256": camera_hash,
                "camera_crc32": entry["crc"],
                "xyz": xyz.tolist(),
                "rounded_xy": list(key),
                "native_image": str(image_path),
            })
        duplicate_mapping = sum(count - 1 for count in used.values())
        unmapped_simple = groups["simple"] - len(used)
        role_audit = {
            "expected_images": len(ids),
            "mapped_images": len(records),
            "missing_cameras": len(missing_cameras),
            "invalid_camera_integrity_or_xyz": len(invalid_cameras),
            "missing_original_image_names_in_official_zip": len(original_names_missing),
            "unmatched_images": len(unmatched),
            "ambiguous_images": len(ambiguous),
            "duplicate_mapped_paths": duplicate_mapping,
            "unmapped_simple_images": unmapped_simple,
            "native_filename_groups": dict(groups),
            "max_abs_xy_quantization_error": max_quantization_error,
            "mapping_proof": proof,
        }
        audit[role + "_mapping"] = role_audit
        audit["missing_" + role] = len(ids) - len(records)
        manifest[role] = records
        coordinates[role] = np.asarray(xyz_rows, dtype=np.float64)
        if (
            missing_cameras or invalid_cameras or original_names_missing or unmatched or ambiguous
            or duplicate_mapping or unmapped_simple or len(records) != len(ids)
        ):
            errors.append(f"{role}: camera integrity, author-name coverage or image bijection is incomplete")

    if errors:
        audit["blocked_reasons"] = errors
        return manifest
    positives = [
        sorted(map(int, row))
        for row in cKDTree(coordinates["references"]).query_ball_point(coordinates["queries"], 10.0, workers=1)
    ]
    manifest["positives"] = positives
    audit.update(
        queries_without_positives=sum(not row for row in positives),
        positive_edges=sum(map(len, positives)),
        camera_files_crc_and_sha256_verified=len(camera_index),
        author_metadata_sha256=dict(provenance.get("author_metadata_sha256", {})),
        recall_denominator=len(manifest["queries"]),
    )
    manifest["status"] = "ready"
    return manifest
