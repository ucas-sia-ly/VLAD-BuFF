"""Build small, explicit evaluation views without changing source datasets.

The author-supplied NPY arrays define image order and positives.  Native image
names are used only to locate those images; they never redefine ground truth.
This module imports neither torch nor any GPU library.
"""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np


def _load(root: Path, name: str) -> np.ndarray:
    return np.load(root / name, allow_pickle=True)


def _records(ids: np.ndarray, paths: list[Path]) -> list[dict[str, str]]:
    return [
        {"id": str(identifier), "path": str(path.resolve())}
        for identifier, path in zip(ids, paths)
    ]


def _radius_positives(db: np.ndarray, queries: np.ndarray) -> list[list[int]]:
    # Only small coordinate arrays enter RAM.  A single CPU worker is sufficient.
    from scipy.spatial import cKDTree

    def xy(names: np.ndarray) -> np.ndarray:
        return np.asarray(
            [[float(str(name).split("@")[i]) for i in (1, 2)] for name in names],
            dtype=np.float64,
        )

    return [
        sorted(map(int, row))
        for row in cKDTree(xy(db)).query_ball_point(xy(queries), 25.0, workers=1)
    ]


def _manifest(
    name: str,
    ids_db: np.ndarray,
    ids_q: np.ndarray,
    paths_db: list[Path],
    paths_q: list[Path],
    positives: list[list[int]] | None,
    protocol: dict[str, Any],
    extra_audit: dict[str, Any] | None = None,
    blocked: bool = False,
) -> dict[str, Any]:
    missing_db = [str(i) for i, p in zip(ids_db, paths_db) if not p.is_file()]
    missing_q = [str(i) for i, p in zip(ids_q, paths_q) if not p.is_file()]
    audit: dict[str, Any] = {
        "expected_references": len(ids_db),
        "expected_queries": len(ids_q),
        "missing_references": len(missing_db),
        "missing_queries": len(missing_q),
        "missing_reference_examples": missing_db[:5],
        "missing_query_examples": missing_q[:5],
    }
    if positives is not None:
        if len(positives) != len(ids_q):
            raise ValueError(f"{name}: positive list does not match query count")
        if any(i < 0 or i >= len(ids_db) for row in positives for i in row):
            raise ValueError(f"{name}: positive index is outside the author database")
        audit.update(
            queries_without_positives=sum(not row for row in positives),
            positive_edges=sum(map(len, positives)),
        )
    if extra_audit:
        audit.update(extra_audit)
    return {
        "schema_version": 1,
        "dataset": name,
        "status": "blocked" if blocked or missing_db or missing_q or positives is None else "ready",
        "references": _records(ids_db, paths_db),
        "queries": _records(ids_q, paths_q),
        "positives": positives,
        "protocol": protocol,
        "audit": audit,
    }


def _protocol(name: str, definition: str, comparability: str = "author_protocol") -> dict[str, Any]:
    return {
        "name": f"VLAD-BuFF {name} author metadata",
        "source": f"dataloaders/val/{name}Dataset.py and datasets/{name}/*.npy",
        "positive_definition": definition,
        "comparability": comparability,
        "notes": [],
    }


def _frame_index(directory: Path) -> tuple[dict[int, Path], list[int], int]:
    by_id: dict[int, Path] = {}
    duplicate_ids: list[int] = []
    invalid = 0
    for path in sorted(directory.glob("*.jpg")):
        try:
            # Downloader format: @x@y@zone@band@lat@lon@frame_id@...
            frame_id = int(path.name.split("@")[7])
        except (IndexError, ValueError):
            invalid += 1
            continue
        if frame_id in by_id:
            duplicate_ids.append(frame_id)
        else:
            by_id[frame_id] = path
    return by_id, duplicate_ids, invalid


def _nordland(root: Path) -> dict[str, Any]:
    db = _load(root, "Nordland_dbImages.npy")
    queries = _load(root, "Nordland_qImages.npy")
    positives = [list(map(int, row)) for row in _load(root, "Nordland_gt.npy")]
    protocol = _protocol(
        "Nordland", "Unmodified Nordland_gt.npy: author query i uses reference frame 10*i +/- 1.",
        "conditional",
    )
    protocol["notes"] = [
        "The native downloader data describe the Patch-NetVLAD split; image identity with the VPR-Bench release is unverified.",
        "Paper supplementary section 10 uses winter references and summer queries. Local downloader roles are reversed, so native queries supply author references and native database supplies author queries.",
        "Season direction was checked visually at native frame IDs 0, 5000, 15000 and 25000: native database is summer and native queries is winter.",
        "Only author queries are evaluated: author query stem i maps to native frame ID 10*i, not lexicographic filename order.",
        "The native README's full-query +/-10-frame protocol is not used.",
        "Frame ID is the eighth @-separated field (Python split('@')[7]).",
    ]
    protocol["source_roles"] = {
        "references": "images/test/queries (winter)",
        "queries": "images/test/database (summer, every tenth frame)",
    }
    db_index, db_duplicates, db_invalid = _frame_index(root / "images/test/queries")
    q_index, q_duplicates, q_invalid = _frame_index(root / "images/test/database")

    def locate(name: str, query: bool) -> Path:
        original = root / name
        if original.is_file():
            return original
        frame_id = int(Path(name).stem) * (10 if query else 1)
        index = q_index if query else db_index
        # A missing path is retained for an honest, blocked manifest.
        return index.get(frame_id, root / "__missing__" / name)

    return _manifest(
        "Nordland", db, queries,
        [locate(str(x), False) for x in db],
        [locate(str(x), True) for x in queries], positives, protocol,
        {
            "native_references": len(db_index),
            "native_queries": len(q_index),
            "native_queries_not_selected": len(set(q_index) - {int(Path(str(x)).stem) * 10 for x in queries}),
            "duplicate_reference_frame_ids": db_duplicates,
            "duplicate_query_frame_ids": q_duplicates,
            "invalid_reference_filenames": db_invalid,
            "invalid_query_filenames": q_invalid,
            "source_identity_verified": False,
            "native_roles_swapped_to_match_paper_seasons": True,
            "season_evidence_frame_ids": [0, 5000, 15000, 25000],
        },
        blocked=bool(db_duplicates or q_duplicates or db_invalid or q_invalid),
    )


def _sped(root: Path) -> dict[str, Any]:
    db = _load(root, "SPED_dbImages.npy")
    queries = _load(root, "SPED_qImages.npy")
    positives = [list(map(int, row)) for row in _load(root, "SPED_gt.npy")]
    protocol = _protocol("SPED", "Unmodified SPED_gt.npy, one paired reference per author query.")
    extra: dict[str, Any] = {}
    original_gt = root / "ground_truth_new.npy"
    if original_gt.is_file():
        native = _load(root, original_gt.name)
        equal = len(native) == len(queries) and all(
            int(row[0]) == i and list(map(int, row[1])) == positives[i]
            for i, row in enumerate(native)
        )
        extra["native_ground_truth_matches_author"] = equal
        if not equal:
            protocol["notes"].append("Native ground_truth_new.npy disagrees with author GT; review is required.")
    return _manifest(
        "SPED", db, queries, [root / str(x) for x in db],
        [root / str(x) for x in queries], positives, protocol, extra,
        blocked=extra.get("native_ground_truth_matches_author") is False,
    )


def _st_lucia(root: Path) -> dict[str, Any]:
    db = _load(root, "st_lucia_dbImages.npy")
    queries = _load(root, "st_lucia_qImages.npy")
    protocol = _protocol("st_lucia", "Euclidean distance in author filename UTM XY coordinates <= 25m.")
    protocol["source"] = "dataloaders/val/StluciaDataset.py and datasets/st_lucia/*.npy"
    return _manifest(
        "st_lucia", db, queries,
        [root / "images/test" / str(x) for x in db],
        [root / "images/test" / str(x) for x in queries],
        _radius_positives(db, queries), protocol,
    )


def _amstertime(root: Path, view_root: Path) -> dict[str, Any]:
    db = _load(root, "amstertime_dbImages.npy")
    queries = _load(root, "amstertime_qImages.npy")
    protocol = _protocol(
        "amstertime", "Author dummy-UTM XY radius <= 25m; matching historical/modern pair has the same ID.",
    )
    protocol["source"] = "dataloaders/val/AmstertimeDataset.py; author NPY arrays; image_pairs.csv; VPR-datasets-downloader/download_amstertime.py:43-55"
    protocol["notes"] = [
        "Modern PNGs are converted to RGB and saved as JPEG with Pillow defaults, matching the official download_amstertime.py call.",
        "Historical JPEGs are read directly; source files are never moved, renamed, overwritten, or deleted.",
        "Dummy coordinates encode image pairs, not measured geographic positions.",
        "Image encoding follows the official conversion; different Pillow/JPEG library versions can still differ in bytes.",
    ]
    db_ids = [str(x).split("@")[7] for x in db]
    q_ids = [str(x).split("@")[7] for x in queries]
    source_db = [root / "new" / f"{identifier}.png" for identifier in db_ids]
    source_q = [root / "old" / f"{identifier}.jpg" for identifier in q_ids]
    extra: dict[str, Any] = {
        "missing_source_references": sum(not p.is_file() for p in source_db),
        "missing_source_queries": sum(not p.is_file() for p in source_q),
        "modern_conversion": "PIL.Image.open(source).convert('RGB').save(target_with_jpg_suffix)",
    }
    csv_path = root / "image_pairs.csv"
    csv_valid = False
    if csv_path.is_file():
        with csv_path.open(newline="", encoding="utf-8") as stream:
            rows = list(csv.DictReader(stream))
        pairs = [(f"{int(r['new_num']):04d}", f"{int(r['old_num']):04d}") for r in rows]
        csv_valid = (
            len(pairs) == len(db_ids)
            and len(set(pairs)) == len(pairs)
            and all(a == b for a, b in pairs)
            and {a for a, _ in pairs} == set(db_ids) == set(q_ids)
        )
        extra["native_pair_csv_rows"] = len(rows)
    extra["native_pair_csv_matches_author"] = csv_valid
    blocked = bool(extra["missing_source_references"] or extra["missing_source_queries"] or not csv_valid)
    target_db = [view_root / "amstertime" / str(x) for x in db]
    if not blocked:
        from PIL import Image

        for source, target in zip(source_db, target_db):
            # The sidecar prevents a stale output from surviving a source change.
            state = {"source": str(source.resolve()), "size": source.stat().st_size,
                     "mtime_ns": source.stat().st_mtime_ns, "conversion": "RGB/default-JPEG/v1"}
            sidecar = target.with_suffix(target.suffix + ".source.json")
            cached = False
            if target.is_file() and sidecar.is_file():
                try:
                    cached = json.loads(sidecar.read_text(encoding="utf-8")) == state
                except (ValueError, OSError):
                    pass
            if cached:
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_suffix(".tmp.jpg")
            try:
                with Image.open(source) as image:
                    converted = image.convert("RGB")
                    try:
                        converted.save(temporary)
                    finally:
                        converted.close()
                temporary.replace(target)
                sidecar.write_text(json.dumps(state, sort_keys=True) + "\n", encoding="utf-8")
            finally:
                temporary.unlink(missing_ok=True)
    else:
        protocol["notes"].append("Missing source images or inconsistent image_pairs.csv prevent conversion and evaluation.")
    return _manifest(
        "amstertime", db, queries, target_db, source_q,
        _radius_positives(db, queries), protocol, extra, blocked=blocked,
    )


def _baidu(root: Path) -> dict[str, Any]:
    db = _load(root, "baidu_dbImages.npy")
    queries = _load(root, "baidu_qImages.npy")
    groups: dict[str, dict[str, int]] = {}
    for role in ("database", "queries"):
        counts: Counter[str] = Counter()
        for path in (root / "images/test" / role).glob("*.jpg"):
            parts = path.name.split("@")
            if len(parts) == 16 and all(not x for x in parts[3:-1]):
                counts["xy_only"] += 1
            elif len(parts) == 16 and parts[3] == "_":
                counts["extended_other_coordinate_system"] += 1
            else:
                counts["unrecognized"] += 1
        groups[role] = dict(counts)
    protocol = _protocol(
        "baidu", "Author loader requires .camera XYZ Euclidean distance <= 10m; angular filtering is disabled.",
        "unavailable",
    )
    protocol["source"] = "dataloaders/val/BaiduDataset.py and datasets/baidu/*Images*.npy"
    protocol["notes"] = [
        "The local directory mixes an XY-only 689-reference/2292-query set with a second, incompatible coordinate system.",
        "Original basenames and .camera files are absent; equal image counts do not establish the author image mapping.",
        "XY-only radius truth is refused: it cannot reproduce the author's XYZ distance criterion.",
        "Recover the author-matching images and camera poses before reporting a comparable Baidu score.",
    ]
    result = _manifest(
        "baidu", db, queries, [root / "images/test" / str(x) for x in db],
        [root / "images/test" / str(x) for x in queries], None, protocol,
        {"native_filename_groups": groups, "native_camera_files": sum(1 for _ in root.rglob("*.camera")),
         "blocked_reason": "Author image-ID mapping and 3D camera ground truth are unavailable."},
        blocked=True,
    )
    # There are no verified author-to-native image correspondences to expose.
    result["references"] = []
    result["queries"] = []
    return result


def build_native(name: str, repo_root: Path, view_root: Path) -> dict[str, Any] | None:
    """Adapt one supported local layout, returning None for unsupported names.

    ``view_root`` is used only for the official AmsterTime PNG-to-JPEG conversion.
    Missing data and unavailable truth return a blocked manifest, never guessed
    positives.  NPY metadata and source images are always read-only.
    """
    root = Path(repo_root).resolve() / "datasets" / name
    if name == "Nordland":
        metadata_root = Path(repo_root).resolve() / "logs/nordland_official_protocol"
        if (metadata_root / "provenance.json").is_file():
            from reproduction.nordland_official import build_nordland_official_manifest
            return build_nordland_official_manifest(repo_root, metadata_root)
        return _nordland(root)
    if name == "SPED":
        return _sped(root)
    if name == "st_lucia":
        return _st_lucia(root)
    if name == "amstertime":
        return _amstertime(root, Path(view_root).resolve())
    if name == "baidu":
        metadata_root = Path(repo_root).resolve() / "logs/baidu_protocol_metadata"
        if (metadata_root / "provenance.json").is_file():
            from reproduction.baidu_ground_truth import build_baidu_author_manifest
            return build_baidu_author_manifest(repo_root, metadata_root)
        return _baidu(root)
    return None
