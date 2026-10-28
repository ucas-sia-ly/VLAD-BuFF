"""Map supplied retrieval releases to the author's fixed NPY protocols.

This module does no model work and never changes a source image or NPY file.
Dataset IDs and ordering come from the repository's original metadata, except
MSLS test with public GT, which is verified against the official current loader.
"""
from __future__ import annotations

from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import hashlib
import json
import os
from pathlib import Path

import numpy as np


_SIZES = {
    "MSLS": (18871, 740),
    "MSLS_Test": (36565, 1135),
    "pitts30k_val": (10000, 7608),
    "pitts30k_test": (10000, 6816),
    "pitts250k_test": (83952, 8280),
    "sfsm": (27191, 1000),
    "tokyo247": (75984, 315),
}


def _rows(names, paths):
    return [{"id": str(name), "path": str(Path(path).resolve())}
            for name, path in zip(names, paths)]


def _radius_positives(database, queries):
    # Deliberately use the same rounded UTM values and 25 m radius as the
    # author's loaders, rather than replacing them with raw-release metadata.
    from sklearn.neighbors import NearestNeighbors

    def coordinates(names):
        xy = np.asarray([str(name).split("@")[1:3] for name in names], dtype=float)
        if xy.shape != (len(names), 2) or not np.isfinite(xy).all():
            raise ValueError("Invalid author NPY UTM coordinates")
        return xy

    neighbors = NearestNeighbors(n_jobs=1).fit(coordinates(database))
    return [list(map(int, indices)) for indices in neighbors.radius_neighbors(
        coordinates(queries), radius=25, return_distance=False)]


def _tokyo_sources(root, database, queries):
    candidates = defaultdict(list)
    for path in (root / "database_gsv_vga").glob("*/*/*.png"):
        pano, mode, angle_text = path.stem.rsplit("_", 2)
        angle = int(angle_text)
        if mode != "012" or angle not in range(0, 360, 30):
            raise ValueError(f"Unexpected Tokyo panorama view: {path}")
        candidates[(pano, angle // 30)].append(path)
    paths, unresolved, ambiguous = [], [], []
    for value in database:
        parts = str(value).split("@")
        matches = candidates.get((parts[7], int(parts[8])), [])
        if len(matches) == 1:
            paths.append(matches[0])
        else:
            paths.append(root / str(value))
            (unresolved if not matches else ambiguous).append(str(value))
    names = [str(value).split("@")[7] for value in queries]
    if len(set(names)) != len(names):
        raise ValueError("Duplicate Tokyo query image IDs")
    return paths, [root / "queries" / value for value in names], {
        "raw_database_png_count": sum(map(len, candidates.values())),
        "raw_query_jpeg_count": sum(1 for _ in (root / "queries").glob("*.jpg")),
        "unresolved_reference_ids": unresolved,
        "ambiguous_reference_ids": ambiguous,
    }


def _verify_tokyo_primary(repo_root, database, queries, positives):
    """Cross-check identity sets and GT, accounting for different index orders."""
    primary = repo_root / "datasets/Pittsburgh/Pittsburgh/_dataset_specs/datasets/tokyo247.mat"
    if not primary.is_file():
        return {"official_mat_crosscheck": "not_available"}
    from scipy.io import loadmat
    from sklearn.neighbors import NearestNeighbors

    struct = loadmat(primary, squeeze_me=True, struct_as_record=False)["dbStruct"]
    db_keys = [(str(x).split("@")[7], int(str(x).split("@")[8])) for x in database]
    primary_keys = [(Path(str(x)).stem.rsplit("_", 2)[0],
                     int(Path(str(x)).stem.rsplit("_", 2)[2]) // 30)
                    for x in np.atleast_1d(struct.dbImageFns)]
    query_names = [str(x).split("@")[7] for x in queries]
    primary_queries = list(map(str, np.atleast_1d(struct.qImageFns)))
    if (set(db_keys) != set(primary_keys) or set(query_names) != set(primary_queries)
            or len(set(db_keys)) != len(db_keys) or float(struct.posDistThr) != 25):
        raise ValueError("Tokyo author NPY identities differ from the local official MAT")
    db_indices = {key: i for i, key in enumerate(primary_keys)}
    q_indices = {key: i for i, key in enumerate(primary_queries)}
    primary_db_xy = struct.utmDb.T[[db_indices[key] for key in db_keys]]
    primary_q_xy = struct.utmQ.T[[q_indices[key] for key in query_names]]
    primary_gt = NearestNeighbors(n_jobs=1).fit(primary_db_xy).radius_neighbors(
        primary_q_xy, radius=25, return_distance=False)
    differences = sum(set(map(int, original)) != set(map(int, checked))
                      for original, checked in zip(positives, primary_gt))
    if differences:
        raise ValueError(f"Tokyo NPY versus official MAT positive sets differ for {differences} queries")
    return {"official_mat_crosscheck": "identity_sets_and_all_positive_sets_match",
            "official_mat_path": str(primary),
            "official_mat_sha256": hashlib.sha256(primary.read_bytes()).hexdigest(),
            "official_mat_positive_set_differences": differences}


def materialize_retrieval_views(manifest: dict, workers: int = 4) -> None:
    """Execute the official formatter's image operations in a separate view.

    This is CPU preprocessing, not evaluation or training. It uses Pillow's
    default JPEG save settings and torchvision's scalar Resize(480), exactly
    as the published formatter does. Sources are never overwritten. A recipe
    signature permits resuming interrupted conversion and rejects stale views.
    """
    items = [row for key in ("references", "queries") for row in manifest[key]
             if "source_path" in row]
    if not items:
        return
    from PIL import Image, __version__ as pillow_version
    import torchvision
    from torchvision.transforms import Resize

    if not 1 <= workers <= 4:
        raise ValueError("Use between 1 and 4 CPU image conversion workers")
    audit = manifest["audit"]
    if audit.get("unresolved_reference_ids") or audit.get("ambiguous_reference_ids"):
        raise ValueError("Tokyo reference identity mapping is incomplete or ambiguous")
    target_root = Path(audit["derived_image_root"])
    sources, targets = [Path(row["source_path"]) for row in items], [Path(row["path"]) for row in items]
    if any(source.resolve() == target.resolve() for source, target in zip(sources, targets)):
        raise ValueError("Derived images must not overwrite source images")
    signature = hashlib.sha256()
    for row, source in zip(items, sources):
        stat = source.stat()
        signature.update(f"{source}\0{stat.st_size}\0{stat.st_mtime_ns}\0".encode())
        signature.update(json.dumps(row["preprocessing"], sort_keys=True).encode())
    recipe = {
        "recipe": "official-format_tokyo247-default-jpeg-v1",
        "source_inventory_sha256": signature.hexdigest(),
        "pillow": pillow_version,
        "torchvision": torchvision.__version__,
        "database": "PIL.Image.open(source_png).save(target_jpg), default JPEG parameters",
        "query": "torchvision.transforms.Resize(480)(PIL.Image.open(source)).save(target_jpg)",
        "references": len(manifest["references"]), "queries": len(manifest["queries"]),
    }
    marker = target_root / "format-recipe.json"
    try:
        previous = json.loads(marker.read_text()) if marker.is_file() else None
    except (OSError, ValueError):
        previous = None
    resume = previous is not None and all(previous.get(k) == v for k, v in recipe.items())
    target_root.mkdir(parents=True, exist_ok=True)
    for directory in {target.parent for target in targets}:
        directory.mkdir(parents=True, exist_ok=True)

    def save_marker(status):
        temporary = marker.with_suffix(".tmp")
        temporary.write_text(json.dumps({**recipe, "status": status}, indent=2) + "\n")
        temporary.replace(marker)

    save_marker("preparing")
    resize = Resize(480)

    def convert(row):
        source, target = Path(row["source_path"]), Path(row["path"])
        if resume and target.is_file():
            try:
                with Image.open(target) as existing:
                    if existing.format != "JPEG":
                        raise ValueError("Expected derived JPEG")
                    existing.verify()
                return
            except (OSError, ValueError):
                pass
        temporary = target.with_suffix(f".tmp-{os.getpid()}.jpg")
        try:
            with Image.open(source) as image:
                # The official code does not add a color-mode conversion.
                if row["preprocessing"].get("resize_short_edge") == 480:
                    image = resize(image)
                image.save(temporary)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        # At most twice the worker count is queued; source images are decoded
        # inside workers and never accumulated in a producer-side image list.
        iterator, pending = iter(items), set()
        while True:
            while len(pending) < workers * 2:
                row = next(iterator, None)
                if row is None:
                    break
                pending.add(pool.submit(convert, row))
            if not pending:
                break
            done, pending = wait(pending, return_when=FIRST_COMPLETED)
            for future in done:
                future.result()
    save_marker("complete")
    audit["format_recipe"] = {**recipe, "status": "complete"}
    audit["preparation_complete"] = True
    audit["pending_derivations"] = 0
    manifest["protocol"]["formatting_versions"] = {
        "pillow": pillow_version, "torchvision": torchvision.__version__}
    for key in ("references", "queries"):
        missing = [row["id"] for row in manifest[key] if not Path(row["path"]).is_file()]
        audit[f"missing_{key}"] = len(missing)
        audit[f"missing_{key}_examples"] = missing[:10]
    if not audit["missing_references"] and not audit["missing_queries"]:
        manifest["status"] = "ready"


def build_retrieval(name: str, repo_root: Path, view_root: Path) -> dict | None:
    """Build one manifest, or return None for another adapter's dataset.

    Tokyo's image paths point to the separate author-format view. Call
    :func:`materialize_retrieval_views` to prepare or resume that CPU conversion.
    """
    if name not in _SIZES:
        return None
    repo_root, view_root = Path(repo_root).resolve(), Path(view_root).resolve()
    if name == 'MSLS_Test':
        from reproduction.msls_test_protocol import metadata_archive, build_msls_test_manifest
        if metadata_archive(repo_root) is not None:
            return build_msls_test_manifest(repo_root)
    expected_refs, expected_queries = _SIZES[name]
    manifest = {
        "schema_version": 1,
        "dataset": name,
        "status": "blocked",
        "references": [],
        "queries": [],
        "positives": None,
        "protocol": {
            "name": f"VLAD-BuFF author metadata: {name}",
            "source": [],
            "positive_definition": "Author-supplied ground truth, unchanged",
            "comparability": "author_protocol",
            "notes": [],
        },
        "audit": {
            "expected_references": expected_refs,
            "expected_queries": expected_queries,
            "missing_references": 0,
            "missing_queries": 0,
            "metadata_sha256": {},
            "source_images_modified": False,
        },
    }
    audit, protocol = manifest["audit"], manifest["protocol"]

    def load(relative, allow_pickle=False):
        path = repo_root / "datasets" / relative
        protocol["source"].append(str(path.relative_to(repo_root)))
        audit["metadata_sha256"][str(path.relative_to(repo_root))] = hashlib.sha256(
            path.read_bytes()).hexdigest()
        array = np.load(path, allow_pickle=allow_pickle)
        if array.ndim != 1:
            raise ValueError(f"Expected one-dimensional metadata: {path}")
        return array

    try:
        if name in {"MSLS", "MSLS_Test"}:
            folder = "msls_val" if name == "MSLS" else "msls_test"
            db = load(f"{folder}/{folder}_dbImages.npy", allow_pickle=True)
            all_queries = load(f"{folder}/{folder}_qImages.npy", allow_pickle=True)
            if name == "MSLS":
                indices = load("msls_val/msls_val_qIdx.npy")
                if (not np.issubdtype(indices.dtype, np.integer)
                        or len(set(map(int, indices))) != len(indices)
                        or np.any(indices < 0) or np.any(indices >= len(all_queries))):
                    raise ValueError("Invalid or duplicate MSLS validation query indices")
                queries = all_queries[indices]
                positives = load("msls_val/msls_val_pIdx.npy", allow_pickle=True)
                audit["raw_query_metadata_count"] = len(all_queries)
                audit["excluded_query_indices"] = sorted(
                    set(range(len(all_queries))) - set(map(int, indices)))
                protocol["notes"].append(
                    "Use the fixed cph/sf database and qIdx-selected 740 queries; "
                    "extra downloaded cities and queries do not enter evaluation.")
            else:
                queries, positives = all_queries, None
                protocol["positive_definition"] = "Test ground truth not loaded locally; official test_meta.zip must be integrated and validated before scoring"
                protocol["comparability"] = "unavailable"
                protocol["notes"].append(
                    "This local manifest has no loaded ground truth. Integrate official "
                    "test_meta.zip before calculating a test Recall score.")
            prefix = "train_val/" if name == "MSLS" else "test/"

            def msls_path(value):
                value = str(value)
                if not value.startswith(prefix):
                    raise ValueError(f"Unexpected MSLS metadata prefix: {value}")
                return repo_root / "datasets" / folder / value[len(prefix):]

            db_paths = [msls_path(value) for value in db]
            query_paths = [msls_path(value) for value in queries]
            audit["path_mapping"] = f"Remove only leading {prefix!r} from original IDs"
        elif name.startswith("pitts"):
            db = load(f"Pittsburgh/{name}_dbImages.npy")
            queries = load(f"Pittsburgh/{name}_qImages.npy")
            positives = load(f"Pittsburgh/{name}_gt.npy", allow_pickle=True)
            root = repo_root / "datasets/Pittsburgh/Pittsburgh"
            db_paths = [root / str(value) for value in db]
            query_paths = [root / str(value) for value in queries]
            audit["path_mapping"] = "Original raw relative path under Pittsburgh/Pittsburgh"
            protocol["notes"].append(
                "Retain original pitch/yaw views, database order, query order, "
                "and supplied ground truth; do not merge the formatted duplicate release.")
        elif name == "sfsm":
            db = load("sfsm/sfsm_dbImages.npy")
            queries = load("sfsm/sfsm_qImages.npy")
            root = repo_root / "datasets/sfsm/test"
            db_paths = [root / str(value) for value in db]
            query_paths = [root / str(value) for value in queries]
            positives = _radius_positives(db, queries)
            protocol["positive_definition"] = (
                "Euclidean UTM distance <= 25 m, using coordinates in the original NPY IDs")
            protocol["notes"].append(
                "Keep all 1000 original queries, including queries without any "
                "25 m database positive; do not expand the radius or discard queries.")
        else:
            db = load("tokyo247/tokyo247_dbImages.npy")
            queries = load("tokyo247/tokyo247_qImages.npy")
            root = repo_root / "datasets/tokyo247"
            source_refs, source_queries, source_audit = _tokyo_sources(root, db, queries)
            audit.update(source_audit)
            derived_root = view_root / "tokyo247"
            db_paths = [derived_root / str(value) for value in db]
            query_paths = [derived_root / str(value) for value in queries]
            positives = _radius_positives(db, queries)
            audit.update(_verify_tokyo_primary(repo_root, db, queries, positives))
            audit.update({
                "missing_source_references": sum(not path.is_file() for path in source_refs),
                "missing_source_queries": sum(not path.is_file() for path in source_queries),
                "path_mapping": "Database (panorama ID, NPY tile) -> original "
                                "(panorama ID, angle / 30); query NPY field 7 -> original JPG",
                "image_encoding": "Derived default Pillow JPEG; query short edge 480",
                "derived_image_root": str(derived_root),
                "preparation_function": "reproduction.adapt_retrieval.materialize_retrieval_views",
            })
            marker = derived_root / "format-recipe.json"
            recipe = json.loads(marker.read_text()) if marker.is_file() else {}
            audit["format_recipe"] = recipe
            audit["preparation_complete"] = recipe.get("status") == "complete"
            protocol["positive_definition"] = (
                "Euclidean UTM distance <= 25 m, using coordinates in the original NPY IDs")
            protocol["comparability"] = "conditional"
            protocol["notes"].extend([
                "Use precisely the 315 NPY query IDs (105 each day/sunset/night); "
                "exclude the 810 extra queries from the expanded local release.",
                "Each original PNG is one of the existing 12 panorama views. "
                "No extra crops, panorama fusion, or direction inference are applied.",
                "The published formatter decodes each original PNG and saves "
                "default Pillow JPEG; query images use scalar Resize(480) then "
                "default JPEG. materialize_retrieval_views reproduces those operations "
                "without overwriting source images and records library versions.",
                "Local queries are the expanded release, while the formatter "
                "downloads 247query_subset_v2. The 315 selected IDs match, but "
                "original v2 versus local image bytes have not been verified; "
                "report this release-identity condition with results.",
            ])
            if audit["official_mat_crosscheck"] == "identity_sets_and_all_positive_sets_match":
                protocol["notes"].append(
                    "The local primary NetVLAD MAT cross-check confirms all query/database "
                    "identities and all 315 positive sets; original NPY order is retained.")
            protocol["formatting_source"] = (
                "https://github.com/gmberton/VPR-datasets-downloader/blob/main/format_tokyo247.py")
            protocol["filename_source"] = (
                "https://github.com/gmberton/VPR-datasets-downloader/blob/main/util.py")

        manifest["references"] = _rows(db, db_paths)
        manifest["queries"] = _rows(queries, query_paths)
        if name == "tokyo247":
            for key, sources in (("references", source_refs), ("queries", source_queries)):
                for row, source in zip(manifest[key], sources):
                    row["source_path"] = str(source.resolve())
                    row["preprocessing"] = {
                        "resize_short_edge": 480 if key == "queries" else None,
                        "output_format": "JPEG",
                        "save_parameters": "Pillow defaults (no quality/subsampling overrides)",
                    }
        if positives is not None:
            positives = [list(map(int, row)) for row in positives]
            if len(positives) != len(queries) or any(
                    index < 0 or index >= len(db) for row in positives for index in row):
                raise ValueError("Ground-truth length or reference index is invalid")
            manifest["positives"] = positives
            empty = [i for i, row in enumerate(positives) if not row]
            audit.update({"queries_with_positives": len(queries) - len(empty),
                          "queries_without_positives": len(empty),
                          "queries_without_positive_indices": empty})
        audit["actual_references"] = len(db)
        audit["actual_queries"] = len(queries)
        for key in ("references", "queries"):
            missing = [row["id"] for row in manifest[key] if not Path(row["path"]).is_file()]
            audit[f"missing_{key}"] = len(missing)
            audit[f"missing_{key}_examples"] = missing[:10]
        if name == "tokyo247":
            audit["pending_derivations"] = audit["missing_references"] + audit["missing_queries"]
            audit["ready_for_materialization"] = not (
                audit["missing_source_references"] or audit["missing_source_queries"]
                or audit["unresolved_reference_ids"] or audit["ambiguous_reference_ids"])
            protocol["notes"].append(
                "A ready preparation plan still requires materialization and a final "
                "strict target-path check before descriptor extraction.")
        if ((len(db), len(queries)) != (expected_refs, expected_queries)
                or (name != "tokyo247" and (
                    audit["missing_references"] or audit["missing_queries"]))
                or audit.get("ambiguous_reference_ids")
                or audit.get("unresolved_reference_ids")
                or audit.get("missing_source_references") or audit.get("missing_source_queries")):
            protocol["notes"].append("Missing, ambiguous, or wrong-count inputs block evaluation.")
        else:
            manifest["status"] = "ready"
    except (OSError, ValueError, IndexError, TypeError, ImportError) as exc:
        audit["error"] = f"{type(exc).__name__}: {exc}"
        protocol["notes"].append("Metadata or mapping validation failed; no subset was substituted.")
    return manifest
