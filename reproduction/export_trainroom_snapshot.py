"""Export existing benchmark recalls as one Trainroom evaluation snapshot.

The sole JSONL record has step=0, an evaluation snapshot index rather than a
training or optimizer step.  This script evaluates no model and changes no
benchmark result.  It uses only Python's standard library.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
from typing import Any


def export_snapshot(source: Path, output_dir: Path, expected_results: int | None = None) -> dict[str, Any]:
    source, output_dir = Path(source).resolve(), Path(output_dir).resolve()
    payload = source.read_bytes()
    rows = json.loads(payload)
    if not isinstance(rows, list) or not rows:
        raise ValueError("Expected a nonempty benchmark results list")
    if expected_results is not None and len(rows) != expected_results:
        raise ValueError(f"Expected {expected_results} model/dataset results, found {len(rows)}")
    metrics: dict[str, float] = {}
    sources: list[dict[str, Any]] = []
    for row in rows:
        if row.get("status") not in ("completed", "reused"):
            raise ValueError(f"Refusing incomplete result: {row.get('model')}/{row.get('dataset')}")
        result = row["result"]
        if not result.get("ground_truth_available", False):
            raise ValueError("Refusing a recall result without ground truth")
        model, dataset, comparability = (str(row[key]) for key in ("model", "dataset", "comparability"))
        if any("/" in value or not value for value in (model, dataset, comparability)):
            raise ValueError("Model, dataset and comparability must be nonempty tag components without slashes")
        prefix = f"evaluation_snapshot/{model}/{dataset}/{comparability}"
        for rank in (1, 5, 10):
            value = result["recalls"][str(rank)]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or not 0 <= value <= 1:
                raise ValueError(f"Invalid recorded R@{rank}: {value!r}")
            tag = f"{prefix}/R@{rank}_percent"
            if tag in metrics:
                raise ValueError(f"Duplicate model/dataset/protocol result: {tag}")
            metrics[tag] = float(value) * 100.0
        sources.append({
            "model": model, "dataset": dataset, "comparability": comparability,
            "source_status": row["status"], "result_path": row.get("result_path"),
            "checkpoint_sha256": result.get("checkpoint_sha256"),
            "protocol_fingerprint": result.get("protocol_fingerprint"),
        })
    metadata = {
        "total_steps": None,
        "config": {
            "record_type": "final_evaluation_snapshot",
            "step_definition": "step=0 is the sole final evaluation snapshot index; it is not a training, batch, epoch or optimizer step",
            "metric_unit": "percent; exactly the recorded recall fraction multiplied by 100",
            "snapshot_points_per_metric": 1,
            "source_results": str(source),
            "source_results_sha256": hashlib.sha256(payload).hexdigest(),
            "model_dataset_results": len(rows),
            "scalar_metrics": len(metrics),
            "exported_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
            "result_sources": sources,
        },
    }
    record = {"step": 0, "event": "final_evaluation_snapshot", "metrics": metrics}
    output_dir.mkdir(parents=True, exist_ok=True)
    for filename, text in (
        ("metrics.jsonl", json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n"),
        ("metadata.json", json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False) + "\n"),
    ):
        temporary = output_dir / ("." + filename + ".tmp")
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(output_dir / filename)
    (output_dir / "README.md").write_text(
        "# Final evaluation snapshot\n\n"
        "This file contains one existing final evaluation snapshot per scalar. "
        "The horizontal step value 0 is a snapshot index, not a training or optimizer step. "
        "No training curve, speed or progress is inferred.\n\n"
        "Metric names include the model, dataset and comparability label. Values are percentages "
        "computed by multiplying each recorded R@1/5/10 fraction by 100. "
        "The original results and per-dataset metrics are unchanged. "
        "See metadata.json for the source hash and result paths.\n",
        encoding="utf-8",
    )
    return {"source": str(source), "source_sha256": metadata["config"]["source_results_sha256"],
            "output": str(output_dir / "metrics.jsonl"), "model_dataset_results": len(rows),
            "scalar_metrics": len(metrics), "points_per_metric": 1, "snapshot_step": 0}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--expected-results", type=int)
    args = parser.parse_args()
    output_dir = args.output_dir or args.source.resolve().parent / "trainroom"
    print(json.dumps(export_snapshot(args.source, output_dir, args.expected_results), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
