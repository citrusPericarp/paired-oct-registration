from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from .pipeline import register_pair


REQUIRED_COLUMNS = (
    "sample_id",
    "real_A",
    "real_B",
    "mask_A",
    "mask_B",
    "boundary_A",
    "boundary_B",
)


def _resolve(dataset_root: Path, value: str) -> Path:
    path = (dataset_root / value).resolve()
    try:
        path.relative_to(dataset_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Manifest path leaves the dataset root: {value}") from exc
    return path


def _relative(path: Path, root: Path) -> str:
    return path.resolve().relative_to(root.resolve()).as_posix()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Register postoperative OCT scans to paired preoperative scans."
    )
    parser.add_argument("--dataset", type=Path, required=True, help="Downloaded dataset directory")
    parser.add_argument("--output", type=Path, required=True, help="Directory for generated files")
    parser.add_argument("--manifest", default="pairs.csv", help="Manifest path relative to dataset")
    parser.add_argument("--sample-id", action="append", default=[], help="Run only selected sample IDs")
    parser.add_argument("--limit", type=int, default=None, help="Run the first N selected rows")
    args = parser.parse_args(argv)

    dataset_root = args.dataset.resolve()
    output_root = args.output.resolve()
    manifest_path = _resolve(dataset_root, args.manifest)
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    missing_columns = [name for name in REQUIRED_COLUMNS if not rows or name not in rows[0]]
    if missing_columns:
        parser.error(f"Manifest is missing columns: {', '.join(missing_columns)}")
    if args.sample_id:
        selected = set(args.sample_id)
        rows = [row for row in rows if row["sample_id"] in selected]
    if args.limit is not None:
        if args.limit < 1:
            parser.error("--limit must be positive")
        rows = rows[: args.limit]
    if not rows:
        parser.error("No manifest rows selected")

    results: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    for index, row in enumerate(rows, start=1):
        sample_id = row["sample_id"]
        try:
            result = register_pair(
                sample_id=sample_id,
                preoperative_image_path=_resolve(dataset_root, row["real_A"]),
                postoperative_image_path=_resolve(dataset_root, row["real_B"]),
                preoperative_mask_path=_resolve(dataset_root, row["mask_A"]),
                postoperative_mask_path=_resolve(dataset_root, row["mask_B"]),
                preoperative_boundary_path=_resolve(dataset_root, row["boundary_A"]),
                postoperative_boundary_path=_resolve(dataset_root, row["boundary_B"]),
                output_root=output_root,
            )
            flat = {
                "sample_id": sample_id,
                "status": "ok",
                "aligned_B": _relative(result["aligned_B"], output_root),
                "aligned_mask_B": _relative(result["aligned_mask_B"], output_root),
                "aligned_boundary_B": _relative(result["aligned_boundary_B"], output_root),
                "valid_mask": _relative(result["valid_mask"], output_root),
                "evaluation_mask": _relative(result["evaluation_mask"], output_root),
                "deformation": _relative(result["deformation"], output_root),
                "registration_domain_fraction": result["registration_domain_fraction"],
                "source_map_valid_fraction": result["source_map_valid_fraction"],
                "local_registration_enabled": result["local_registration_enabled"],
            }
            results.append(flat)
            print(f"[{index}/{len(rows)}] {sample_id}: ok", flush=True)
        except Exception as exc:
            failures.append({"sample_id": sample_id, "error": str(exc)})
            print(f"[{index}/{len(rows)}] {sample_id}: failed: {exc}", flush=True)

    output_root.mkdir(parents=True, exist_ok=True)
    _write_csv(output_root / "registration_manifest.csv", results)
    (output_root / "failures.json").write_text(
        json.dumps(failures, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps({"selected": len(rows), "succeeded": len(results), "failed": len(failures)}))
    return 0 if not failures else 2


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "sample_id",
        "status",
        "aligned_B",
        "aligned_mask_B",
        "aligned_boundary_B",
        "valid_mask",
        "evaluation_mask",
        "deformation",
        "registration_domain_fraction",
        "source_map_valid_fraction",
        "local_registration_enabled",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    raise SystemExit(main())
