from __future__ import annotations

import argparse
import csv
import filecmp
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from paired_oct_registration.pipeline import register_pair  # noqa: E402


INPUT_PRODUCTS = {
    "mask_A": "mask_A.png",
    "mask_B": "mask_B.png",
    "boundary_A": "boundary_A.npz",
    "boundary_B": "boundary_B.npz",
}
DERIVED_PRODUCTS = {
    "aligned_B": ("aligned_B", "aligned_B.png"),
    "aligned_mask_B": ("aligned_mask_B", "aligned_mask_B.png"),
    "aligned_boundary_B": ("aligned_boundary_B", "aligned_boundary_B.npz"),
    "valid_mask": ("valid_mask", "valid_mask.png"),
    "evaluation_mask": ("evaluation_mask", "auxiliary/evaluation_mask.png"),
    "deformation": ("deformation", "auxiliary/deformation.npz"),
}


def rows_from(dataset: Path) -> list[dict[str, str]]:
    with (dataset / "pairs.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 2000 or len({row["sample_id"] for row in rows}) != 2000:
        raise RuntimeError(f"expected 2000 unique pairs, found {len(rows)}")
    return rows


def run_inference(args: argparse.Namespace, inference: Path) -> None:
    command = [
        sys.executable,
        str(args.segmentation_project / "scripts" / "infer_structured_release.py"),
        "--checkpoint", str(args.checkpoint),
        "--decoder-config", str(args.decoder_config),
        "--manifest", str(args.manifest),
        "--out-dir", str(inference),
        "--device", args.device,
        "--batch-size", str(args.batch_size),
    ]
    subprocess.run(command, cwd=args.segmentation_project, check=True)
    with (inference / "predictions.csv").open(encoding="utf-8-sig", newline="") as handle:
        predictions = list(csv.DictReader(handle))
    if len(predictions) != 4000:
        raise RuntimeError(f"expected 4000 mask predictions, found {len(predictions)}")


def inferred_paths(inference: Path, sample_id: str) -> dict[str, Path]:
    return {
        "mask_A": inference / "masks" / "real_A" / f"{sample_id}_arc.png",
        "boundary_A": inference / "masks" / "real_A" / f"{sample_id}_arc_boundaries.npz",
        "mask_B": inference / "masks" / "real_B" / f"{sample_id}_clean.png",
        "boundary_B": inference / "masks" / "real_B" / f"{sample_id}_clean_boundaries.npz",
    }


def stage_inputs(dataset: Path, inference: Path, stage: Path, rows: list[dict[str, str]]) -> None:
    for index, row in enumerate(rows, 1):
        sources = inferred_paths(inference, row["sample_id"])
        for key, source in sources.items():
            if not source.is_file():
                raise FileNotFoundError(source)
            destination = stage / row[key]
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if index % 200 == 0:
            print(f"staged input masks/boundaries: {index}/{len(rows)}", flush=True)


def copy_derived(flat: Path, stage: Path, row: dict[str, str]) -> None:
    for key, (source_dir, _) in DERIVED_PRODUCTS.items():
        destination = stage / row[key]
        source = flat / source_dir / f"{row['sample_id']}{destination.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)


def regenerate(dataset: Path, stage: Path, rows: list[dict[str, str]]) -> None:
    flat = stage / "_generated_flat"
    manifest: list[dict[str, object]] = []
    for index, row in enumerate(rows, 1):
        result = register_pair(
            sample_id=row["sample_id"],
            preoperative_image_path=dataset / row["real_A"],
            postoperative_image_path=dataset / row["real_B"],
            preoperative_mask_path=stage / row["mask_A"],
            postoperative_mask_path=stage / row["mask_B"],
            preoperative_boundary_path=stage / row["boundary_A"],
            postoperative_boundary_path=stage / row["boundary_B"],
            output_root=flat,
        )
        copy_derived(flat, stage, row)
        manifest.append({
            "sample_id": row["sample_id"],
            "patient_id": row["patient_id"],
            "scan_id": row["scan_id"],
            "label": row["label"],
            "translation_source": result["translation"]["translation_source"],
            "source_map_valid_fraction": result["source_map_valid_fraction"],
        })
        if index == 1 or index % 50 == 0 or index == len(rows):
            print(f"registered pairs: {index}/{len(rows)}", flush=True)
    with (stage / "regeneration_manifest.csv").open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    shutil.rmtree(flat)


def copy_subset_views(stage: Path, rows: list[dict[str, str]]) -> None:
    keys = tuple(INPUT_PRODUCTS) + tuple(DERIVED_PRODUCTS)
    for index, row in enumerate(rows, 1):
        for key in keys:
            source = stage / row[key]
            parts = Path(row[key]).parts
            destination = stage / f"{row['label']}_dataset" / Path(*parts[1:])
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, destination)
        if index % 200 == 0:
            print(f"staged subset views: {index}/{len(rows)}", flush=True)


def validate_image(path: Path, binary: bool, allow_empty: bool = False) -> None:
    array = np.asarray(Image.open(path))
    if array.shape != (496, 768):
        raise RuntimeError(f"invalid image shape {array.shape}: {path}")
    if not allow_empty and int(array.max()) == 0:
        raise RuntimeError(f"empty image: {path}")
    if binary and not set(np.unique(array)).issubset({0, 255}):
        raise RuntimeError(f"non-binary mask: {path}")


def validate_archive(path: Path, required: tuple[str, ...]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        missing = [key for key in required if key not in archive.files]
        if missing:
            raise RuntimeError(f"missing fields {missing}: {path}")
        for key in archive.files:
            value = archive[key]
            if np.issubdtype(value.dtype, np.floating) and not np.isfinite(value).all():
                raise RuntimeError(f"non-finite field {key}: {path}")


def qa(stage: Path, rows: list[dict[str, str]]) -> dict[str, object]:
    valid_widths: list[int] = []
    for index, row in enumerate(rows, 1):
        for key in ("mask_A", "mask_B", "aligned_mask_B", "valid_mask", "evaluation_mask"):
            validate_image(stage / row[key], binary=True)
        validate_image(stage / row["aligned_B"], binary=False)
        for key in ("boundary_A", "boundary_B", "aligned_boundary_B"):
            validate_archive(stage / row[key], ("upper_y", "lower_y", "valid_columns"))
        validate_archive(stage / row["deformation"], ("residual_nodes", "common_domain", "registration_domain"))
        with np.load(stage / row["boundary_A"], allow_pickle=False) as archive:
            valid_widths.append(int(np.count_nonzero(archive["valid_columns"])))
        if index % 200 == 0:
            print(f"QA pairs: {index}/{len(rows)}", flush=True)
    summary = {
        "pairs": len(rows),
        "predictions": 2 * len(rows),
        "files_to_replace": 2 * len(rows) * (len(INPUT_PRODUCTS) + len(DERIVED_PRODUCTS)),
        "mask_A_valid_columns_min": min(valid_widths),
        "mask_A_valid_columns_median": float(np.median(valid_widths)),
        "mask_A_valid_columns_max": max(valid_widths),
        "status": "passed",
    }
    (stage / "qa_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def relative_targets(rows: list[dict[str, str]]) -> list[Path]:
    keys = tuple(INPUT_PRODUCTS) + tuple(DERIVED_PRODUCTS)
    targets: list[Path] = []
    for row in rows:
        for key in keys:
            all_path = Path(row[key])
            targets.append(all_path)
            targets.append(Path(f"{row['label']}_dataset") / Path(*all_path.parts[1:]))
    if len(targets) != len(set(targets)):
        raise RuntimeError("duplicate replacement targets")
    return targets


def backup(dataset: Path, backup_root: Path, targets: list[Path]) -> None:
    if backup_root.exists():
        raise FileExistsError(f"backup already exists: {backup_root}")
    for index, relative in enumerate(targets, 1):
        source, destination = dataset / relative, backup_root / relative
        if not source.is_file():
            raise FileNotFoundError(source)
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)
        if index % 2000 == 0:
            print(f"backup files: {index}/{len(targets)}", flush=True)
    (backup_root / "backup_summary.json").write_text(
        json.dumps({"files": len(targets), "dataset": str(dataset)}, indent=2), encoding="utf-8"
    )


def atomic_copy(source: Path, destination: Path) -> None:
    temporary = destination.with_name(destination.name + ".repairing")
    shutil.copyfile(source, temporary)
    os.replace(temporary, destination)


def apply(stage: Path, dataset: Path, backup_root: Path, targets: list[Path]) -> None:
    backup(dataset, backup_root, targets)
    replaced: list[Path] = []
    try:
        for index, relative in enumerate(targets, 1):
            atomic_copy(stage / relative, dataset / relative)
            replaced.append(relative)
            if index % 2000 == 0:
                print(f"replaced files: {index}/{len(targets)}", flush=True)
    except Exception:
        for relative in replaced:
            atomic_copy(backup_root / relative, dataset / relative)
        raise
    mismatches = [
        str(relative) for relative in targets
        if not filecmp.cmp(stage / relative, dataset / relative, shallow=False)
    ]
    if mismatches:
        raise RuntimeError(f"post-replacement mismatch count={len(mismatches)} first={mismatches[:5]}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Regenerate, QA, back up, and replace all released OCT masks and dependent products")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--segmentation-project", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--decoder-config", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    parser.add_argument("--backup-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--apply", action="store_true", help="Replace the release only after staging and QA pass")
    args = parser.parse_args()
    args.dataset = args.dataset.resolve()
    args.segmentation_project = args.segmentation_project.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.decoder_config = args.decoder_config.resolve()
    args.manifest = args.manifest.resolve()
    args.work_root = args.work_root.resolve()
    args.backup_root = args.backup_root.resolve()
    if args.work_root.exists():
        raise FileExistsError(f"work root already exists: {args.work_root}")
    args.work_root.mkdir(parents=True)
    inference, stage = args.work_root / "inference", args.work_root / "stage"
    stage.mkdir()
    rows = rows_from(args.dataset)
    run_inference(args, inference)
    stage_inputs(args.dataset, inference, stage, rows)
    regenerate(args.dataset, stage, rows)
    copy_subset_views(stage, rows)
    summary = qa(stage, rows)
    targets = relative_targets(rows)
    summary.update({"applied": bool(args.apply), "backup_root": str(args.backup_root)})
    if args.apply:
        apply(stage, args.dataset, args.backup_root, targets)
        summary["post_replacement_match"] = True
    (args.work_root / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
