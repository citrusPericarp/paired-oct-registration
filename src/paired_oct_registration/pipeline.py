from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .registration_core import (
    A_OCT_SCALE_MAX,
    A_OCT_SCALE_MIN,
    REGISTRATION_TERMINAL_GUARD,
    build_a_oct_map,
    build_regularized_d_oct_field,
    choose_translation,
    compose_paper_map,
    estimate_column_observability,
    geometry_valid_from_observation,
    longest_contiguous_run,
    make_registration_boundary,
    phase_forward_translation,
    remap_with_map,
    retinal_feature,
    safe_geometry_domain,
)
from .registration_utils import (
    boundary_from_mask,
    load_boundary_file,
    load_image,
    load_mask,
    save_binary,
    save_gray,
)
from .reconstruction import sampling_map


def register_pair(
    *,
    sample_id: str,
    preoperative_image_path: Path,
    postoperative_image_path: Path,
    preoperative_mask_path: Path,
    postoperative_mask_path: Path,
    preoperative_boundary_path: Path,
    postoperative_boundary_path: Path,
    output_root: Path,
    node_count: int = 64,
    minimum_valid_fraction: float = 0.70,
    observability_threshold: float = 0.50,
) -> dict[str, Any]:
    """Register one postoperative B-scan to its paired preoperative B-scan.

    The masks and boundaries are model-derived structural priors. They are not
    manual annotations or clinical ground truth.
    """
    preoperative = load_image(preoperative_image_path)
    postoperative = load_image(postoperative_image_path)
    if preoperative.shape != postoperative.shape:
        raise ValueError(f"Paired image shapes differ for {sample_id}")

    preoperative_mask = load_mask(preoperative_mask_path, preoperative.shape)
    postoperative_mask = load_mask(postoperative_mask_path, postoperative.shape)
    preoperative_boundary = load_boundary_file(preoperative_boundary_path, preoperative_mask)
    postoperative_boundary = load_boundary_file(postoperative_boundary_path, postoperative_mask)

    preoperative_observation = estimate_column_observability(
        preoperative, preoperative_boundary, score_threshold=observability_threshold
    )
    postoperative_observation = estimate_column_observability(
        postoperative, postoperative_boundary, score_threshold=observability_threshold
    )
    preoperative_observable = np.asarray(preoperative_observation["registration_valid"], dtype=bool)
    postoperative_observable = np.asarray(postoperative_observation["registration_valid"], dtype=bool)

    phase = phase_forward_translation(
        retinal_feature(preoperative, preoperative_boundary, preoperative_observable),
        retinal_feature(postoperative, postoperative_boundary, postoperative_observable),
    )
    translation = choose_translation(
        preoperative_boundary,
        postoperative_boundary,
        phase,
        a_valid_override=preoperative_observable,
        b_valid_override=postoperative_observable,
    )
    forward_tx = float(translation["selected_forward_tx"])

    preoperative_geometry = geometry_valid_from_observation(
        preoperative_observation, preoperative_observable
    )
    postoperative_geometry = geometry_valid_from_observation(
        postoperative_observation, postoperative_observable
    )
    preoperative_registration_boundary = make_registration_boundary(
        preoperative_boundary, preoperative_geometry
    )
    postoperative_registration_boundary = make_registration_boundary(
        postoperative_boundary, postoperative_geometry
    )

    source_x_initial, source_y_initial, axial_scale, postoperative_top, _, target_top = build_a_oct_map(
        preoperative_registration_boundary,
        postoperative_registration_boundary,
        forward_tx,
        preoperative.shape,
        a_valid_override=preoperative_geometry,
        b_valid_override=postoperative_geometry,
        local_geometry=True,
    )
    if _needs_phase_translation_retry(translation, phase, axial_scale):
        forward_tx = float(phase["phase_forward_tx"])
        translation["selected_forward_tx"] = forward_tx
        translation["selected_forward_ty"] = float(phase["phase_forward_ty"])
        translation["phase_weight"] = 1.0
        translation["translation_source"] = "phase_after_a_oct_saturation_guard"
        translation["translation_safety_reason"] = "fovea_phase_disagreement_and_two_sided_scale_saturation"
        source_x_initial, source_y_initial, axial_scale, postoperative_top, _, target_top = build_a_oct_map(
            preoperative_registration_boundary,
            postoperative_registration_boundary,
            forward_tx,
            preoperative.shape,
            a_valid_override=preoperative_geometry,
            b_valid_override=postoperative_geometry,
            local_geometry=True,
        )
    initial_map_valid = _map_valid(source_x_initial, source_y_initial, postoperative.shape)
    initially_registered = remap_with_map(
        postoperative,
        source_x_initial,
        source_y_initial,
        cv2.INTER_CUBIC,
        cv2.BORDER_CONSTANT,
        0.0,
    )
    initially_registered[~initial_map_valid] = 0.0

    coverage_columns = preoperative_boundary.valid & postoperative_boundary.valid
    common_domain, _, _, _ = safe_geometry_domain(coverage_columns)
    source_x_line = source_x_initial[0]
    source_in_bounds = (
        np.isfinite(source_x_line)
        & (source_x_line >= 0.0)
        & (source_x_line <= float(postoperative.shape[1] - 1))
    )
    source_valid_at_target = np.zeros_like(source_in_bounds)
    if source_in_bounds.any():
        source_valid_at_target[source_in_bounds] = (
            np.interp(
                source_x_line[source_in_bounds],
                np.arange(postoperative.shape[1], dtype=np.float32),
                postoperative_boundary.valid.astype(np.float32),
            )
            >= 0.5
        )
    target_x = np.arange(preoperative.shape[1])
    guarded_domain = (
        common_domain
        & source_in_bounds
        & source_valid_at_target
        & (target_x >= REGISTRATION_TERMINAL_GUARD)
        & (target_x < preoperative.shape[1] - REGISTRATION_TERMINAL_GUARD)
        & (source_x_line >= REGISTRATION_TERMINAL_GUARD)
        & (source_x_line <= postoperative.shape[1] - 1 - REGISTRATION_TERMINAL_GUARD)
    )
    domain_start, domain_end = longest_contiguous_run(guarded_domain)
    registration_domain = np.zeros_like(guarded_domain)
    if domain_start >= 0:
        registration_domain[domain_start : domain_end + 1] = True

    postoperative_for_local_registration = initially_registered.copy()
    postoperative_for_local_registration[:, ~registration_domain] = 0.0
    preoperative_mask_in_domain = preoperative_mask & registration_domain[None, :]
    postoperative_mask_in_domain = postoperative_mask & postoperative_boundary.valid[None, :]
    local_valid_columns = preoperative_geometry & postoperative_geometry & registration_domain

    residual_field, local_stats, residual_nodes = build_regularized_d_oct_field(
        preoperative,
        postoperative_for_local_registration,
        preoperative_registration_boundary,
        local_valid_columns,
        node_count=node_count,
        minimum_valid_fraction=minimum_valid_fraction,
    )
    source_x, source_y, initial_stats = compose_paper_map(
        preoperative_registration_boundary,
        postoperative_registration_boundary,
        forward_tx,
        residual_field,
        a_valid_override=preoperative_geometry,
        b_valid_override=postoperative_geometry,
        local_geometry=True,
    )
    # Save the actual coordinate anchors, rather than recovering them later
    # from rounded observability scores or unprocessed model boundaries.
    parameters = dict(
        format_version=np.int32(2),
        image_shape=np.asarray(preoperative.shape, dtype=np.int32),
        source_x_line=source_x[0].copy(),
        initial_target_top=target_top.copy(),
        residual_target_upper=preoperative_registration_boundary.upper.copy(),
        residual_target_lower=preoperative_registration_boundary.lower.copy(),
        residual_nodes=residual_nodes,
        initial_axial_scale=axial_scale,
        initial_postoperative_top=postoperative_top,
        forward_translation_x=np.float64(forward_tx),
        common_domain=common_domain.astype(np.uint8),
        registration_domain=registration_domain.astype(np.uint8),
        preoperative_observable_domain=preoperative_observable.astype(np.uint8),
        postoperative_observable_domain=postoperative_observable.astype(np.uint8),
        preoperative_observation_score=np.asarray(preoperative_observation["score"], dtype=np.float32),
        postoperative_observation_score=np.asarray(postoperative_observation["score"], dtype=np.float32),
    )
    source_x, source_y = sampling_map(parameters)
    source_map_valid = _map_valid(source_x, source_y, postoperative.shape)
    registered_postoperative = remap_with_map(
        postoperative,
        source_x,
        source_y,
        cv2.INTER_CUBIC,
        cv2.BORDER_CONSTANT,
        0.0,
    )
    registered_postoperative[~source_map_valid] = 0.0
    registered_postoperative_mask = remap_with_map(
        postoperative_mask_in_domain.astype(np.float32),
        source_x,
        source_y,
        cv2.INTER_NEAREST,
        cv2.BORDER_CONSTANT,
        0.0,
    ) > 0.5
    registered_postoperative_mask &= registration_domain[None, :]

    registered_path = output_root / "aligned_B" / f"{sample_id}.png"
    source_valid_path = output_root / "valid_mask" / f"{sample_id}.png"
    evaluation_mask_path = output_root / "evaluation_mask" / f"{sample_id}.png"
    registered_mask_path = output_root / "aligned_mask_B" / f"{sample_id}.png"
    registered_boundary_path = output_root / "aligned_boundary_B" / f"{sample_id}.npz"
    deformation_path = output_root / "deformation" / f"{sample_id}.npz"
    save_gray(registered_path, registered_postoperative)
    save_binary(source_valid_path, source_map_valid)
    save_binary(evaluation_mask_path, preoperative_mask_in_domain & registered_postoperative_mask)
    save_binary(registered_mask_path, registered_postoperative_mask)
    aligned_upper, aligned_lower, aligned_valid = boundary_from_mask(registered_postoperative_mask)
    registered_boundary_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        registered_boundary_path,
        upper_y=aligned_upper.astype(np.float32),
        lower_y=aligned_lower.astype(np.float32),
        valid_columns=aligned_valid.astype(np.uint8),
    )
    deformation_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        deformation_path,
        **parameters,
    )
    return {
        "sample_id": sample_id,
        "status": "ok",
        "aligned_B": registered_path,
        "valid_mask": source_valid_path,
        "evaluation_mask": evaluation_mask_path,
        "aligned_mask_B": registered_mask_path,
        "aligned_boundary_B": registered_boundary_path,
        "deformation": deformation_path,
        "registration_domain_fraction": float(registration_domain.mean()),
        "source_map_valid_fraction": float(source_map_valid.mean()),
        "local_registration_enabled": bool(local_stats.get("d_oct_enabled", False)),
        "translation": translation,
        "initial_registration": initial_stats,
        "local_registration": local_stats,
    }


def _needs_phase_translation_retry(
    translation: dict[str, float],
    phase: dict[str, float],
    axial_scale: np.ndarray,
) -> bool:
    """Reject a pathological fovea match only when its A-OCT geometry also fails."""
    fovea_tx = float(translation["fovea_tx"])
    phase_tx = float(phase["phase_forward_tx"])
    scale_p05, scale_p95 = np.percentile(np.asarray(axial_scale, dtype=np.float32), [5, 95])
    return bool(
        translation.get("translation_source") == "fovea_safe"
        and 64.0 <= abs(fovea_tx) <= 96.0
        and abs(phase_tx) <= 32.0
        and abs(fovea_tx - phase_tx) >= 48.0
        and scale_p05 <= A_OCT_SCALE_MIN + 1e-3
        and scale_p95 >= A_OCT_SCALE_MAX - 1e-3
    )


def _map_valid(source_x: np.ndarray, source_y: np.ndarray, source_shape: tuple[int, int]) -> np.ndarray:
    return (
        np.isfinite(source_x)
        & np.isfinite(source_y)
        & (source_x >= 0.0)
        & (source_x <= float(source_shape[1] - 1))
        & (source_y >= 0.0)
        & (source_y <= float(source_shape[0] - 1))
    )
