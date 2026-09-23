from __future__ import annotations

from typing import Any, Dict, List, Mapping, Tuple

import cv2
import numpy as np
from scipy.ndimage import gaussian_filter1d, median_filter

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False

from .registration_utils import BoundaryData, normalize_percentile


REGISTRATION_TERMINAL_GUARD = 32


GEOMETRY_SCORE_THRESHOLD = 0.65


MAX_TRUSTED_FOVEA_SHIFT = 96.0


MIN_PHASE_RESPONSE_FOR_FALLBACK = 0.15


A_OCT_SCALE_MIN = 0.80


A_OCT_SCALE_MAX = 1.25


A_OCT_LATERAL_SMOOTH_SIGMA = 12.0


REGISTRATION_BOUNDARY_SMOOTH_SIGMA = 10.0


D_OCT_LATERAL_SMOOTH_SIGMA = 6.0


D_OCT_AXIAL_SMOOTH_SIGMA = 2.0


D_OCT_MAX_RESIDUAL = 12.0


D_OCT_SAFE_MIN_COMMON_FRACTION = 0.75


def enforce_monotone_residual_nodes(
    residual_nodes: np.ndarray,
    boundary: BoundaryData,
    minimum_node_step: float = 0.05,
) -> np.ndarray:
    """Project residual nodes onto an anchored, strictly monotone axial map."""
    corrected = np.asarray(residual_nodes, dtype=np.float32).copy()
    node_count, width = corrected.shape
    node_axis = np.linspace(0.0, 1.0, node_count, dtype=np.float32)
    node_index = np.arange(node_count, dtype=np.float32)
    for x in range(width):
        top = float(boundary.upper[x])
        bottom = float(boundary.lower[x])
        if bottom <= top + 4.0:
            corrected[:, x] = 0.0
            continue
        base = top + node_axis * (bottom - top)
        lower = np.maximum(base - D_OCT_MAX_RESIDUAL, base[0] + minimum_node_step * node_index)
        upper = np.minimum(
            base + D_OCT_MAX_RESIDUAL,
            base[-1] - minimum_node_step * (node_count - 1 - node_index),
        )
        warped = np.clip(base + corrected[:, x], lower, upper)
        warped[0] = base[0]
        for node in range(1, node_count):
            warped[node] = max(warped[node], warped[node - 1] + minimum_node_step)
        warped[-1] = base[-1]
        corrected[:, x] = warped - base
    corrected[0, :] = 0.0
    corrected[-1, :] = 0.0
    return corrected


def fovea_candidate(boundary: BoundaryData, valid_override: np.ndarray | None = None) -> Dict[str, float]:
    """Find a conservative fovea proxy from the thickness curve.

    The paper uses the superior point of the thinnest retinal region.  Because
    our masks can have terminal invalid runs, the implementation searches only in the
    central 20%-80% interval and reports a confidence that is reduced for
    terminal or low-validity minima.
    """
    width = boundary.upper.size
    thickness = gaussian_filter1d(np.maximum(boundary.lower - boundary.upper, 4.0), sigma=8.0, mode="nearest")
    # Do not treat a minimum at the edge of a terminal search interval as a
    # trustworthy fovea. Some supplied trajectories contain weak terminal
    # columns, so phase correlation remains the fallback there.
    # Keep the central search interval so terminal minima are excluded, but do
    # not penalize a legitimate fovea merely because it lies at the interval
    # boundary (as happens for samples such as 0027/0028).
    lo, hi = int(0.25 * width), int(0.75 * width)
    candidates = np.arange(lo, max(lo + 1, hi), dtype=int)
    valid = boundary.valid if valid_override is None else np.asarray(valid_override, dtype=bool).reshape(-1)
    if valid.size != width:
        valid = boundary.valid
    candidates = candidates[valid[candidates]] if candidates.size else np.arange(width)
    if candidates.size == 0:
        candidates = np.arange(width)
    x = int(candidates[np.argmin(thickness[candidates])])
    centrality = float(1.0 - abs(x - 0.5 * (width - 1)) / max(0.5 * width, 1.0))
    confidence = float(np.clip(0.35 + 0.65 * centrality, 0.05, 1.0))
    confidence *= float(np.clip(boundary.quality.get("valid_fraction", 0.0), 0.25, 1.0))
    return {
        "x": float(x),
        "y": float(boundary.upper[x]),
        "thickness": float(thickness[x]),
        "centrality": centrality,
        "confidence": float(np.clip(confidence, 0.0, 1.0)),
    }


def phase_forward_translation(fixed_feature: np.ndarray, moving_feature: np.ndarray) -> Dict[str, float]:
    try:
        (dx, dy), response = cv2.phaseCorrelate(fixed_feature.astype(np.float32), moving_feature.astype(np.float32))
    except cv2.error:
        dx, dy, response = 0.0, 0.0, 0.0
    return {
        "phase_dx_reported": float(dx),
        "phase_dy_reported": float(dy),
        "phase_forward_tx": float(np.clip(-dx, -128.0, 128.0)),
        "phase_forward_ty": float(np.clip(-dy, -128.0, 128.0)),
        "phase_response": float(response),
    }


def retinal_feature(image: np.ndarray, boundary: BoundaryData, valid_override: np.ndarray | None = None) -> np.ndarray:
    normalized = normalize_percentile(image)
    smooth = cv2.GaussianBlur(normalized, (0, 0), 1.0)
    vertical = np.abs(cv2.Sobel(smooth, cv2.CV_32F, 0, 1, ksize=3))
    vertical = cv2.normalize(vertical, None, 0.0, 1.0, cv2.NORM_MINMAX)
    feature = (0.55 * normalized + 0.45 * vertical).astype(np.float32)
    y = np.arange(image.shape[0], dtype=np.float32)[:, None]
    band = (y >= boundary.upper[None, :] - 16.0) & (y <= boundary.lower[None, :] + 16.0)
    feature[~band] = 0.0
    if valid_override is not None:
        valid = np.asarray(valid_override, dtype=bool).reshape(-1)
        if valid.size == image.shape[1]:
            feature[:, ~valid] = 0.0
    return feature


def choose_translation(
    a_boundary: BoundaryData,
    b_boundary: BoundaryData,
    phase: Mapping[str, float],
    a_valid_override: np.ndarray | None = None,
    b_valid_override: np.ndarray | None = None,
) -> Dict[str, float]:
    a_fovea = fovea_candidate(a_boundary, a_valid_override)
    b_fovea = fovea_candidate(b_boundary, b_valid_override)
    raw_fovea_tx = a_fovea["x"] - b_fovea["x"]
    fovea_ty = a_fovea["y"] - b_fovea["y"]
    fovea_conf = min(a_fovea["confidence"], b_fovea["confidence"])
    phase_tx = float(phase["phase_forward_tx"])
    phase_ty = float(phase["phase_forward_ty"])
    phase_response = float(phase.get("phase_response", 0.0))

    # Phase correlation on partially observed OCT bands is prone to locking
    # onto the black/background boundary.  It is therefore a fallback only
    # when its response is genuinely strong and the fovea estimate is weak.
    # In particular, a clipped phase shift of -128 px is never allowed to
    # override two nearly coincident fovea candidates.
    fovea_outlier = abs(float(raw_fovea_tx)) > MAX_TRUSTED_FOVEA_SHIFT
    geometry_tx = 0.0 if fovea_outlier else float(raw_fovea_tx)
    phase_weight = 0.0
    if (not fovea_outlier) and fovea_conf < 0.35 and phase_response >= MIN_PHASE_RESPONSE_FOR_FALLBACK:
        phase_weight = float(np.clip((phase_response - MIN_PHASE_RESPONSE_FOR_FALLBACK) / 0.20, 0.0, 1.0))
    if fovea_outlier:
        translation_source = "zero_after_fovea_outlier"
        safety_reason = "fovea_shift_exceeded_safe_limit"
    elif phase_weight > 0.0:
        translation_source = "fovea_phase_fallback"
        safety_reason = "phase_response_passed_threshold"
    else:
        translation_source = "fovea_safe"
        safety_reason = "low_response_phase_ignored"
    tx = float(np.clip((1.0 - phase_weight) * geometry_tx + phase_weight * phase_tx, -MAX_TRUSTED_FOVEA_SHIFT, MAX_TRUSTED_FOVEA_SHIFT))
    ty = float(np.clip((1.0 - phase_weight) * fovea_ty + phase_weight * phase_ty, -128.0, 128.0))
    return {
        "fovea_A_x": a_fovea["x"],
        "fovea_A_y": a_fovea["y"],
        "fovea_B_x": b_fovea["x"],
        "fovea_B_y": b_fovea["y"],
        "fovea_A_confidence": a_fovea["confidence"],
        "fovea_B_confidence": b_fovea["confidence"],
        "fovea_confidence_used": fovea_conf,
        "fovea_tx": float(raw_fovea_tx),
        "fovea_ty": float(fovea_ty),
        "phase_weight": float(phase_weight),
        "phase_response_used": float(phase_response),
        "selected_forward_tx": tx,
        "selected_forward_ty": ty,
        "translation_source": translation_source,
        "translation_safety_reason": safety_reason,
    }


def longest_contiguous_run(values: np.ndarray) -> Tuple[int, int]:
    """Return the longest contiguous true interval in a 1-D validity mask.

    Registration must reason about a continuous anatomical overlap, not about
    a collection of isolated columns.  The inference decoder normally emits a
    contiguous interval, but this guard also handles older or externally
    supplied masks without turning gaps into artificial anatomy.
    """
    values = np.asarray(values, dtype=bool).reshape(-1)
    indices = np.flatnonzero(values)
    if indices.size == 0:
        return -1, -1
    breaks = np.flatnonzero(np.diff(indices) != 1)
    starts = np.r_[indices[0], indices[breaks + 1]]
    ends = np.r_[indices[breaks], indices[-1]]
    run = int(np.argmax(ends - starts + 1))
    return int(starts[run]), int(ends[run])


def fill_short_false_gaps(values: np.ndarray, max_gap: int = 24) -> np.ndarray:
    """Close only short gaps inside a 1-D observation run."""
    result = np.asarray(values, dtype=bool).reshape(-1).copy()
    true_indices = np.flatnonzero(result)
    if true_indices.size < 2:
        return result
    for left, right in zip(true_indices[:-1], true_indices[1:]):
        if int(right - left - 1) <= int(max_gap):
            result[left : right + 1] = True
    return result


def estimate_column_observability(
    image: np.ndarray,
    boundary: BoundaryData,
    *,
    band_margin: int = 8,
    context_gap: int = 8,
    context_radius: int = 56,
    score_threshold: float = 0.50,
) -> Dict[str, Any]:
    """Estimate whether each mask-labelled A-scan is actually observable.

    A decoder-valid column is not necessarily a usable registration column.
    Cataract, shadow, clipping, and acquisition dropout can make a retinal
    band look valid geometrically while its layer texture has disappeared.
    This gate compares the proposed retinal band with nearby background using
    intensity, local contrast, and within-column texture.  It is deliberately
    an observation-domain gate, not a replacement for the model-derived segmentation
    mask and not a clinical tissue classifier.

    The returned ``domain`` is the longest contiguous reliable run.  Keeping a
    single run prevents a warp from jumping over an unobserved gap and then
    inventing an anatomical correspondence on the other side.
    """
    height, width = image.shape
    normalized = normalize_percentile(image)
    signal = np.full(width, np.nan, dtype=np.float32)
    contrast = np.full(width, np.nan, dtype=np.float32)
    texture = np.full(width, np.nan, dtype=np.float32)
    whole_mean = np.mean(normalized, axis=0).astype(np.float32)
    bright_fraction = np.mean(normalized >= 0.985, axis=0).astype(np.float32)
    base_valid = np.asarray(boundary.valid, dtype=bool).reshape(-1)
    if base_valid.size != width:
        base_valid = np.ones(width, dtype=bool)

    for x in range(width):
        if not base_valid[x]:
            continue
        top = max(0, int(round(float(boundary.upper[x]) - band_margin)))
        bottom = min(height, int(round(float(boundary.lower[x]) + band_margin + 1.0)))
        if bottom <= top + 2:
            continue
        band = normalized[top:bottom, x]
        context_parts: List[np.ndarray] = []
        above_start = max(0, top - context_radius)
        above_end = max(above_start, top - context_gap)
        below_start = min(height, bottom + context_gap)
        below_end = min(height, bottom + context_radius)
        if above_end > above_start:
            context_parts.append(normalized[above_start:above_end, x])
        if below_end > below_start:
            context_parts.append(normalized[below_start:below_end, x])
        context = np.concatenate(context_parts) if context_parts else normalized[:, x]
        signal[x] = float(np.mean(band))
        contrast[x] = float(np.mean(band) - np.median(context))
        texture[x] = float(np.percentile(band, 90) - np.percentile(band, 10))

    finite = base_valid & np.isfinite(signal) & np.isfinite(contrast) & np.isfinite(texture)
    # A saturated full-height column is usually an image border/padding column,
    # not a strong retinal observation. It must not become the registration
    # endpoint merely because percentile normalization maps it to white.
    border_artifact = (bright_fraction >= 0.80) | (whole_mean >= 0.92)
    reference = finite & ~border_artifact
    if int(reference.sum()) < 16:
        reference = finite
    if int(reference.sum()) == 0:
        zero = np.zeros(width, dtype=np.float32)
        return {
            "score": zero,
            "signal": zero,
            "contrast": zero,
            "texture": zero,
            "registration_valid": np.zeros(width, dtype=bool),
            "domain": np.zeros(width, dtype=bool),
            "domain_start": -1,
            "domain_end": -1,
            "fallback_used": True,
            "fallback_reason": "no_finite_mask_columns",
            "border_artifact_fraction": float(border_artifact.mean()),
            "base_valid_fraction": float(base_valid.mean()),
            "observable_fraction": 0.0,
            "score_threshold": float(score_threshold),
        }

    signal_ref = max(float(np.percentile(signal[reference], 75)), 1e-3)
    contrast_ref = max(float(np.percentile(contrast[reference], 75)), 1e-3)
    texture_ref = max(float(np.percentile(texture[reference], 75)), 1e-3)
    signal_ratio = np.clip(signal / signal_ref, 0.0, 1.25)
    contrast_ratio = np.clip(contrast / contrast_ref, 0.0, 1.25)
    texture_ratio = np.clip(texture / texture_ref, 0.0, 1.25)
    score = (0.55 * signal_ratio + 0.30 * contrast_ratio + 0.15 * texture_ratio).astype(np.float32)
    score[~finite] = 0.0
    # Smooth only the score; the hard mask and border-artifact mask remain
    # explicit so smoothing cannot resurrect decoder-invalid columns.
    score_smooth = gaussian_filter1d(score, sigma=4.0, mode="nearest")
    reliable = (
        finite
        & ~border_artifact
        & (score_smooth >= float(score_threshold))
        & (signal >= max(0.030, 0.30 * signal_ref))
        & (texture >= max(0.060, 0.30 * texture_ref))
    )
    # Local contrast can fluctuate when the retina is sloped or shadowed even
    # though the A-scan still contains usable layer texture. The score already
    # includes contrast, so close only short binary gaps after the score gate;
    # a long low-signal interval remains invalid.
    reliable = fill_short_false_gaps(reliable, max_gap=24)
    start, end = longest_contiguous_run(reliable)
    fallback_used = False
    fallback_reason = ""
    # If an image is unusually dark but still has no score-supported run, do
    # not silently pass its whole mask to the registration. The caller will
    # produce a zero-coverage, auditable result instead of forcing a warp.
    if start >= 0 and end - start + 1 >= max(32, int(0.08 * width)):
        domain = np.zeros(width, dtype=bool)
        domain[start : end + 1] = True
    else:
        domain = np.zeros(width, dtype=bool)
        start, end = -1, -1
        fallback_reason = "no_sufficient_contiguous_observable_run"

    return {
        "score": np.clip(score_smooth, 0.0, 1.0).astype(np.float32),
        "signal": np.nan_to_num(signal, nan=0.0).astype(np.float32),
        "contrast": np.nan_to_num(contrast, nan=0.0).astype(np.float32),
        "texture": np.nan_to_num(texture, nan=0.0).astype(np.float32),
        "registration_valid": domain.copy(),
        "domain": domain,
        "domain_start": int(start),
        "domain_end": int(end),
        "fallback_used": bool(fallback_used),
        "fallback_reason": fallback_reason,
        "border_artifact_fraction": float(border_artifact.mean()),
        "base_valid_fraction": float(base_valid.mean()),
        "observable_fraction": float(domain.mean()),
        "score_threshold": float(score_threshold),
        "signal_reference": signal_ref,
        "contrast_reference": contrast_ref,
        "texture_reference": texture_ref,
    }


def safe_geometry_domain(
    valid_columns: np.ndarray,
    taper_columns: int = 24,
) -> Tuple[np.ndarray, np.ndarray, int, int]:
    """Build a continuous, softly tapered domain for mask-derived geometry.

    The binary domain is deliberately the longest common A/B run.  The weight
    tapers to zero at its ends, so a weak terminal boundary cannot create an
    abrupt affine or deformation jump.  Outside this domain the caller uses a
    global fallback rather than extrapolating a local boundary.
    """
    valid = np.asarray(valid_columns, dtype=bool).reshape(-1)
    start, end = longest_contiguous_run(valid)
    domain = np.zeros(valid.size, dtype=bool)
    weights = np.zeros(valid.size, dtype=np.float32)
    if start < 0:
        return domain, weights, start, end
    domain[start : end + 1] = True
    distance = np.minimum(np.arange(valid.size) - start, end - np.arange(valid.size)).astype(np.float32)
    taper = max(1, int(taper_columns))
    phase = np.clip(distance / float(taper), 0.0, 1.0)
    weights = (0.5 - 0.5 * np.cos(np.pi * phase)) * domain.astype(np.float32)
    # Keep the interior fully trusted when the common interval is wide enough.
    weights[distance >= taper] = 1.0
    return domain, weights.astype(np.float32), start, end


def geometry_valid_from_observation(
    observation: Mapping[str, Any],
    fallback_valid: np.ndarray,
    *,
    score_threshold: float = GEOMETRY_SCORE_THRESHOLD,
) -> np.ndarray:
    """Return columns allowed to drive local boundary geometry.

    ``registration_valid`` is intentionally broader because it defines the
    auditable output coverage.  Local A-OCT/D-OCT geometry needs a stricter
    subset: otherwise a low-signal but mask-labelled column can dictate a
    sharp boundary change.  The result remains one continuous run.
    """
    fallback = np.asarray(fallback_valid, dtype=bool).reshape(-1)
    scores = np.asarray(observation.get("score", np.zeros_like(fallback, dtype=np.float32)), dtype=np.float32).reshape(-1)
    if scores.size != fallback.size:
        return fallback.copy()
    candidate = fallback & np.isfinite(scores) & (scores >= float(score_threshold))
    candidate = fill_short_false_gaps(candidate, max_gap=24)
    start, end = longest_contiguous_run(candidate)
    result = np.zeros_like(candidate, dtype=bool)
    if start >= 0:
        result[start : end + 1] = True
    return result


def make_registration_boundary(
    boundary: BoundaryData,
    valid_override: np.ndarray | None = None,
    *,
    median_window: int = 9,
) -> BoundaryData:
    """Create a smoothed boundary used only by the registration map.

    The released mask/boundary arrays remain unchanged.  Registration uses a
    modest median-plus-Gaussian regularization so isolated decoder guesses do
    not become a high-curvature warp.  Columns outside the observation domain
    are filled only for numerical continuity; the caller still excludes them
    from the actual registration domain.
    """
    valid = np.asarray(boundary.valid, dtype=bool).reshape(-1).copy()
    if valid_override is not None:
        override = np.asarray(valid_override, dtype=bool).reshape(-1)
        if override.size == valid.size:
            valid &= override
    x_axis = np.arange(valid.size, dtype=np.float32)

    def smooth_curve(values: np.ndarray) -> np.ndarray:
        curve = np.asarray(values, dtype=np.float32).reshape(-1)
        finite = valid & np.isfinite(curve)
        if int(finite.sum()) >= 2:
            filled = np.interp(x_axis, x_axis[finite], curve[finite]).astype(np.float32)
        else:
            finite_all = np.isfinite(curve)
            filled = np.full(curve.size, float(np.nanmedian(curve[finite_all])) if finite_all.any() else 0.0, dtype=np.float32)
        # A small median stage removes one-column decoder jumps; the Gaussian
        # stage limits curvature without flattening the broad retinal profile.
        filtered = median_filter(filled, size=max(3, int(median_window) | 1), mode="nearest")
        return gaussian_filter1d(
            filtered.astype(np.float32),
            sigma=REGISTRATION_BOUNDARY_SMOOTH_SIGMA,
            mode="nearest",
        ).astype(np.float32)

    upper = smooth_curve(boundary.upper)
    lower = smooth_curve(boundary.lower)
    thickness = np.maximum(lower - upper, 4.0)
    center = 0.5 * (upper + lower)
    upper = center - 0.5 * thickness
    lower = center + 0.5 * thickness
    quality = dict(boundary.quality)
    quality["registration_smoothed"] = 1.0
    quality["registration_observable_fraction"] = float(valid.mean())
    return BoundaryData(upper=upper, lower=lower, valid=np.asarray(boundary.valid, dtype=bool).copy(), quality=quality)


def extrapolate_registration_curve(
    values: np.ndarray,
    valid: np.ndarray,
    default: float,
    *,
    slope_limit: float = 0.45,
) -> np.ndarray:
    """Continue a trusted curve smoothly beyond its valid interval.

    A constant global fallback can create a visible step exactly where local
    geometry becomes unreliable.  A short, slope-limited continuation is a
    safer handover: it preserves the last trusted trend without allowing a
    weak terminal mask to invent a new bend.
    """
    values = np.asarray(values, dtype=np.float32).reshape(-1)
    valid = np.asarray(valid, dtype=bool).reshape(-1)
    result = np.full(values.size, float(default), dtype=np.float32)
    start, end = longest_contiguous_run(valid)
    if start < 0:
        return result
    result[start : end + 1] = values[start : end + 1]
    span = max(4, min(32, end - start))
    if end > start:
        left_delta = np.diff(values[start : start + span + 1])
        right_delta = np.diff(values[end - span : end + 1])
        left_slope = float(np.median(left_delta)) if left_delta.size else 0.0
        right_slope = float(np.median(right_delta)) if right_delta.size else 0.0
    else:
        left_slope = right_slope = 0.0
    left_slope = float(np.clip(left_slope, -slope_limit, slope_limit))
    right_slope = float(np.clip(right_slope, -slope_limit, slope_limit))
    if start > 0:
        left_x = np.arange(start, dtype=np.float32)
        result[:start] = values[start] + left_slope * (left_x - float(start))
    if end < values.size - 1:
        right_x = np.arange(end + 1, values.size, dtype=np.float32)
        result[end + 1 :] = values[end] + right_slope * (right_x - float(end))
    return result.astype(np.float32)


def build_a_oct_map(
    a_boundary: BoundaryData,
    b_boundary: BoundaryData,
    forward_tx: float,
    shape: Tuple[int, int],
    *,
    a_valid_override: np.ndarray | None = None,
    b_valid_override: np.ndarray | None = None,
    local_geometry: bool = True,
    terminal_guard: int = REGISTRATION_TERMINAL_GUARD,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Build the target-to-source map for the paper's 1-D A-OCT affine stage."""
    height, width = shape
    target_x = np.arange(width, dtype=np.float32)
    source_x = target_x - float(forward_tx)
    source_x_clip = np.clip(source_x, 0.0, width - 1.0)
    source_b_top_raw = np.interp(source_x_clip, np.arange(width, dtype=np.float32), b_boundary.upper)
    source_b_bottom_raw = np.interp(source_x_clip, np.arange(width, dtype=np.float32), b_boundary.lower)
    target_a_top_raw = a_boundary.upper.astype(np.float32)
    target_a_bottom_raw = a_boundary.lower.astype(np.float32)
    source_thickness = np.maximum(source_b_bottom_raw - source_b_top_raw, 4.0)
    target_thickness = np.maximum(target_a_bottom_raw - target_a_top_raw, 4.0)
    # Paper Eq. (8): r_m = source thickness / target thickness.  This is the
    # pullback scale from target axial coordinate to source axial coordinate.
    scale_raw = np.clip(source_thickness / target_thickness, 0.45, 2.20).astype(np.float32)
    # Never extrapolate a local weak boundary into an unobservable or terminal
    # region.  The source-domain check is evaluated after the horizontal
    # translation, so a large bad shift cannot turn a source edge into a valid
    # target edge.
    a_valid = np.asarray(a_boundary.valid, dtype=bool).reshape(-1)
    b_valid = np.asarray(b_boundary.valid, dtype=bool).reshape(-1)
    if a_valid_override is not None and np.asarray(a_valid_override).size == width:
        a_valid &= np.asarray(a_valid_override, dtype=bool).reshape(-1)
    if b_valid_override is not None and np.asarray(b_valid_override).size == width:
        b_valid &= np.asarray(b_valid_override, dtype=bool).reshape(-1)
    source_valid = np.interp(source_x_clip, np.arange(width, dtype=np.float32), b_valid.astype(np.float32)) >= 0.5
    guard = max(0, int(terminal_guard))
    target_guard = (target_x >= guard) & (target_x <= float(width - 1 - guard))
    source_guard = (source_x >= float(guard)) & (source_x <= float(width - 1 - guard))
    common = a_valid & source_valid & target_guard & source_guard
    _, domain_weight, _, _ = safe_geometry_domain(common)
    trusted = common & np.isfinite(scale_raw) & np.isfinite(source_b_top_raw)
    if int(trusted.sum()) >= 2:
        fallback_scale = float(np.median(scale_raw[trusted]))
        fallback_target_top = float(np.median(target_a_top_raw[trusted]))
        fallback_target_thickness = float(np.median(target_thickness[trusted]))
        fallback_source_top = float(np.median(source_b_top_raw[trusted]))
    else:
        fallback_scale = 1.0
        fallback_target_top = float(np.median(target_a_top_raw))
        fallback_target_thickness = float(np.median(target_thickness))
        fallback_source_top = float(np.median(source_b_top_raw))
    fallback_scale = float(np.clip(fallback_scale, A_OCT_SCALE_MIN, A_OCT_SCALE_MAX))
    if not local_geometry:
        domain_weight = np.zeros_like(domain_weight, dtype=np.float32)
    fallback_scale_curve = extrapolate_registration_curve(scale_raw, common, fallback_scale, slope_limit=0.01)
    fallback_target_curve = extrapolate_registration_curve(target_a_top_raw, common, fallback_target_top, slope_limit=0.45)
    fallback_source_curve = extrapolate_registration_curve(source_b_top_raw, common, fallback_source_top, slope_limit=0.45)
    fallback_thickness_curve = extrapolate_registration_curve(
        target_thickness,
        common,
        max(fallback_target_thickness, 4.0),
        slope_limit=0.25,
    )
    fallback_scale_curve = np.clip(fallback_scale_curve, A_OCT_SCALE_MIN, A_OCT_SCALE_MAX).astype(np.float32)
    fallback_thickness_curve = np.maximum(fallback_thickness_curve, 4.0).astype(np.float32)
    scale = ((1.0 - domain_weight) * fallback_scale_curve + domain_weight * scale_raw).astype(np.float32)
    target_a_top = ((1.0 - domain_weight) * fallback_target_curve + domain_weight * target_a_top_raw).astype(np.float32)
    target_thickness_used = ((1.0 - domain_weight) * fallback_thickness_curve + domain_weight * target_thickness).astype(np.float32)
    source_b_top = ((1.0 - domain_weight) * fallback_source_curve + domain_weight * source_b_top_raw).astype(np.float32)
    # The local boundary estimate is only a weak prior.  Smooth the lateral
    # profiles once more and impose a conservative axial-scale bound so a
    # single bad A-scan cannot make B expand/contract sharply.
    scale = gaussian_filter1d(scale, sigma=A_OCT_LATERAL_SMOOTH_SIGMA, mode="nearest")
    target_a_top = gaussian_filter1d(target_a_top, sigma=A_OCT_LATERAL_SMOOTH_SIGMA, mode="nearest")
    target_thickness_used = gaussian_filter1d(target_thickness_used, sigma=A_OCT_LATERAL_SMOOTH_SIGMA, mode="nearest")
    source_b_top = gaussian_filter1d(source_b_top, sigma=A_OCT_LATERAL_SMOOTH_SIGMA, mode="nearest")
    scale = np.clip(scale, A_OCT_SCALE_MIN, A_OCT_SCALE_MAX).astype(np.float32)
    target_thickness_used = np.maximum(target_thickness_used, 4.0).astype(np.float32)
    source_b_bottom = source_b_top + scale * target_thickness_used
    yy = np.arange(height, dtype=np.float32)[:, None]
    source_y = source_b_top[None, :] + scale[None, :] * (yy - target_a_top[None, :])
    source_x_grid = np.broadcast_to(source_x[None, :], (height, width)).astype(np.float32)
    return source_x_grid, source_y.astype(np.float32), scale, source_b_top, source_b_bottom, target_a_top


def remap_with_map(image: np.ndarray, source_x: np.ndarray, source_y: np.ndarray, interpolation: int, border_mode: int, border_value: float = 0.0) -> np.ndarray:
    return cv2.remap(
        image.astype(np.float32),
        source_x.astype(np.float32),
        source_y.astype(np.float32),
        interpolation,
        borderMode=border_mode,
        borderValue=border_value,
    ).astype(np.float32)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    return (values - float(values.mean())) / max(float(values.std()), 1e-5)


if NUMBA_AVAILABLE:

    @njit(cache=False)
    def _zscore_numba(values: np.ndarray) -> np.ndarray:
        result = np.empty(values.size, dtype=np.float32)
        mean = np.float32(0.0)
        for index in range(values.size):
            mean += values[index]
        mean /= max(values.size, 1)
        variance = np.float32(0.0)
        for index in range(values.size):
            delta = values[index] - mean
            variance += delta * delta
        variance /= max(values.size, 1)
        scale = np.float32(np.sqrt(variance))
        if scale < np.float32(1e-5):
            scale = np.float32(1e-5)
        for index in range(values.size):
            result[index] = (values[index] - mean) / scale
        return result


    @njit(cache=False)
    def _gradient_numba(values: np.ndarray) -> np.ndarray:
        result = np.empty(values.size, dtype=np.float32)
        if values.size == 1:
            result[0] = 0.0
            return result
        result[0] = values[1] - values[0]
        result[-1] = values[-1] - values[-2]
        for index in range(1, values.size - 1):
            result[index] = np.float32(0.5) * (values[index + 1] - values[index - 1])
        return result


    @njit(cache=False)
    def _dtw_monotone_numba(ref_in: np.ndarray, moving_in: np.ndarray, max_warp_steps: int) -> Tuple[np.ndarray, float, float]:
        ref = _zscore_numba(ref_in)
        moving = _zscore_numba(moving_in)
        ref_grad = _zscore_numba(_gradient_numba(ref))
        moving_grad = _zscore_numba(_gradient_numba(moving))
        n = ref.size
        m = moving.size
        inf = np.float32(1e30)
        dp = np.full((n, m), inf, dtype=np.float32)
        back = np.full((n, m), -1, dtype=np.int8)
        dp[0, 0] = np.float32(0.72 * (ref[0] - moving[0]) ** 2 + 0.28 * (ref_grad[0] - moving_grad[0]) ** 2)
        gap_penalty = np.float32(0.10)
        for i in range(n):
            j0 = max(0, i - max_warp_steps)
            j1 = min(m, i + max_warp_steps + 1)
            for j in range(j0, j1):
                if i == 0 and j == 0:
                    continue
                best = inf
                code = np.int8(-1)
                if i > 0 and j > 0 and dp[i - 1, j - 1] < inf:
                    best = dp[i - 1, j - 1]
                    code = np.int8(0)
                if i > 0 and dp[i - 1, j] < inf and dp[i - 1, j] + gap_penalty < best:
                    best = dp[i - 1, j] + gap_penalty
                    code = np.int8(1)
                if j > 0 and dp[i, j - 1] < inf and dp[i, j - 1] + gap_penalty < best:
                    best = dp[i, j - 1] + gap_penalty
                    code = np.int8(2)
                if code >= 0:
                    local_cost = np.float32(0.72 * (ref[i] - moving[j]) ** 2 + 0.28 * (ref_grad[i] - moving_grad[j]) ** 2)
                    dp[i, j] = local_cost + best
                    back[i, j] = code
        mapping = np.empty(n, dtype=np.float32)
        counts = np.zeros(n, dtype=np.int32)
        sums = np.zeros(n, dtype=np.float32)
        if dp[n - 1, m - 1] >= inf:
            for i in range(n):
                mapping[i] = np.float32(i) * np.float32(m - 1) / max(n - 1, 1)
            return mapping, np.float32(np.nan), np.float32(np.nan)
        i = n - 1
        j = m - 1
        while i >= 0 and j >= 0:
            sums[i] += np.float32(j)
            counts[i] += 1
            if i == 0 and j == 0:
                break
            code = back[i, j]
            if code == 0:
                i -= 1
                j -= 1
            elif code == 1:
                i -= 1
            elif code == 2:
                j -= 1
            else:
                break
        first = -1
        last = -1
        for index in range(n):
            if counts[index] > 0:
                mapping[index] = sums[index] / counts[index]
                if first < 0:
                    first = index
                last = index
            else:
                mapping[index] = np.float32(np.nan)
        if first < 0:
            for index in range(n):
                mapping[index] = np.float32(index) * np.float32(m - 1) / max(n - 1, 1)
        else:
            for index in range(first):
                mapping[index] = mapping[first]
            for index in range(last + 1, n):
                mapping[index] = mapping[last]
            previous = first
            for index in range(first + 1, last + 1):
                if counts[index] > 0:
                    next_index = index
                    gap = next_index - previous
                    if gap > 1:
                        start_value = mapping[previous]
                        end_value = mapping[next_index]
                        for fill_index in range(previous + 1, next_index):
                            fraction = np.float32(fill_index - previous) / gap
                            mapping[fill_index] = start_value + fraction * (end_value - start_value)
                    previous = next_index
            if previous < last:
                for fill_index in range(previous + 1, last + 1):
                    mapping[fill_index] = mapping[previous]
        for index in range(1, n):
            if mapping[index] < mapping[index - 1]:
                mapping[index] = mapping[index - 1]
        for index in range(n):
            mapping[index] = min(max(mapping[index], 0.0), np.float32(m - 1))
        before = np.float32(0.0)
        upto = min(n, m)
        for index in range(upto):
            difference = ref[index] - moving[index]
            before += difference * difference
        before /= max(upto, 1)
        after = np.float32(0.0)
        for index in range(n):
            position = mapping[index]
            left = int(np.floor(position))
            right = min(left + 1, m - 1)
            fraction = position - left
            moved = moving[left] * (np.float32(1.0) - fraction) + moving[right] * fraction
            difference = ref[index] - moved
            after += difference * difference
        after /= max(n, 1)
        return mapping, float(before), float(after)

else:
    _dtw_monotone_numba = None


def dtw_monotone(ref: np.ndarray, moving: np.ndarray, max_warp_steps: int = 12) -> Tuple[np.ndarray, float, float]:
    """Monotone 1-D texture alignment for one A-scan.

    The path maps target depth samples to A-OCT depth samples.  It uses a
    normalized intensity/vertical-gradient cost and a small gap penalty to
    discourage gratuitous repeats.  The subsequent field smoothing is the
    cross-A-scan regularization stage.
    """
    # Use one reference numerical path regardless of optional Numba presence.
    # Different accumulation precision can change a tied DTW path.
    ref = zscore(ref)
    moving = zscore(moving)
    ref_grad = zscore(np.gradient(ref))
    moving_grad = zscore(np.gradient(moving))
    cost = 0.72 * (ref[:, None] - moving[None, :]) ** 2 + 0.28 * (ref_grad[:, None] - moving_grad[None, :]) ** 2
    n, m = cost.shape
    inf = np.float32(1e30)
    dp = np.full((n, m), inf, dtype=np.float32)
    back = np.full((n, m), -1, dtype=np.int8)
    dp[0, 0] = cost[0, 0]
    gap_penalty = 0.10
    for i in range(n):
        j0, j1 = max(0, i - max_warp_steps), min(m, i + max_warp_steps + 1)
        for j in range(j0, j1):
            if i == 0 and j == 0:
                continue
            candidates: List[Tuple[float, int]] = []
            if i > 0 and j > 0 and np.isfinite(dp[i - 1, j - 1]):
                candidates.append((float(dp[i - 1, j - 1]), 0))
            if i > 0 and np.isfinite(dp[i - 1, j]):
                candidates.append((float(dp[i - 1, j] + gap_penalty), 1))
            if j > 0 and np.isfinite(dp[i, j - 1]):
                candidates.append((float(dp[i, j - 1] + gap_penalty), 2))
            if candidates:
                value, code = min(candidates, key=lambda item: item[0])
                dp[i, j] = cost[i, j] + value
                back[i, j] = code
    if not np.isfinite(dp[-1, -1]):
        mapping = np.arange(n, dtype=np.float32) * (m - 1) / max(n - 1, 1)
        return mapping, float("nan"), float("nan")
    i, j = n - 1, m - 1
    path: List[Tuple[int, int]] = []
    while i >= 0 and j >= 0:
        path.append((i, j))
        if i == 0 and j == 0:
            break
        code = int(back[i, j])
        if code == 0:
            i, j = i - 1, j - 1
        elif code == 1:
            i -= 1
        elif code == 2:
            j -= 1
        else:
            break
    path.reverse()
    mapping = np.full(n, np.nan, dtype=np.float32)
    buckets: Dict[int, List[int]] = {}
    for ii, jj in path:
        buckets.setdefault(ii, []).append(jj)
    valid_i = sorted(buckets)
    if valid_i:
        mapping[valid_i] = np.asarray([np.mean(buckets[ii]) for ii in valid_i], dtype=np.float32)
        mapping = np.interp(np.arange(n), valid_i, mapping[valid_i]).astype(np.float32)
    else:
        mapping = np.arange(n, dtype=np.float32) * (m - 1) / max(n - 1, 1)
    mapping = np.maximum.accumulate(mapping)
    mapping = np.clip(mapping, 0.0, float(m - 1))
    before = float(np.mean((ref - moving[:n]) ** 2)) if m >= n else float("nan")
    moved = np.interp(np.arange(n, dtype=np.float32), np.arange(m, dtype=np.float32), moving, left=float(moving[0]), right=float(moving[-1]))
    after = float(np.mean((ref - np.interp(mapping, np.arange(m, dtype=np.float32), moving)) ** 2))
    return mapping, before, after


def build_regularized_d_oct_field(
    a: np.ndarray,
    a_oct: np.ndarray,
    a_boundary: BoundaryData,
    valid_columns: np.ndarray,
    node_count: int = 64,
    minimum_valid_fraction: float = 0.70,
) -> Tuple[np.ndarray, Dict[str, float], np.ndarray]:
    """Estimate a smooth axial layer deformation field from A-scan texture."""
    height, width = a.shape
    a_norm = normalize_percentile(a)
    a_oct_norm = normalize_percentile(a_oct)
    residual_nodes = np.full((node_count, width), np.nan, dtype=np.float32)
    domain, domain_weight, domain_start, domain_end = safe_geometry_domain(valid_columns)
    common_fraction = float(domain.mean())
    # A low-coverage weak mask is not allowed to drive a dense deformation
    # field.  A-OCT/global translation remains available, while D-OCT falls
    # back to the zero residual field and records the reason for auditability.
    safe_fraction_threshold = max(float(minimum_valid_fraction), D_OCT_SAFE_MIN_COMMON_FRACTION)
    d_oct_enabled = common_fraction >= safe_fraction_threshold
    if not d_oct_enabled:
        zero_nodes = np.zeros((node_count, width), dtype=np.float32)
        return np.zeros((height, width), dtype=np.float32), {
            "d_oct_enabled": 0.0,
            "d_oct_disabled_low_common_fraction": 1.0,
            "d_oct_valid_columns": 0.0,
            "d_oct_valid_fraction": 0.0,
            "d_oct_common_domain_fraction": common_fraction,
            "d_oct_safe_min_common_fraction": safe_fraction_threshold,
            "d_oct_domain_start": float(domain_start),
            "d_oct_domain_end": float(domain_end),
            "d_oct_ssd_before_mean": float("nan"),
            "d_oct_ssd_after_mean": float("nan"),
            "d_oct_residual_mean": 0.0,
            "d_oct_residual_std": 0.0,
            "d_oct_residual_p95_abs": 0.0,
            "d_oct_residual_max_abs": 0.0,
        }, zero_nodes
    before_cost: List[float] = []
    after_cost: List[float] = []
    valid_count = 0
    for x in range(width):
        if not domain[x]:
            continue
        top = float(a_boundary.upper[x])
        bottom = float(a_boundary.lower[x])
        thickness = bottom - top
        if thickness < 24.0 or thickness > 240.0:
            continue
        target_y = np.linspace(top, bottom, node_count, dtype=np.float32)
        ref = np.interp(target_y, np.arange(height, dtype=np.float32), a_norm[:, x])
        moving = np.interp(target_y, np.arange(height, dtype=np.float32), a_oct_norm[:, x])
        mapping, before, after = dtw_monotone(ref, moving, max_warp_steps=max(6, int(0.20 * node_count)))
        residual = (mapping - np.arange(node_count, dtype=np.float32)) * (thickness / max(node_count - 1, 1))
        # Preserve the ILM/BrM anchors from A-OCT; any residual deformation is
        # forced to taper to zero at both retinal boundaries.
        residual -= np.linspace(residual[0], residual[-1], node_count, dtype=np.float32)
        residual[0] = 0.0
        residual[-1] = 0.0
        residual_nodes[:, x] = np.clip(residual, -D_OCT_MAX_RESIDUAL, D_OCT_MAX_RESIDUAL)
        if np.isfinite(before):
            before_cost.append(before)
        if np.isfinite(after):
            after_cost.append(after)
        valid_count += 1
    x_axis = np.arange(width, dtype=np.float32)
    for node in range(node_count):
        row = residual_nodes[node]
        valid = np.isfinite(row)
        if int(valid.sum()) == 0:
            row[:] = 0.0
        elif int(valid.sum()) == 1:
            row[:] = 0.0
        else:
            lo = int(np.flatnonzero(valid)[0])
            hi = int(np.flatnonzero(valid)[-1])
            known_x = x_axis[valid].copy()
            known_values = row[valid].copy()
            row[:] = 0.0
            row[lo : hi + 1] = np.interp(x_axis[lo : hi + 1], known_x, known_values).astype(np.float32)
        residual_nodes[node] = row
    # Compact-RBF-like regularization in the axial direction and across
    # neighboring A-scans.  The support is deliberately local: broad anatomy
    # remains, isolated bad columns do not.
    residual_nodes = gaussian_filter1d(residual_nodes, sigma=D_OCT_AXIAL_SMOOTH_SIGMA, axis=0, mode="nearest")
    residual_nodes = gaussian_filter1d(residual_nodes, sigma=D_OCT_LATERAL_SMOOTH_SIGMA, axis=1, mode="nearest")
    residual_nodes = np.clip(residual_nodes, -D_OCT_MAX_RESIDUAL, D_OCT_MAX_RESIDUAL)
    residual_nodes *= domain_weight[None, :]
    residual_nodes[0, :] = 0.0
    residual_nodes[-1, :] = 0.0
    residual_nodes = enforce_monotone_residual_nodes(residual_nodes, a_boundary)
    field = np.zeros((height, width), dtype=np.float32)
    yy = np.arange(height, dtype=np.float32)
    node_axis = np.linspace(0.0, 1.0, node_count, dtype=np.float32)
    for x in range(width):
        top = float(a_boundary.upper[x])
        bottom = float(a_boundary.lower[x])
        if bottom <= top + 4.0:
            continue
        target_y = top + node_axis * (bottom - top)
        field[:, x] = np.interp(yy, target_y, residual_nodes[:, x], left=0.0, right=0.0).astype(np.float32)
        # Keep the deformation axial and topology-safe in the implementation.
        field[:, x] = np.clip(field[:, x], -D_OCT_MAX_RESIDUAL, D_OCT_MAX_RESIDUAL)
    stats = {
        "d_oct_enabled": 1.0,
        "d_oct_disabled_low_common_fraction": 0.0,
        "d_oct_valid_columns": float(valid_count),
        "d_oct_valid_fraction": float(valid_count / max(width, 1)),
        "d_oct_common_domain_fraction": common_fraction,
        "d_oct_safe_min_common_fraction": safe_fraction_threshold,
        "d_oct_domain_start": float(domain_start),
        "d_oct_domain_end": float(domain_end),
        "d_oct_ssd_before_mean": float(np.mean(before_cost)) if before_cost else float("nan"),
        "d_oct_ssd_after_mean": float(np.mean(after_cost)) if after_cost else float("nan"),
        "d_oct_residual_mean": float(np.mean(field)),
        "d_oct_residual_std": float(np.std(field)),
        "d_oct_residual_p95_abs": float(np.percentile(np.abs(field), 95)),
        "d_oct_residual_max_abs": float(np.max(np.abs(field))),
    }
    return field, stats, residual_nodes


def compose_paper_map(
    a_boundary: BoundaryData,
    b_boundary: BoundaryData,
    tx: float,
    residual_field: np.ndarray,
    *,
    a_valid_override: np.ndarray | None = None,
    b_valid_override: np.ndarray | None = None,
    local_geometry: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    height, width = residual_field.shape
    source_x, source_y_a_oct, scale, b_top, b_bottom, target_top = build_a_oct_map(
        a_boundary,
        b_boundary,
        tx,
        (height, width),
        a_valid_override=a_valid_override,
        b_valid_override=b_valid_override,
        local_geometry=local_geometry,
    )
    yy = np.arange(height, dtype=np.float32)[:, None]
    source_y = b_top[None, :] + scale[None, :] * (yy + residual_field - target_top[None, :])
    vertical_step = np.diff(source_y, axis=0)
    scale_lateral_step = np.diff(scale)
    stats = {
        "a_oct_scale_median": float(np.median(scale)),
        "a_oct_scale_p05": float(np.percentile(scale, 5)),
        "a_oct_scale_p95": float(np.percentile(scale, 95)),
        "a_oct_local_geometry_enabled": float(bool(local_geometry)),
        "a_oct_source_x_outside_fraction": float(np.mean((source_x < 0.0) | (source_x > width - 1.0))),
        "a_oct_source_y_min": float(np.min(source_y)),
        "a_oct_source_y_max": float(np.max(source_y)),
        "a_oct_scale_lateral_step_p95_abs": float(np.percentile(np.abs(scale_lateral_step), 95)) if scale_lateral_step.size else 0.0,
        "a_oct_vertical_step_p01": float(np.percentile(vertical_step, 1)) if vertical_step.size else 0.0,
        "a_oct_vertical_step_p99": float(np.percentile(vertical_step, 99)) if vertical_step.size else 0.0,
        "a_oct_max_abs_residual_input": float(np.max(np.abs(residual_field))) if residual_field.size else 0.0,
    }
    return source_x, source_y, stats
