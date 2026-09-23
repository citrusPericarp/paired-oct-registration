from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter1d, median_filter
from scipy.signal import savgol_filter


@dataclass
class BoundaryData:
    upper: np.ndarray
    lower: np.ndarray
    valid: np.ndarray
    quality: Dict[str, float]


def load_image(path: Path) -> np.ndarray:
    image = np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0
    return image


def load_mask(path: Path, shape: Tuple[int, int] | None = None) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    if shape is not None and tuple(mask.shape) != tuple(shape):
        mask = cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST) > 0
    return mask


def save_gray(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = np.clip(np.asarray(image) * 255.0, 0, 255).astype(np.uint8)
    Image.fromarray(array, mode="L").save(path)


def save_binary(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(mask, dtype=bool).astype(np.uint8) * 255), mode="L").save(path)


def normalize_percentile(image: np.ndarray, low: float = 1.0, high: float = 99.0) -> np.ndarray:
    finite = image[np.isfinite(image)]
    if finite.size == 0:
        return np.zeros_like(image, dtype=np.float32)
    lo, hi = np.percentile(finite, [low, high])
    return np.clip((image - lo) / max(float(hi - lo), 1e-6), 0.0, 1.0).astype(np.float32)


def boundary_from_mask(mask: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = mask.shape
    top = np.full(width, np.nan, dtype=np.float32)
    bottom = np.full(width, np.nan, dtype=np.float32)
    valid = np.zeros(width, dtype=bool)
    for x in range(width):
        ys = np.flatnonzero(mask[:, x])
        if ys.size:
            top[x] = float(ys[0])
            bottom[x] = float(ys[-1])
            valid[x] = True
    columns = np.arange(width)
    good = np.flatnonzero(valid)
    if good.size == 0:
        center = np.full(width, height * 0.5, dtype=np.float32)
        return center - 25.0, center + 25.0, valid
    top = np.interp(columns, good, top[good]).astype(np.float32)
    bottom = np.interp(columns, good, bottom[good]).astype(np.float32)
    return top, bottom, valid


def odd_window(size: int, requested: int) -> int:
    if size < 3:
        return 3
    result = min(int(requested), size if size % 2 else size - 1)
    result = max(result, 3)
    return result if result % 2 else result - 1


def load_boundary_file(boundary_path: Path, mask: np.ndarray) -> BoundaryData:
    data = np.load(boundary_path, allow_pickle=False)
    upper = np.asarray(data["upper_y"], dtype=np.float32)
    lower = np.asarray(data["lower_y"], dtype=np.float32)
    valid = np.asarray(data["valid_columns"], dtype=np.uint8) > 0
    data.close()
    mask_top, mask_bottom, mask_valid = boundary_from_mask(mask)
    width = mask.shape[1]
    if upper.size != width or lower.size != width or valid.size != width:
        upper = cv2.resize(upper.reshape(1, -1), (width, 1), interpolation=cv2.INTER_LINEAR).reshape(-1)
        lower = cv2.resize(lower.reshape(1, -1), (width, 1), interpolation=cv2.INTER_LINEAR).reshape(-1)
        valid = cv2.resize(valid.astype(np.uint8).reshape(1, -1), (width, 1), interpolation=cv2.INTER_NEAREST).reshape(-1) > 0
    finite = np.isfinite(upper) & np.isfinite(lower)
    valid = valid & finite & (lower - upper >= 4.0) & (lower - upper <= 260.0)
    # Interpolate only from columns explicitly marked valid by the decoder.
    # The binary PNG is retained for auditing, but it must not silently turn an
    # invalid terminal prediction into a registration landmark. If an entire
    # trajectory is invalid, the mask extrema are the last-resort fallback.
    columns = np.arange(width)
    valid_idx = np.flatnonzero(valid)
    if valid_idx.size >= 2:
        upper = np.interp(columns, valid_idx, upper[valid_idx]).astype(np.float32)
        lower = np.interp(columns, valid_idx, lower[valid_idx]).astype(np.float32)
    else:
        upper = mask_top.astype(np.float32)
        lower = mask_bottom.astype(np.float32)
    upper = np.clip(upper, 0.0, mask.shape[0] - 2.0)
    lower = np.clip(lower, upper + 4.0, mask.shape[0] - 1.0)
    upper = median_filter(upper.astype(np.float32), size=odd_window(width, 7), mode="nearest")
    lower = median_filter(lower.astype(np.float32), size=odd_window(width, 7), mode="nearest")
    if width >= 11:
        window = odd_window(width, 31)
        upper = savgol_filter(upper, window_length=window, polyorder=2, mode="interp").astype(np.float32)
        lower = savgol_filter(lower, window_length=window, polyorder=2, mode="interp").astype(np.float32)
    upper = gaussian_filter1d(upper, sigma=1.1, mode="nearest").astype(np.float32)
    lower = gaussian_filter1d(lower, sigma=1.1, mode="nearest").astype(np.float32)
    lower = np.maximum(lower, upper + 4.0)
    lower = np.minimum(lower, mask.shape[0] - 1.0)
    thickness = lower - upper
    slopes = np.r_[np.diff(upper), np.diff(lower)]
    quality = {
        "valid_fraction": float(valid.mean()),
        "finite_fraction": float(finite.mean()),
        "thickness_median_px": float(np.median(thickness)),
        "thickness_p05_px": float(np.percentile(thickness, 5)),
        "thickness_p95_px": float(np.percentile(thickness, 95)),
        "p99_abs_slope_px": float(np.percentile(np.abs(slopes), 99)),
        "mask_occupied_fraction": float(mask.any(axis=0).mean()),
    }
    return BoundaryData(upper=upper, lower=lower, valid=valid, quality=quality)
