"""Reconstruct the archived sampling map without re-estimating any boundary."""
from __future__ import annotations

from typing import Mapping
from pathlib import Path

import numpy as np


def sampling_map(parameters: Mapping[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    height, width = (int(v) for v in parameters["image_shape"])
    nodes = np.asarray(parameters["residual_nodes"], dtype=np.float32)
    upper = np.asarray(parameters["residual_target_upper"], dtype=np.float32)
    lower = np.asarray(parameters["residual_target_lower"], dtype=np.float32)
    field = np.zeros((height, width), dtype=np.float32)
    y = np.arange(height, dtype=np.float32)
    node_axis = np.linspace(0.0, 1.0, nodes.shape[0], dtype=np.float32)
    for x in range(width):
        top, bottom = float(upper[x]), float(lower[x])
        if bottom <= top + 4.0:
            continue
        positions = top + node_axis * (bottom - top)
        field[:, x] = np.interp(y, positions, nodes[:, x], left=0.0, right=0.0).astype(np.float32)
        field[:, x] = np.clip(field[:, x], -12.0, 12.0)
    scale = np.asarray(parameters["initial_axial_scale"], dtype=np.float32)
    source_top = np.asarray(parameters["initial_postoperative_top"], dtype=np.float32)
    target_top = np.asarray(parameters["initial_target_top"], dtype=np.float32)
    source_y = source_top[None, :] + scale[None, :] * (y[:, None] + field - target_top[None, :])
    source_x = np.broadcast_to(np.asarray(parameters["source_x_line"], dtype=np.float32)[None, :], (height, width)).copy()
    return source_x, source_y.astype(np.float32)


def reconstruct_pair(dataset: Path, row: Mapping[str, str], output: Path) -> None:
    """Recreate the five image/boundary products from one saved transform."""
    import cv2
    from .registration_utils import load_image, load_mask, boundary_from_mask, save_gray, save_binary

    with np.load(dataset / row['deformation'], allow_pickle=False) as archive:
        parameters = {key: archive[key].copy() for key in archive.files}
    source_x, source_y = sampling_map(parameters)
    source_image = load_image(dataset / row['real_B'])
    height, width = source_image.shape
    valid = (np.isfinite(source_x) & np.isfinite(source_y)
             & (source_x >= 0) & (source_x <= width - 1)
             & (source_y >= 0) & (source_y <= height - 1))
    image = cv2.remap(source_image, source_x, source_y, cv2.INTER_CUBIC,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0.)
    image[~valid] = 0.
    mask = load_mask(dataset / row['mask_B'])
    with np.load(dataset / row['boundary_B'], allow_pickle=False) as boundary:
        mask &= boundary['valid_columns'][None, :] > 0
    aligned_mask = cv2.remap(mask.astype(np.float32), source_x, source_y, cv2.INTER_NEAREST,
                            borderMode=cv2.BORDER_CONSTANT, borderValue=0.) > .5
    aligned_mask &= parameters['registration_domain'][None, :] > 0
    evaluation = aligned_mask & load_mask(dataset / row['mask_A'])
    sample = row['sample_id']
    save_gray(output / 'aligned_B' / f'{sample}.png', image)
    for name, value in [('aligned_mask_B', aligned_mask), ('valid_mask', valid), ('evaluation_mask', evaluation)]:
        save_binary(output / name / f'{sample}.png', value)
    upper, lower, columns = boundary_from_mask(aligned_mask)
    path = output / 'aligned_boundary_B' / f'{sample}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, upper_y=upper.astype(np.float32), lower_y=lower.astype(np.float32),
                        valid_columns=columns.astype(np.uint8))


def main() -> None:
    import argparse
    import csv
    parser = argparse.ArgumentParser(description='Reconstruct corrected-release products from saved transforms.')
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--sample-id', action='append')
    args = parser.parse_args()
    with (args.dataset / 'pairs.csv').open(encoding='utf-8-sig', newline='') as handle:
        rows = list(csv.DictReader(handle))
    if args.sample_id:
        rows = [row for row in rows if row['sample_id'] in args.sample_id]
    for row in rows:
        reconstruct_pair(args.dataset, row, args.output)
    print(f'Reconstructed {len(rows)} records')


if __name__ == '__main__':
    main()
