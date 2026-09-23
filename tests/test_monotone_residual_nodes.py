from __future__ import annotations

import numpy as np

from paired_oct_registration.registration_core import enforce_monotone_residual_nodes
from paired_oct_registration.registration_utils import BoundaryData


def test_residual_node_projection_preserves_anchors_and_positive_steps() -> None:
    node_count = 64
    width = 3
    upper = np.full(width, 100.0, dtype=np.float32)
    lower = np.full(width, 220.0, dtype=np.float32)
    boundary = BoundaryData(
        upper=upper,
        lower=lower,
        valid=np.ones(width, dtype=bool),
        quality={},
    )
    nodes = np.zeros((node_count, width), dtype=np.float32)
    nodes[30, 1] = 8.0
    nodes[31, 1] = -8.0

    corrected = enforce_monotone_residual_nodes(nodes, boundary)
    base = upper[None, :] + np.linspace(0.0, 1.0, node_count, dtype=np.float32)[:, None] * (
        lower - upper
    )[None, :]
    warped = base + corrected

    assert np.all(corrected[0] == 0.0)
    assert np.all(corrected[-1] == 0.0)
    assert np.all(np.diff(warped, axis=0) > 0.0)
    assert float(np.max(np.abs(corrected))) <= 12.0

