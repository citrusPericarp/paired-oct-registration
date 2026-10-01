import numpy as np

from paired_oct_registration import registration_core as core
from paired_oct_registration.registration_utils import BoundaryData


def test_zero_coverage_is_recorded_as_no_deformation_even_without_gate(monkeypatch):
    monkeypatch.setattr(core, "D_OCT_SAFE_MIN_COMMON_FRACTION", 0.0)
    image = np.zeros((96, 128), dtype=np.float32)
    boundary = BoundaryData(np.full(128, 20.0), np.full(128, 70.0), np.ones(128, bool), {})
    field, stats, nodes = core.build_regularized_d_oct_field(
        image, image, boundary, np.zeros(128, bool), minimum_valid_fraction=0.0)
    assert stats["d_oct_enabled"] == 0.0
    assert stats["d_oct_disabled_no_common_columns"] == 1.0
    assert stats["d_oct_disabled_low_common_fraction"] == 0.0
    assert not np.any(nodes) and not np.any(field)


def test_low_nonempty_support_runs_with_legacy_threshold():
    image = np.random.default_rng(3).random((96, 128), dtype=np.float32)
    boundary = BoundaryData(np.full(128, 20.0), np.full(128, 70.0), np.ones(128, bool), {})
    support = np.zeros(128, bool)
    support[50:75] = True
    field, stats, nodes = core.build_regularized_d_oct_field(
        image, image, boundary, support, minimum_valid_fraction=0.9)
    assert 0 < stats["d_oct_common_domain_fraction"] < 0.75
    assert stats["d_oct_enabled"] == 1.0
    assert stats["d_oct_valid_columns"] > 0
    assert not np.any(nodes[:, ~support])
    assert not np.any(field[:, ~support])
    assert not np.any(nodes[[0, -1]])
