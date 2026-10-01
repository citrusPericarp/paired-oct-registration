"""Local boundary trends must not deform unobserved image regions."""
import numpy as np

from paired_oct_registration.registration_core import build_a_oct_map
from paired_oct_registration.registration_utils import BoundaryData


def boundaries(valid_start=94, valid_end=170):
    x = np.arange(768, dtype=np.float32)
    valid = (x >= valid_start) & (x <= valid_end)
    upper_a = np.full(768, 180, dtype=np.float32)
    upper_b = 200 + .4 * (x - 132)
    a = BoundaryData(upper_a, upper_a + 100, valid.copy(), {})
    b = BoundaryData(upper_b, upper_b + 100, valid.copy(), {})
    return a, b


def test_short_trusted_interval_does_not_extrapolate_its_slope_across_image():
    a, b = boundaries()
    _, source_y, scale, _, _, _ = build_a_oct_map(a, b, 0, (496, 768))
    # The observed interval is centred on a 20-pixel axial translation.
    # Beyond its support, the transform should retain that global translation.
    np.testing.assert_allclose(source_y[250, 250:], 270, atol=1e-3)
    np.testing.assert_allclose(scale[250:], 1, atol=1e-5)
    assert np.min(np.diff(source_y, axis=0)) > 0


def test_global_only_mode_has_no_lateral_local_geometry():
    a, b = boundaries(180, 580)
    _, source_y, _, _, _, _ = build_a_oct_map(a, b, 0, (496, 768), local_geometry=False)
    assert np.ptp(source_y[250]) < 1e-3


def test_broad_supported_tilt_is_preserved_outside_interval():
    a, b = boundaries(214, 720)
    _, source_y, _, _, _, _ = build_a_oct_map(a, b, 0, (496, 768))
    # A true image tilt requires lateral variation, including near the edge.
    expected = 250 + 20 + 0.4 * (np.arange(768) - 132)
    np.testing.assert_allclose(source_y[250, 48:-48], expected[48:-48], atol=0.03)


def test_terminal_mask_closure_does_not_create_a_growing_edge_bend():
    x = np.arange(768, dtype=np.float32)
    valid = (x >= 200) & (x <= 600)
    a_top = np.full(768, 180.0, dtype=np.float32)
    b_top = a_top + 20.0 + np.clip(x - 570.0, 0.0, 30.0) * 0.4
    a = BoundaryData(a_top, a_top + 100.0, valid.copy(), {})
    b = BoundaryData(b_top, b_top + 100.0, valid.copy(), {})
    _, source_y, _, _, _, _ = build_a_oct_map(a, b, 0, (496, 768))
    np.testing.assert_allclose(source_y[250, 670:], 270.0, atol=0.02)


def test_known_translation_is_preserved_with_partial_coverage():
    x = np.arange(768, dtype=np.float32)
    upper = 170 + 20 * np.cos(x / 120)
    valid = (x >= 200) & (x <= 650)
    a = BoundaryData(upper, upper + 90, valid.copy(), {})
    b = BoundaryData(upper + 12, upper + 102, valid.copy(), {})
    _, source_y, _, _, _, _ = build_a_oct_map(a, b, 0, (496, 768))
    expected = np.broadcast_to(np.arange(496, dtype=np.float32)[:, None] + 12, source_y.shape)
    np.testing.assert_allclose(source_y, expected, atol=1e-4)
