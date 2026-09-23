import numpy as np

from paired_oct_registration.reconstruction import sampling_map


def test_archive_roundtrip_uses_saved_anchors_not_raw_boundary(tmp_path):
    # The processed anchor intentionally differs from the original model edge.
    parameters = dict(
        image_shape=np.array([9, 2]), source_x_line=np.array([-0.5, 0.5], dtype=np.float32),
        residual_nodes=np.zeros((4, 2), dtype=np.float32),
        residual_target_upper=np.array([1, 1], dtype=np.float32),
        residual_target_lower=np.array([7, 7], dtype=np.float32),
        initial_axial_scale=np.array([0.8, 1.25], dtype=np.float32),
        initial_postoperative_top=np.array([2, 3], dtype=np.float32),
        initial_target_top=np.array([1.125, 1.375], dtype=np.float32),
    )
    path = tmp_path / 'deformation.npz'
    np.savez_compressed(path, **parameters)
    with np.load(path, allow_pickle=False) as stored:
        x, y = sampling_map(stored)
    expected = parameters['initial_postoperative_top'][None, :] + parameters['initial_axial_scale'][None, :] * (
        np.arange(9, dtype=np.float32)[:, None] - parameters['initial_target_top'][None, :]
    )
    np.testing.assert_array_equal(y, expected)
    np.testing.assert_array_equal(x[0], [-0.5, 0.5])
    assert np.all(np.diff(y, axis=0) > 0)


def test_dtw_does_not_change_with_optional_acceleration(monkeypatch):
    from paired_oct_registration import registration_core as core
    def unexpected(*args):
        raise AssertionError('optional accelerated path must not control reference output')
    monkeypatch.setattr(core, '_dtw_monotone_numba', unexpected)
    values = np.arange(8, dtype=np.float32)
    mapping, _, _ = core.dtw_monotone(values, values)
    np.testing.assert_array_equal(mapping, values)
