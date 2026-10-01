import numpy as np

from paired_oct_registration import pipeline
from paired_oct_registration.registration_utils import BoundaryData


def test_d_oct_domain_follows_shifted_source_columns(tmp_path, monkeypatch):
    shape = (496, 768)
    valid = np.ones(shape[1], bool)
    boundary = BoundaryData(np.full(768, 180.0), np.full(768, 270.0), valid, {"valid_fraction": 1.0})
    monkeypatch.setattr(pipeline, "load_image", lambda path: np.zeros(shape, np.float32))
    monkeypatch.setattr(pipeline, "load_mask", lambda path, shape: np.zeros(shape, bool))
    monkeypatch.setattr(pipeline, "load_boundary_file", lambda *args: BoundaryData(
        boundary.upper.copy(), boundary.lower.copy(), valid.copy(), boundary.quality.copy()))
    observations = iter([
        {"registration_valid": valid.copy(), "score": np.ones(768, np.float32)},
        {"registration_valid": valid.copy(), "score": ((np.arange(768) >= 200) & (np.arange(768) <= 600)).astype(np.float32)},
    ])
    monkeypatch.setattr(pipeline, "estimate_column_observability", lambda *a, **k: next(observations))
    monkeypatch.setattr(pipeline, "choose_translation", lambda *a, **k: {
        "selected_forward_tx": 16.0, "fovea_tx": 16.0, "translation_source": "fovea_safe"})
    captured = {}

    def local_field(a, ao, b, columns, **kwargs):
        captured["columns"] = columns.copy()
        return np.zeros(shape, np.float32), {"d_oct_enabled": 0.0}, np.zeros((64, 768), np.float32)

    monkeypatch.setattr(pipeline, "build_regularized_d_oct_field", local_field)
    paths = {name: tmp_path / "unused" for name in ["preoperative_image_path", "postoperative_image_path",
             "preoperative_mask_path", "postoperative_mask_path", "preoperative_boundary_path", "postoperative_boundary_path"]}
    result = pipeline.register_pair(sample_id="test", **paths, output_root=tmp_path)
    expected = (np.arange(768) >= 216) & (np.arange(768) <= 616)
    np.testing.assert_array_equal(captured["columns"], expected)
    with np.load(result["deformation"]) as archive:
        np.testing.assert_array_equal(archive["local_geometry_domain"] > 0, expected)
