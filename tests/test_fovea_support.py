import numpy as np

from paired_oct_registration.registration_core import choose_translation, fovea_candidate
from paired_oct_registration.registration_utils import BoundaryData


def boundary(center):
    x = np.arange(768, dtype=np.float32)
    thickness = 90.0 - 35.0 * np.exp(-0.5 * ((x - center) / 20.0) ** 2)
    return BoundaryData(200.0 - thickness, np.full(768, 200.0, np.float32),
                        np.ones(768, bool), {"valid_fraction": 1.0})


def test_unsupported_mask_dip_does_not_drive_horizontal_translation():
    a_geometry = np.arange(768) < 171
    phase = {"phase_forward_tx": -37.5, "phase_forward_ty": -7.4, "phase_response": 0.049}
    result = choose_translation(boundary(438), boundary(376), phase,
                                a_geometry_valid=a_geometry, b_geometry_valid=np.ones(768, bool))
    assert result["fovea_tx"] == 62.0
    assert result["selected_forward_tx"] == 0.0
    assert result["fovea_A_confidence"] == 0.0
    assert result["translation_source"] == "zero_after_unsupported_fovea"


def test_supported_fovea_retains_known_horizontal_translation():
    phase = {"phase_forward_tx": -37.5, "phase_forward_ty": -7.4, "phase_response": 0.049}
    result = choose_translation(boundary(392), boundary(376), phase,
                                a_geometry_valid=np.ones(768, bool), b_geometry_valid=np.ones(768, bool))
    assert result["selected_forward_tx"] == 16.0


def test_missing_central_observation_cannot_be_a_confident_fovea():
    valid = np.arange(768) < 171
    assert fovea_candidate(boundary(438), valid)["confidence"] == 0.0


def test_supported_phase_can_replace_unsupported_fovea():
    phase = {"phase_forward_tx": 5.0, "phase_forward_ty": 3.0, "phase_response": 0.4}
    result = choose_translation(boundary(438), boundary(376), phase,
                                a_geometry_valid=np.zeros(768, bool), b_geometry_valid=np.ones(768, bool))
    assert result["selected_forward_tx"] == 5.0
    assert result["translation_source"] == "phase_after_unsupported_fovea"
