from __future__ import annotations

import numpy as np

from paired_oct_registration.pipeline import _needs_phase_translation_retry


def test_retries_when_fovea_phase_conflict_causes_two_sided_scale_saturation() -> None:
    translation = {"fovea_tx": 81.0, "translation_source": "fovea_safe"}
    phase = {"phase_forward_tx": 7.59}
    scale = np.ones(768, dtype=np.float32)
    scale[:64] = 0.80
    scale[-64:] = 1.25

    assert _needs_phase_translation_retry(translation, phase, scale)


def test_keeps_fovea_translation_without_two_sided_scale_saturation() -> None:
    translation = {"fovea_tx": 76.0, "translation_source": "fovea_safe"}
    phase = {"phase_forward_tx": 9.34}
    scale = np.linspace(0.95, 1.17, 768, dtype=np.float32)

    assert not _needs_phase_translation_retry(translation, phase, scale)
