import numpy as np
import pytest

from audio_toolset.audio_data import AudioData
from audio_toolset.processing.gain import (
    ClippingError,
    FlatLineError,
    apply_fade,
    apply_gain,
    normalize_to_target,
)
from audio_toolset.util import convert_db_to_factor

TOLERANCE = 1e-3


@pytest.mark.parametrize("gain_db", [-6, 3, 5.39])
def test_apply_gain(test_tone: AudioData, gain_db: float):
    result = apply_gain(test_tone, gain_db)
    expected = test_tone.data * convert_db_to_factor(gain_db)
    np.testing.assert_allclose(result.data, expected, rtol=TOLERANCE)


@pytest.mark.parametrize("target_db", [-10, -3, 0])
def test_normalize_to_target(test_tone: AudioData, target_db: float):
    result = normalize_to_target(test_tone, target_db)
    expected_peak = convert_db_to_factor(target_db)
    assert np.isclose(result.get_peak(), expected_peak, rtol=TOLERANCE)


def test_normalize_to_target_clipping_raises(test_tone: AudioData):
    target_db = 2
    with pytest.raises(ClippingError):
        normalize_to_target(test_tone, target_db)


def test_normalize_to_target_flat_line_raises(test_tone: AudioData):
    flat = AudioData(
        sample_rate=test_tone.sample_rate,
        data=np.zeros_like(test_tone.data),
    )
    with pytest.raises(FlatLineError):
        normalize_to_target(flat, -3)


def test_apply_fade(test_tone: AudioData):
    fade_ms = 50
    result = apply_fade(test_tone, fade_ms)

    fade_len = round((fade_ms / 1000) * test_tone.sample_rate)
    assert np.isclose(result.data[0], 0, atol=TOLERANCE)
    assert np.isclose(result.data[-1], 0, atol=TOLERANCE)
    assert np.isclose(
        result.data[fade_len:-fade_len].mean(),
        test_tone.data.mean(),
        rtol=TOLERANCE,
    )


def test_apply_fade_too_long_raises(test_tone: AudioData):
    fade_ms = int(test_tone.get_duration_s() * 1000)
    with pytest.raises(ValueError):
        apply_fade(test_tone, fade_ms)
