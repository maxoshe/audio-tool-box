from audio_tool_box.audio_data import AudioData
from audio_tool_box.processing.gain import (
    apply_gain,
    normalize_to_target,
    apply_fade,
    FlatLineError,
    ClippingError,
)
import numpy as np

import pytest
from audio_tool_box.util import convert_db_to_factor


TOLERANCE = 1e-3


@pytest.mark.parametrize("gain_db", [-6, 3, 5.39])
def test_apply_gain(audio_test_signal: AudioData, gain_db: float):
    result = apply_gain(audio_test_signal, gain_db)
    expected = audio_test_signal.data * convert_db_to_factor(gain_db)
    np.testing.assert_allclose(result.data, expected, rtol=TOLERANCE)


@pytest.mark.parametrize("target_db", [-10, -3, 0])
def test_normalize_to_target(audio_test_signal: AudioData, target_db: float):
    result = normalize_to_target(audio_test_signal, target_db)
    expected_peak = convert_db_to_factor(target_db)
    assert np.isclose(result.get_peak(), expected_peak, rtol=TOLERANCE)


def test_normalize_to_target_clipping_raises(audio_test_signal: AudioData):
    target_db = 2
    with pytest.raises(ClippingError):
        normalize_to_target(audio_test_signal, target_db)


def test_normalize_to_target_flat_line_raises(audio_test_signal: AudioData):
    flat = AudioData(
        sample_rate=audio_test_signal.sample_rate,
        data=np.zeros_like(audio_test_signal.data),
    )
    with pytest.raises(FlatLineError):
        normalize_to_target(flat, -3)


def test_apply_fade(audio_test_signal: AudioData):
    fade_ms = 50
    result = apply_fade(audio_test_signal, fade_ms)

    fade_len = round((fade_ms / 1000) * audio_test_signal.sample_rate)
    assert np.isclose(result.data[0], 0, atol=TOLERANCE)
    assert np.isclose(result.data[-1], 0, atol=TOLERANCE)
    assert np.isclose(
        result.data[fade_len:-fade_len].mean(),
        audio_test_signal.data.mean(),
        rtol=TOLERANCE,
    )


def test_apply_fade_too_long_raises(audio_test_signal: AudioData):
    fade_ms = int(audio_test_signal.get_duration_s() * 1000)
    with pytest.raises(ValueError):
        apply_fade(audio_test_signal, fade_ms)
