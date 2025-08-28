from audio_tool_box.audio_data import AudioData
from audio_tool_box.processing.dynamics import (
    apply_compressor,
    apply_limiter,
    apply_cubic_non_linearity,
)


def test_compressor_reduces_peaks(test_tone: AudioData):
    result = apply_compressor(test_tone)
    assert result.get_peak() < test_tone.get_peak()


def test_limiter_reduces_peaks(test_tone: AudioData):
    result = apply_limiter(test_tone)
    assert result.get_peak() < test_tone.get_peak()


def test_cubic_non_linearity_reduces_peaks(test_tone: AudioData):
    result = apply_cubic_non_linearity(test_tone)
    assert result.get_peak() < test_tone.get_peak()
