from audio_tool_box.oscillators import generate_sine_wave, generate_white_noise

import pytest
from audio_tool_box.audio_data import AudioData
from audio_tool_box.constants.frequencies import FS_44100HZ, TEST_TONE_1000HZ

TEST_SIGNAL_DURATION_S = 10
TEST_SIGNAL_AMPLITUDE_DBFS = -3.0


@pytest.fixture
def test_tone() -> AudioData:
    return AudioData(
        sample_rate=FS_44100HZ,
        data=generate_sine_wave(
            sample_rate_hz=FS_44100HZ,
            duration_s=TEST_SIGNAL_DURATION_S,
            frequency=TEST_TONE_1000HZ,
            amplitude_dbfs=TEST_SIGNAL_AMPLITUDE_DBFS,
        ),
    )


@pytest.fixture
def white_noise() -> AudioData:
    return AudioData(
        sample_rate=FS_44100HZ,
        data=generate_white_noise(
            sample_rate_hz=FS_44100HZ,
            duration_s=TEST_SIGNAL_DURATION_S,
            amplitude_dbfs=TEST_SIGNAL_AMPLITUDE_DBFS,
        ),
    )
