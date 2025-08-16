from pathlib import Path
from audio_tool_box.audio_data import AudioData, join_to_stereo
from audio_tool_box.oscillators import generate_sine_wave
from audio_tool_box.constants.frequencies import FS_44100HZ, TEST_TONE_1000HZ
import pytest
import numpy as np

TEST_SIGNAL_DURATION_S = 10
TEST_SIGNAL_AMPLITUDE_DBFS = -3.0


@pytest.fixture
def audio_test_signal() -> AudioData:
    return AudioData(
        sample_rate=FS_44100HZ,
        data=generate_sine_wave(
            sample_rate_hz=FS_44100HZ,
            duration_s=TEST_SIGNAL_DURATION_S,
            frequency=TEST_TONE_1000HZ,
            amplitude_dbfs=TEST_SIGNAL_AMPLITUDE_DBFS,
        ),
    )


def test_audio_data_write_to_file(tmp_path: Path, audio_test_signal: AudioData) -> None:
    file_path = tmp_path / "test.wav"
    audio_test_signal.write_to_file(file_path)
    data = file_path.read_bytes()
    assert len(data) > 0


def test_audio_data_read_from_file(
    tmp_path: Path, audio_test_signal: AudioData
) -> None:
    file_path = tmp_path / "test.wav"
    audio_test_signal.write_to_file(file_path)
    audio_data = AudioData.read_from_file(file_path)
    assert audio_data.info.duration == TEST_SIGNAL_DURATION_S
    assert audio_data.info.samplerate == FS_44100HZ
    assert len(audio_data.data) == TEST_SIGNAL_DURATION_S * FS_44100HZ
    assert audio_test_signal.is_mono() and audio_data.is_mono()


def test_stereo_handling(audio_test_signal: AudioData) -> None:
    stereo_signal = join_to_stereo(
        left_channel=audio_test_signal,
        right_channel=audio_test_signal,
    )
    left_channel, right_channel = stereo_signal.split_to_mono()
    assert np.array_equal(left_channel.data, audio_test_signal.data)
    assert np.array_equal(right_channel.data, audio_test_signal.data)
    assert np.array_equal(stereo_signal.sum_to_mono().data, audio_test_signal.data)
