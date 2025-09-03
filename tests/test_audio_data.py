from pathlib import Path

import numpy as np

from audio_toolset.audio_data import AudioData, join_to_stereo
from audio_toolset.constants.frequencies import FS_44100HZ
from tests.conftest import TEST_SIGNAL_DURATION_S


def test_audio_data_write_to_file(tmp_path: Path, test_tone: AudioData) -> None:
    file_path = tmp_path / "test.wav"
    test_tone.write_to_file(file_path)
    data = file_path.read_bytes()
    assert len(data) > 0


def test_audio_data_read_from_file(tmp_path: Path, test_tone: AudioData) -> None:
    file_path = tmp_path / "test.wav"
    test_tone.write_to_file(file_path)
    audio_data = AudioData.read_from_file(file_path)
    assert audio_data.info.duration == TEST_SIGNAL_DURATION_S
    assert audio_data.info.samplerate == FS_44100HZ
    assert len(audio_data.data) == TEST_SIGNAL_DURATION_S * FS_44100HZ
    assert test_tone.is_mono() and audio_data.is_mono()


def test_stereo_handling(test_tone: AudioData) -> None:
    stereo_signal = join_to_stereo(
        left_channel=test_tone,
        right_channel=test_tone,
    )
    left_channel, right_channel = stereo_signal.split_to_mono()
    assert np.array_equal(left_channel.data, test_tone.data)
    assert np.array_equal(right_channel.data, test_tone.data)
    assert np.array_equal(stereo_signal.sum_to_mono().data, test_tone.data)
