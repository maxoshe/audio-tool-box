from pathlib import Path

import numpy as np

from audio_toolset.audio_data import AudioData
from audio_toolset.channel import Channel


def test_channel_read_write(tmp_path: Path, test_tone: AudioData):
    test_channel = Channel(source=test_tone)
    assert test_channel.audio_data.get_duration_s() > 0

    file_path = tmp_path / "test.wav"
    test_channel.write(file_path)

    test_channel = Channel(source=file_path)
    assert test_channel.audio_data.get_duration_s() > 0


def test_channel_processing_smoke_test(test_tone: AudioData):
    test_channel = Channel(source=test_tone)
    signal_duration = test_channel.audio_data.get_duration_s()

    test_channel.compressor()
    test_channel.eq_band()
    test_channel.fade()
    test_channel.gain()
    test_channel.highpass()
    test_channel.limiter()
    test_channel.lowpass()
    test_channel.noise_reduction()
    test_channel.normalize()
    test_channel.soft_clipping()

    assert test_channel.audio_data.get_duration_s() == signal_duration
    assert np.any(test_channel.audio_data.data != 0)


def test_channel_chain_processing(test_tone: AudioData):
    test_channel = Channel(source=test_tone)
    test_channel.compressor()
    test_channel.eq_band()
    test_channel.fade()
    test_channel.gain()
    test_channel.highpass()
    test_channel.limiter()
    test_channel.lowpass()
    test_channel.noise_reduction()
    test_channel.normalize()
    test_channel.soft_clipping()

    test_channel_chain = (
        Channel(source=test_tone)
        .compressor()
        .eq_band()
        .fade()
        .gain()
        .highpass()
        .limiter()
        .lowpass()
        .noise_reduction()
        .normalize()
        .soft_clipping()
    )

    assert np.allclose(
        test_channel_chain.audio_data.data,
        test_channel.audio_data.data,
        rtol=1e-6,
        atol=1e-8,
    )
