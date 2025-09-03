import numpy as np
import pytest
from scipy.signal import welch
from scipy.stats import linregress

from audio_toolset.audio_data import AudioData
from audio_toolset.processing.filters import ButterFilterType, apply_butterworth_filter
from audio_toolset.util import convert_power_to_db


@pytest.mark.parametrize("filter_frequency", [50, 800, 1000, 8000])
def test_lowpass_filter_reduces_target_frequency(
    white_noise: AudioData, filter_frequency: int
):
    processed_signal = apply_butterworth_filter(
        audio_data=white_noise,
        filter_type=ButterFilterType.LOWPASS,
        cutoff_frequency=filter_frequency,
    )

    f, psd_pre = welch(white_noise.data, fs=white_noise.sample_rate)
    _, psd_post = welch(processed_signal.data, fs=processed_signal.sample_rate)

    mask = f > filter_frequency
    assert np.all(psd_post[mask] < psd_pre[mask])


@pytest.mark.parametrize("db_per_octave", [-6, -12, -18, -24])
def test_lowpass_filter_db_per_octave(white_noise: AudioData, db_per_octave: int):
    filter_frequency = 1000
    processed_signal = apply_butterworth_filter(
        audio_data=white_noise,
        filter_type=ButterFilterType.LOWPASS,
        cutoff_frequency=filter_frequency,
        db_per_octave=db_per_octave,
    )

    f, psd_post = welch(processed_signal.data, fs=processed_signal.sample_rate)
    psd_db = convert_power_to_db(psd_post)

    test_octave_start = filter_frequency * 2
    test_octave_end = filter_frequency * 4
    octave_mask = (f > test_octave_start) & (f < test_octave_end)

    log_freq = np.log2(f[octave_mask])
    slope_db_per_octave, _, _, _, _ = linregress(x=log_freq, y=psd_db[octave_mask])
    assert np.isclose(slope_db_per_octave, db_per_octave, atol=2)


@pytest.mark.parametrize("filter_frequency", [50, 800, 1000, 8000])
def test_highpass_filter_reduces_target_frequency(
    white_noise: AudioData, filter_frequency: int
):
    processed_signal = apply_butterworth_filter(
        audio_data=white_noise,
        filter_type=ButterFilterType.HIGHPASS,
        cutoff_frequency=filter_frequency,
    )

    f, psd_pre = welch(white_noise.data, fs=white_noise.sample_rate)
    _, psd_post = welch(processed_signal.data, fs=processed_signal.sample_rate)

    mask = f < filter_frequency
    assert np.all(psd_post[mask] < psd_pre[mask])


@pytest.mark.parametrize("db_per_octave", [-6, -12, -18, -24])
def test_highpass_filter_db_per_octave(white_noise: AudioData, db_per_octave: int):
    filter_frequency = 10000
    processed_signal = apply_butterworth_filter(
        audio_data=white_noise,
        filter_type=ButterFilterType.HIGHPASS,
        cutoff_frequency=filter_frequency,
        db_per_octave=db_per_octave,
    )

    f, psd_post = welch(processed_signal.data, fs=processed_signal.sample_rate)
    psd_db = convert_power_to_db(psd_post)

    test_octave_start = filter_frequency / 4
    test_octave_end = filter_frequency / 2
    octave_mask = (f > test_octave_start) & (f < test_octave_end)

    log_freq = np.log2(f[octave_mask])
    slope_db_per_octave, _, _, _, _ = linregress(x=log_freq, y=psd_db[octave_mask])
    assert np.isclose(slope_db_per_octave, abs(db_per_octave), atol=2)
