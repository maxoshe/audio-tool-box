import numpy as np
from scipy.signal import welch

from audio_toolset.audio_data import AudioData
from audio_toolset.constants.frequencies import TEST_TONE_1000HZ
from audio_toolset.oscillators import generate_white_noise
from audio_toolset.processing.noise_reduction import apply_spectral_gating


def test_spectral_gating(test_tone: AudioData):
    noise_level_db = -20
    test_tone.data += generate_white_noise(
        sample_rate_hz=test_tone.sample_rate,
        duration_s=test_tone.get_duration_s(),
        amplitude_dbfs=noise_level_db,
    )

    result = apply_spectral_gating(
        test_tone,
        noise_threshold_db=-10,
        attenuation_db=-10,
    )

    f, psd_pre = welch(test_tone.data, fs=test_tone.sample_rate)
    _, psd_post = welch(result.data, fs=result.sample_rate)

    mask = f != TEST_TONE_1000HZ

    assert np.all(psd_post[mask] < psd_pre[mask])
