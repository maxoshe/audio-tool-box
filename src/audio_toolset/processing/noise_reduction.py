import numpy as np
from scipy.fftpack import fft, ifft

from audio_toolset.audio_data import AudioData
from audio_toolset.util import convert_db_to_factor


def apply_spectral_gating(
    audio_data: AudioData, noise_threshold_db: float = -50, attenuation_db: float = -1
) -> AudioData:
    new_data = audio_data.get_copy()

    noise_threshold = convert_db_to_factor(noise_threshold_db)
    attenuation = convert_db_to_factor(attenuation_db)

    signal_fft = fft(new_data.data)
    sample_size = new_data.get_number_of_samples()
    signal_fft[abs(signal_fft) * (2 / sample_size) < noise_threshold] *= attenuation
    new_data.data = np.real(ifft(signal_fft))
    return new_data
