import numpy as np

from audio_toolset.audio_data import AudioData
from audio_toolset.constants.math import MS_TO_S
from audio_toolset.util import convert_db_to_factor


class FlatLineError(Exception): ...


class ClippingError(Exception): ...


def apply_gain(audio_data: AudioData, gain_db: float) -> AudioData:
    new_data = audio_data.get_copy()

    if gain_db == 0:
        return new_data

    factor = convert_db_to_factor(gain_db)
    new_data.data *= factor
    return new_data


def normalize_to_target(audio_data: AudioData, target_db: float) -> AudioData:
    if target_db > 0:
        raise ClippingError("the provided gain value will result in a clipped signal")

    new_data = audio_data.get_copy()

    target = convert_db_to_factor(target_db)
    peak = new_data.get_peak()

    if peak == 0:
        raise FlatLineError("Audio data represents a flat line")

    factor = target / peak

    if factor == 1:
        return new_data

    new_data.data *= factor
    return new_data


def apply_fade(audio_data: AudioData, fade_duration_ms: int) -> AudioData:
    if fade_duration_ms * MS_TO_S * 2 > audio_data.get_duration_s():
        raise ValueError("Fade duration is longer than audio")

    new_data = audio_data.get_copy()

    fade_duration_s = fade_duration_ms * MS_TO_S
    fade_sample_length = round(fade_duration_s * new_data.sample_rate)
    total_samples = new_data.get_number_of_samples()

    fade_in_factor = np.linspace(0, 1, fade_sample_length)
    fade_out_factor = np.linspace(1, 0, fade_sample_length)
    no_scale_factor = np.ones(total_samples - 2 * fade_sample_length)
    factor = np.concatenate([fade_in_factor, no_scale_factor, fade_out_factor])
    new_data.data *= factor
    return new_data
