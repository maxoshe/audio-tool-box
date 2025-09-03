from enum import Enum
from typing import Literal

import numpy as np
from scipy.signal import butter, sosfilt, tf2sos

from audio_toolset.audio_data import AudioData
from audio_toolset.constants.math import PI
from audio_toolset.plots import get_bode_plot


class FilterError(Exception): ...


class ButterFilterType(Enum):
    LOWPASS = "lowpass"
    HIGHPASS = "highpass"


SECOND_ORDER_SECTIONS = "sos"


def db_per_octave_to_filter_order(db_per_octave: Literal[6, 12, 18, 24]) -> int:
    return int(abs(db_per_octave) / 6)


def apply_butterworth_filter(
    audio_data: AudioData,
    filter_type: ButterFilterType,
    cutoff_frequency: float,
    db_per_octave: Literal[6, 12, 18, 24] = 6,
    plot=False,
) -> AudioData:
    if not 0 < cutoff_frequency < audio_data.get_nyquist_frequency():
        raise FilterError(
            f"Cutoff frequency {cutoff_frequency} Hz is out of signal range \
                (0 - {audio_data.get_nyquist_frequency()} Hz)"
        )

    new_data = audio_data.get_copy()

    sos_filter = butter(
        N=db_per_octave_to_filter_order(db_per_octave),
        Wn=cutoff_frequency,
        btype=filter_type.value,
        output=SECOND_ORDER_SECTIONS,
        fs=new_data.sample_rate,
    )

    if plot:
        get_bode_plot(audio_data, sos_filter, title="Lowpass").show()

    new_data.data = sosfilt(sos_filter, new_data.data)
    return new_data


def _compute_biquad_coefficients(
    omega: float, alpha: float, amplitude_linear_scale: float
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    b0 = 1 + alpha * amplitude_linear_scale
    b1 = -2 * np.cos(omega)
    b2 = 1 - alpha * amplitude_linear_scale
    a0 = 1 + alpha / amplitude_linear_scale
    a1 = -2 * np.cos(omega)
    a2 = 1 - alpha / amplitude_linear_scale
    return ((b0, b1, b2), (a0, a1, a2))


def apply_parametric_band(
    audio_data: AudioData,
    center_frequency: float,
    gain_db: float,
    q_factor: float = 1,
    plot=False,
) -> AudioData:
    if not 0 < center_frequency < audio_data.get_nyquist_frequency():
        raise FilterError(
            f"Center frequency {center_frequency} Hz is out of signal range \
                (0 - {audio_data.get_nyquist_frequency()} Hz)"
        )

    new_data = audio_data.get_copy()

    if gain_db == 0:
        return new_data

    amplitude_linear_scale = 10 ** (gain_db / 40)
    omega = 2 * PI * center_frequency / audio_data.sample_rate
    alpha = np.sin(omega) / (2 * q_factor)

    sos_filter = tf2sos(
        *_compute_biquad_coefficients(omega, alpha, amplitude_linear_scale)
    )

    if plot:
        get_bode_plot(audio_data, sos_filter, title="EQ Band").show()

    new_data.data = sosfilt(sos_filter, new_data.data)
    return new_data
