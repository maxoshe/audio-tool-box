import numpy as np

from audio_toolset.util import convert_db_to_factor


class OscillatorError(Exception): ...


def _validate_oscillator_params(
    sample_rate_hz: int,
    duration_s: float,
    frequency: int,
    amplitude_dbfs: float,
) -> None:
    if amplitude_dbfs > 0:
        raise OscillatorError("Amplitude can't be over 0 dBFS")
    if duration_s <= 0:
        raise OscillatorError("Duration must be over 0 seconds")
    if frequency <= 0:
        raise OscillatorError("Frequency must be over 0 Hz")
    if sample_rate_hz <= 0:
        raise OscillatorError("Sample rate must be over 0 Hz")
    if sample_rate_hz < 2 * frequency:
        raise OscillatorError("Sample rate must be at least 2 * signal frequency")


def generate_sine_wave(
    sample_rate_hz: int,
    duration_s: float,
    frequency: int,
    amplitude_dbfs: float,
) -> np.ndarray:
    _validate_oscillator_params(sample_rate_hz, duration_s, frequency, amplitude_dbfs)

    number_of_samples = int(sample_rate_hz * duration_s)
    factor = convert_db_to_factor(amplitude_dbfs)
    n = np.arange(number_of_samples)
    return factor * np.sin(2 * np.pi * frequency * n / sample_rate_hz)


def generate_square_wave(
    sample_rate_hz: int,
    duration_s: float,
    frequency: int,
    amplitude_dbfs: float,
) -> np.ndarray:
    _validate_oscillator_params(sample_rate_hz, duration_s, frequency, amplitude_dbfs)

    sine_wave = generate_sine_wave(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        frequency=frequency,
        amplitude_dbfs=0,
    )
    factor = convert_db_to_factor(amplitude_dbfs)
    return factor * np.sign(sine_wave)


def generate_white_noise(
    sample_rate_hz: int,
    duration_s: float,
    amplitude_dbfs: float,
) -> np.ndarray:
    _validate_oscillator_params(
        sample_rate_hz=sample_rate_hz,
        duration_s=duration_s,
        frequency=1,
        amplitude_dbfs=amplitude_dbfs,
    )

    number_of_samples = int(sample_rate_hz * duration_s)
    factor = convert_db_to_factor(amplitude_dbfs)
    return factor * (np.random.rand(number_of_samples) - 0.5) * 2
