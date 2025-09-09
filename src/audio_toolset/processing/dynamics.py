import numpy as np
from pydantic import BaseModel, Field

from audio_toolset.audio_data import AudioData
from audio_toolset.plots import get_dynamics_plot
from audio_toolset.processing.gain import normalize_to_target
from audio_toolset.util import convert_db_to_factor, convert_linear_to_db

LIMITER_RATIO = 100000
LIMITER_ATTACK = 0.1
LIMITER_RELEASE = 0.1
LIMITER_KNEE = 0


class DynamicsError(Exception): ...


class GainComputer(BaseModel):
    """
    Implementation of the gain computer and peak smoothing from the following paper
    https://eecs.qmul.ac.uk/~josh/documents/2012/GiannoulisMassbergReiss-dynamicrangecompression-JAES2012.pdf
    """

    audio_data: AudioData
    threshold_db: float = Field(ge=-120, le=0)
    compression_ratio: float = Field(gt=0)
    knee_width_db: float = Field(ge=0)
    attack_ms: float = Field(gt=0)
    release_ms: float = Field(gt=0)

    def _apply_knee_characteristic(self, signal_db: np.ndarray) -> np.ndarray:
        if self.knee_width_db == 0:
            return self._apply_compression_characteristic(signal_db)

        return signal_db + (
            (
                (1 / self.compression_ratio - 1)
                * (signal_db - self.threshold_db + self.knee_width_db / 2) ** 2
            )
            / (2 * self.knee_width_db)
        )

    def _apply_compression_characteristic(self, signal_db: np.ndarray) -> np.ndarray:
        return (
            self.threshold_db + (signal_db - self.threshold_db) / self.compression_ratio
        )

    def _get_control_signal(self) -> np.ndarray:
        """
        Following formula from 2.2 The Gain Computer
        """
        signal_db = convert_linear_to_db(self.audio_data.data)

        lower_knee_threshold = self.threshold_db - (self.knee_width_db / 2)
        upper_knee_threshold = self.threshold_db + (self.knee_width_db / 2)

        linear_mask = signal_db < lower_knee_threshold
        knee_mask = (signal_db >= lower_knee_threshold) & (
            signal_db <= upper_knee_threshold
        )
        compression_mask = upper_knee_threshold < signal_db

        side_chain = np.zeros(self.audio_data.get_number_of_samples())

        side_chain[linear_mask] = signal_db[linear_mask]
        side_chain[knee_mask] = self._apply_knee_characteristic(signal_db[knee_mask])
        side_chain[compression_mask] = self._apply_compression_characteristic(
            signal_db[compression_mask]
        )

        return side_chain - signal_db

    def _get_one_pole_smoothing_coefficient(self, time_constant_ms: float) -> float:
        return np.exp(
            -np.log(9) / (self.audio_data.sample_rate * time_constant_ms / 1000)
        )

    def _apply_one_pole_filter_step(
        self, previous_filtered_sample: float, current_sample: float, alpha: float
    ) -> float:
        return alpha * previous_filtered_sample + (1 - alpha) * current_sample

    def _apply_gain_smoothing(self, gain_reduction: np.ndarray) -> np.ndarray:
        """
        Following formula from 2.3.4 SmoothPeakDetectors
        """
        alpha_attack = self._get_one_pole_smoothing_coefficient(self.attack_ms)
        alpha_release = self._get_one_pole_smoothing_coefficient(self.release_ms)

        smooth_gain_reduction = gain_reduction.copy()
        for n in range(1, len(gain_reduction)):
            alpha = (
                alpha_release
                if gain_reduction[n] > smooth_gain_reduction[n - 1]
                else alpha_attack
            )
            smooth_gain_reduction[n] = self._apply_one_pole_filter_step(
                smooth_gain_reduction[n - 1], gain_reduction[n], alpha
            )
        return smooth_gain_reduction

    def get_gain_reduction(self) -> np.ndarray:
        control_signal = self._get_control_signal()
        return self._apply_gain_smoothing(gain_reduction=control_signal)


def apply_compressor(
    audio_data: AudioData,
    threshold_db: float = -20,
    compression_ratio: float = 4,
    knee_width_db: float = 1,
    attack_ms: float = 15,
    release_ms: float = 50,
    plot: bool = False,
    normalize: bool = False,
) -> AudioData:
    try:
        gain_computer = GainComputer(
            audio_data=audio_data,
            threshold_db=threshold_db,
            compression_ratio=compression_ratio,
            knee_width_db=knee_width_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
        )
    except Exception as e:
        raise DynamicsError("Gain computer model validation failed") from e

    gain_reduction = gain_computer.get_gain_reduction()
    new_data = audio_data.get_copy()

    new_data.data *= convert_db_to_factor(gain_reduction)

    if normalize:
        new_data = normalize_to_target(
            audio_data=new_data, target_db=audio_data.get_peak()
        )

    if plot:
        get_dynamics_plot(
            audio_data_pre=audio_data,
            audio_data_post=new_data,
            attenuation_db=gain_reduction,
            threshold_db=threshold_db,
        ).show()

    return new_data


def apply_limiter(
    audio_data: AudioData,
    threshold_db: float = -20,
    plot: bool = False,
    normalize: bool = False,
) -> AudioData:
    try:
        gain_computer = GainComputer(
            audio_data=audio_data,
            threshold_db=threshold_db,
            compression_ratio=LIMITER_RATIO,
            knee_width_db=LIMITER_KNEE,
            attack_ms=LIMITER_ATTACK,
            release_ms=LIMITER_RELEASE,
        )
    except Exception as e:
        raise DynamicsError("Gain computer model validation failed") from e

    gain_reduction = gain_computer.get_gain_reduction()
    new_data = audio_data.get_copy()

    new_data.data *= convert_db_to_factor(gain_reduction)

    if normalize:
        new_data = normalize_to_target(
            audio_data=new_data, target_db=audio_data.get_peak()
        )

    if plot:
        get_dynamics_plot(
            audio_data_pre=audio_data,
            audio_data_post=new_data,
            attenuation_db=gain_reduction,
            threshold_db=threshold_db,
        ).show()

    return new_data


def apply_cubic_non_linearity(audio_data: AudioData) -> AudioData:
    new_data = audio_data.get_copy()

    over_mask = new_data.data > 1
    under_mask = new_data.data < -1
    center_mask = (new_data.data <= 1) & (new_data.data >= -1)

    new_data.data[over_mask] = 2 / 3
    new_data.data[under_mask] = -2 / 3
    new_data.data[center_mask] = (
        new_data.data[center_mask] - (new_data.data[center_mask] ** 3) / 3
    )

    return new_data
