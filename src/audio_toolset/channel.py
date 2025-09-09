from os import PathLike
from typing import Literal

from audio_toolset.audio_data import AudioData
from audio_toolset.plots import get_signal_plot
from audio_toolset.processing.dynamics import (
    apply_compressor,
    apply_cubic_non_linearity,
    apply_limiter,
)
from audio_toolset.processing.filters import (
    ButterFilterType,
    apply_butterworth_filter,
    apply_parametric_band,
)
from audio_toolset.processing.gain import apply_fade, apply_gain, normalize_to_target
from audio_toolset.processing.noise_reduction import apply_spectral_gating


class Channel:
    def __init__(self, source: PathLike[str] | AudioData) -> None:
        """
        Initialize a mono audio channel for processing.

        If the provided audio is not mono, it will be summed to mono.
        For stereo processing, use `split_to_mono` and `join_to_stereo` from \
            `audio_toolset.audio_data`.

        Args:
            source (Union[PathLike[str], AudioData]): Path to an audio file \
                or an AudioData object.
        """
        if isinstance(source, AudioData):
            audio_data = source
        else:
            audio_data = AudioData.read_from_file(file_path=source)

        if not audio_data.is_mono():
            audio_data = audio_data.sum_to_mono()
        self.audio_data = audio_data

    def write(self, output_path: PathLike[str]) -> "Channel":
        """
        Save the audio data to a file.

        Args:
            output_path (PathLike[str]): Path where the audio file will be saved.

        Returns:
            Channel: Returns self to allow method chaining.
        """
        self.audio_data.write_to_file(output_path=output_path)
        return self

    def plot_signal(self, title: str | None = None) -> "Channel":
        """
        Plot the waveform and power spectral density of the audio signal.

        Args:
            title (Optional[str], optional): Title of the plot. \
                Defaults to path of file when None.

        Returns:
            Channel: Returns self for method chaining.
        """
        get_signal_plot(audio_data=self.audio_data, title=title).show()
        return self

    def gain(self, gain_db: float = 1) -> "Channel":
        """
        Adjust the amplitude of the signal by a specified gain in decibels.

        Args:
            gain_db (float, optional): Gain in dB to apply. Defaults to 1 dB.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_gain(audio_data=self.audio_data, gain_db=gain_db)
        return self

    def normalize(self, target_db: float = -0.3) -> "Channel":
        """
        Normalize the signal so its peak reaches a target dBFS value.

        Args:
            target_db (float, optional): Target peak level in dBFS. Defaults to -0.3 dB.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = normalize_to_target(
            audio_data=self.audio_data, target_db=target_db
        )
        return self

    def fade(self, fade_duration_ms: int = 100) -> "Channel":
        """
        Apply a linear fade-in at the start and fade-out at the end of the signal.

        Args:
            fade_duration_ms (int, optional): Duration of fade-in and fade-out in \
                milliseconds. Defaults to 100 ms.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_fade(
            audio_data=self.audio_data, fade_duration_ms=fade_duration_ms
        )
        return self

    def lowpass(
        self,
        cutoff_frequency: float = 10000,
        db_per_octave: Literal[6, 12, 18, 24] = 6,
        plot_filter_bode: bool = False,
    ) -> "Channel":
        """
        Apply a lowpass Butterworth filter to remove frequencies above the cutoff.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency in Hz. \
                Defaults to 10000 Hz.
            db_per_octave (Literal[6, 12, 18, 24], optional): Filter slope. \
                Defaults to 6 dB/octave.
            plot_filter_bode (bool, optional): If True, plots the filter response. \
                Defaults to False.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_butterworth_filter(
            audio_data=self.audio_data,
            filter_type=ButterFilterType.LOWPASS,
            cutoff_frequency=cutoff_frequency,
            db_per_octave=db_per_octave,
            plot=plot_filter_bode,
        )
        return self

    def highpass(
        self,
        cutoff_frequency: float = 80,
        db_per_octave: Literal[6, 12, 18, 24] = 6,
        plot_filter_bode: bool = False,
    ) -> "Channel":
        """
        Apply a highpass Butterworth filter to remove frequencies below the cutoff.

        Args:
            cutoff_frequency (float, optional): Cutoff frequency in Hz. \
                Defaults to 80 Hz.
            db_per_octave (Literal[6, 12, 18, 24], optional): Filter slope. \
                Defaults to 6 dB/octave.
            plot_filter_bode (bool, optional): If True, plots the filter response. \
                Defaults to False.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_butterworth_filter(
            audio_data=self.audio_data,
            filter_type=ButterFilterType.HIGHPASS,
            cutoff_frequency=cutoff_frequency,
            db_per_octave=db_per_octave,
            plot=plot_filter_bode,
        )
        return self

    def eq_band(
        self,
        center_frequency: float = 800,
        gain_db: float = -3,
        q_factor: float = 1,
        plot_filter_bode: bool = False,
    ) -> "Channel":
        """
        Apply a single parametric EQ band to boost or attenuate frequencies.

        Args:
            center_frequency (float, optional): Center frequency of the EQ band in Hz. \
                Defaults to 800 Hz.
            gain_db (float, optional): Gain in dB to apply. Positive boosts, \
                negative attenuates. Defaults to -3 dB.
            q_factor (float, optional): Quality factor controlling bandwidth. \
                Defaults to 1.
            plot_filter_bode (bool, optional): If True, plots the filter response. \
                Defaults to False.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_parametric_band(
            audio_data=self.audio_data,
            center_frequency=center_frequency,
            gain_db=gain_db,
            q_factor=q_factor,
            plot=plot_filter_bode,
        )
        return self

    def noise_reduction(
        self, noise_threshold_db: float = -50, attenuation_db: float = -1
    ) -> "Channel":
        """
        Reduce background noise by attenuating low-level signals below a threshold.

        Args:
            noise_threshold_db (float, optional): Threshold below which signals are \
                attenuated in dBFS. Defaults to -50 dB.
            attenuation_db (float, optional): Amount of attenuation applied to noise \
                in dB. Defaults to -1 dB.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_spectral_gating(
            audio_data=self.audio_data,
            noise_threshold_db=noise_threshold_db,
            attenuation_db=attenuation_db,
        )
        return self

    def compressor(
        self,
        threshold_db: float = -20,
        compression_ratio: int = 2,
        knee_width_db: float = 1,
        attack_ms: int = 15,
        release_ms: int = 50,
        normalize_to_original_peak: bool = False,
        plot_compressor_response: bool = False,
    ) -> "Channel":
        """
        Apply dynamic range compression to reduce the volume of loud sounds.

        Args:
            threshold_db (float, optional): Level above which compression occurs \
                in dBFS. Defaults to -20 dB.
            compression_ratio (int, optional): Ratio of input to output above \
                threshold. Defaults to 2.
            knee_width_db (float, optional): Smoothness of compression around \
                threshold in dB. Defaults to 1 dB.
            attack_ms (int, optional): Attack time in milliseconds. Defaults to 15 ms.
            release_ms (int, optional): Release time in milliseconds. Defaults to 50 ms.
            normalize_to_original_peak (bool, optional): If True, scales output to \
                original peak. Defaults to False.
            plot_compressor_response (bool, optional): If True, plots the dynamics of \
                compression, including input signal, threshold, attenuation applied, \
                    and resulting output signal. Defaults to False.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_compressor(
            audio_data=self.audio_data,
            threshold_db=threshold_db,
            compression_ratio=compression_ratio,
            knee_width_db=knee_width_db,
            attack_ms=attack_ms,
            release_ms=release_ms,
            normalize=normalize_to_original_peak,
            plot=plot_compressor_response,
        )
        return self

    def limiter(
        self,
        thresh_db: float = -10,
        plot_limiter_response: bool = False,
        normalize_to_original_peak: bool = False,
    ) -> "Channel":
        """
        Apply a limiter to strictly prevent the signal from exceeding a threshold.

        Args:
            thresh_db (float, optional): Maximum allowed signal level in dBFS. \
                Defaults to -10 dB.
            plot_limiter_response (bool, optional): If True, plots the limiting \
                process, including input signal, threshold, attenuation applied, \
                    and resulting output signal. Defaults to False.
            normalize_to_original_peak (bool, optional): If True, scales output to \
                original peak. Defaults to False.

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_limiter(
            audio_data=self.audio_data,
            threshold_db=thresh_db,
            normalize=normalize_to_original_peak,
            plot=plot_limiter_response,
        )
        return self

    def soft_clipping(self) -> "Channel":
        """
        Apply soft clipping using a cubic non-linearity to prevent harsh distortion.

        Args:
            None

        Returns:
            Channel: Returns self for chaining.
        """
        self.audio_data = apply_cubic_non_linearity(audio_data=self.audio_data)
        return self
