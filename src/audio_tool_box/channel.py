from os import PathLike
from typing import Literal, Optional, Union
from audio_tool_box.audio_data import AudioData
from audio_tool_box.plots import get_signal_plot
from audio_tool_box.processing.dynamics import (
    apply_compressor,
    apply_cubic_non_linearity,
    apply_limiter,
)
from audio_tool_box.processing.filters import (
    ButterFilterType,
    apply_butterworth_filter,
    apply_parametric_band,
)
from audio_tool_box.processing.gain import apply_gain, normalize_to_target, apply_fade
from audio_tool_box.processing.noise_reduction import apply_spectral_gating


class Channel:
    def __init__(self, source: Union[PathLike[str], AudioData]) -> None:
        """
        Mono channel for processing audio signals.
        Non mono data will be summed to mono.
        To process stereo files use the :func:`split_to_mono` and :func:`join_to_stereo` methods under :module:`audio_tool_box.audio_data`,
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
        Write signal to file
        """
        self.audio_data.write_to_file(output_path=output_path)
        return self

    def plot_signal(self, title: Optional[str] = None) -> "Channel":
        """
        Plots the waveform and power spectral density of the signal.
        """
        get_signal_plot(audio_data=self.audio_data, title=title).show()
        return self

    def gain(self, gain_db: float = 1) -> "Channel":
        """
        Adjusts the the signal amplitude by a decibel amount.
        """
        self.audio_data = apply_gain(audio_data=self.audio_data, gain_db=gain_db)
        return self

    def normalize(self, target_db: float = -0.3) -> "Channel":
        """
        Normalizes the signal to a target dbFS peak value
        """
        self.audio_data = normalize_to_target(
            audio_data=self.audio_data, target_db=target_db
        )
        return self

    def fade(self, fade_duration_ms: int = 100) -> "Channel":
        """
        Creates a fade in and a fade out at the start and end of the signal
        """
        self.audio_data = apply_fade(
            audio_data=self.audio_data, fade_duration_ms=fade_duration_ms
        )
        return self

    def lowpass(
        self,
        cutoff_frequency: float = 80,
        db_per_octave: Literal[6, 12, 18, 24] = 6,
        plot_filter_bode: bool = False,
    ) -> "Channel":
        """
        Filters frequencies above the cutoff frequency
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
        cutoff_frequency: float = 10000,
        db_per_octave: Literal[6, 12, 18, 24] = 6,
        plot_filter_bode: bool = False,
    ) -> "Channel":
        """
        Filters frequencies below the cutoff frequency
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
        Boosts or attenuates frequencies around the cutoff frequency using a a single parametric equalizer band
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
        Reduces noise by attenuating frequencies below the dbFS threshold by a decibel amount
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
        Compresses the signal by reducing sounds that exceed the dbFS threshold,
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
        plot: bool = False,
        normalize_to_original_peak: bool = False,
    ) -> "Channel":
        """
        Limits the signal by strongly reducing sounds that exceed the dbFS threshold.
        """
        self.audio_data = apply_limiter(
            audio_data=self.audio_data,
            threshold_db=thresh_db,
            normalize=normalize_to_original_peak,
            plot=plot,
        )
        return self

    def soft_clipping(self) -> "Channel":
        """
        Performs soft clipping of the signal by using a cubic nonlinearity
        """
        self.audio_data = apply_cubic_non_linearity(audio_data=self.audio_data)
        return self
