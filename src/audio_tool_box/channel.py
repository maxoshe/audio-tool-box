from os import PathLike
from typing import Optional, Union
import numpy as np
import scipy.fftpack as fft
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
    def __init__(self, source: Union[PathLike, AudioData]) -> None:
        """
        Mono channel for processing audio signals.
        Non mono data will be summed to mono.
        To process stereo files use the :func:`split_to_mono` and :func:`join_to_stereo` methods under :module:`audio_tool_box.audio_data`,
        """
        if isinstance(source, AudioData):
            audio_data = source
        else:
            audio_data = AudioData.read_from_file(source)

        if not audio_data.is_mono():
            audio_data = audio_data.sum_to_mono()
        self.audio_data = audio_data

    def export(self, path: PathLike) -> None:
        """
        Write signal to file
        """
        self.audio_data.write_to_file(path)

    def plot_signal(self, title: Optional[str] = None) -> None:
        """
        Plots the waveform and power spectral density of the signal.
        """
        get_signal_plot(audio_data=self.audio_data, title=title).show()

    def gain(self, gain_db: float) -> None:
        """
        Adjusts the the signal amplitude by a decibel amount.
        """
        self.audio_data = apply_gain(self.audio_data, gain_db)

    def normalize(self, target_db: float = -0.3) -> None:
        """
        Normalizes the signal to a target dbFS peak value
        """
        self.audio_data = normalize_to_target(self.audio_data, target_db)

    def fade(self, fade_duration_ms: int = 100) -> None:
        """
        Creates a fade in and a fade out at the start and end of the signal
        """
        self.audio_data = apply_fade(self.audio_data, fade_duration_ms)

    def lowpass(self, fc: int, db_per_octave: int = 12, bode: bool = False) -> None:
        """
        Filters frequencies above the cutoff frequency
        """
        self.audio_data = apply_butterworth_filter(
            self.audio_data, ButterFilterType.LOWPASS, fc, db_per_octave, bode
        )

    def highpass(self, fc: int, db_per_octave: int = 12, bode: bool = False) -> None:
        """
        Filters frequencies below the cutoff frequency
        """
        self.audio_data = apply_butterworth_filter(
            self.audio_data, ButterFilterType.HIGHPASS, fc, db_per_octave, bode
        )

    def eq_band(
        self, fc: int, gain_db: float, q: float = 1, bode: bool = False
    ) -> None:
        """
        Boosts or attenuates frequencies around the cutoff frequency using a a single parametric equalizer band
        """
        self.audio_data = apply_parametric_band(self.audio_data, fc, gain_db, q, bode)

    def noise_reduction(self, thresh_db: float = -50, reduction_db: float = -1) -> None:
        """
        Reduces noise by attenuating frequencies below the dbFS threshold by a decibel amount
        """
        self.audio_data = apply_spectral_gating(
            self.audio_data, thresh_db, reduction_db
        )

    def compressor(
        self,
        thresh_db: float = -20,
        ratio: int = 2,
        attack_ms: int = 15,
        release_ms: int = 50,
        plot: bool = False,
    ) -> None:
        """
        Compresses the signal by reducing sounds that exceed the dbFS threshold,
        """
        self.audio_data = apply_compressor(
            self.audio_data,
            thresh_db,
            ratio,
            attack_ms=attack_ms,
            release_ms=release_ms,
            plot=plot,
        )

    def limiter(
        self,
        thresh_db: float = -10,
        plot: bool = False,
    ) -> None:
        """
        Limits the signal by strongly reducing sounds that exceed the dbFS threshold.
        """
        self.audio_data = apply_limiter(
            self.audio_data,
            thresh_db,
            plot=plot,
        )

    def soft_clipping(self) -> None:
        """
        Performs soft clipping of the signal by using a cubic nonlinearity
        """
        self.audio_data = apply_cubic_non_linearity(self.audio_data)
