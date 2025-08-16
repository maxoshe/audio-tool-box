from os import PathLike
from typing import Union
import numpy as np
import scipy.signal as dsp
import scipy.fftpack as fft
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from audio_tool_box.audio_data import AudioData


class Channel:
    def __init__(self, source: Union[PathLike, AudioData]) -> "Channel":
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

    def plot_signal(self, title: str = "") -> None:
        """
        Plots the waveform and spectrum of the current state of the signal.

            Parameters:
            --------
                title:
                    Title of the plot, default is an empty string
        """
        n = self.audio_data.get_number_of_samples()
        t = np.arange(
            0, n / self.audio_data.sample_rate, 1 / self.audio_data.sample_rate
        )
        psd = np.power(abs(fft.fft(self.audio_data.data)), 2) / (
            self.audio_data.sample_rate * n
        )
        freq = fft.fftfreq(n) * self.audio_data.sample_rate

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=t, y=self.audio_data.data, name="Waveform"), row=1, col=1
        )
        fig.update_yaxes(range=[-1, 1], row=1, col=1)

        fig.add_trace(
            go.Scatter(x=freq[: n // 2], y=10 * np.log10(psd[: n // 2]), name="PSD"),
            row=2,
            col=1,
        )
        fig.update_xaxes(
            type="log", range=[np.log10(20), np.log10(20000)], row=2, col=1
        )
        fig.update_yaxes(range=[-150, 0], row=2, col=1)

        fig.update_layout(title=title)
        fig.show()

    def _bode(self, sos: np.ndarray, title: str) -> None:
        w, h = dsp.filter_design.sosfreqz(sos, fs=self.audio_data.sample_rate)
        db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))

        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=w, y=db, name="Amplitude"),
            row=1,
            col=1,
        )
        fig.update_xaxes(
            type="log", range=[np.log10(20), np.log10(20000)], row=1, col=1
        )
        fig.update_yaxes(range=[-150, 0], row=1, col=1)

        fig.add_trace(
            go.Scatter(x=w, y=np.angle(h), name="Phase"),
            row=2,
            col=1,
        )
        fig.update_xaxes(
            type="log", range=[np.log10(20), np.log10(20000)], row=2, col=1
        )
        fig.update_yaxes(range=[-np.pi, np.pi], row=2, col=1)

        fig.update_layout(title=title)
        fig.show()

    def gain(self, gain_db: float) -> None:
        """
        Adjusts the the signal amplitude by a decibel amount.

            Parameters:
            --------
                gain_db:
                    Amount of gain boost(+) or attenuation(-) to be applied in db
        """
        factor = np.power(10, gain_db / 20)
        self.audio_data.data *= factor

    def normalize(self, target_db: float = -0.3) -> None:
        """
        Normalizes the signal to a target dbFS peak value

            Parameters:
            --------
                target_db:
                    Target dbFS peak value
        """
        target = np.power(10, target_db / 20)
        peak = np.max(np.abs(self.audio_data.data))
        factor = target / peak
        self.audio_data.data *= factor

    def fade(self, ms: int) -> None:
        """
        Creates a fade in and a fade out at the start and end of the signal

            Parameters:
            --------
                ms:
                    Length of the fades in milliseconds
        """
        n = round((ms / 1000) * self.audio_data.sample_rate)
        self.audio_data.data[:n] = self.audio_data.data[:n] * np.linspace(0, 1, num=n)
        self.audio_data.data[-n:] = self.audio_data.data[-n:] * np.linspace(1, 0, num=n)

    def lowpass(self, fc: int, db_per_octave: int = 12, bode: bool = False) -> None:
        """
        Filters frequencies above the cutoff frequency

            Parameters:
            --------
                fc:
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for down sampled signals)
                db_per_octave:
                    Slope of the filter transition band, must be a multiple of 6 (default is 12)
                bode:
                    When True, prints a bode plot of the filter (default is False)
        """
        order = db_per_octave / 6
        sos = dsp.butter(
            order, fc, btype="lowpass", output="sos", fs=self.audio_data.sample_rate
        )
        self.audio_data.data = dsp.sosfilt(sos, self.audio_data.data)
        if bode:
            self._bode(sos, "Lowpass")

    def highpass(self, fc: int, db_per_octave: int = 12, bode: bool = False) -> None:
        """
        Filters frequencies below the cutoff frequency

            Parameters:
            --------
                fc:
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for down sampled signals)
                db_per_octave:
                    Slope of the filter transition band, must be a multiple of 6 (default is 12)
                bode:
                    When True, prints a bode plot of the filter (default is False)
        """
        order = db_per_octave / 6
        sos = dsp.butter(
            order, fc, btype="highpass", output="sos", fs=self.audio_data.sample_rate
        )
        self.audio_data.data = dsp.sosfilt(sos, self.audio_data.data)
        if bode:
            self._bode(sos, "Highpass")

    def eq_band(
        self, fc: int, gain_db: float, q: float = 1, bode: bool = False
    ) -> None:
        """
        Boosts or attenuates frequencies around the cutoff frequency using a a single parametric equalizer band

            Parameters:
            --------
                fc:
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for down sampled signals)
                gain_db:
                    Amount of gain boost(+) or attenuation(-) to be applied in db
                q:
                    quality factor of the equalizer band, higher value results in a narrower band (default is 1)
                bode:
                    When True, prints a bode plot of the filter (default is False)
        """
        a = np.power(10, gain_db / 40)
        wc = fc * 2 * np.pi / self.audio_data.sample_rate
        alpha = np.sin(wc) / (2 * q)
        b0 = 1 + alpha * a
        b1 = -2 * np.cos(wc)
        b2 = 1 - alpha * a
        a0 = 1 + alpha / a
        a1 = -2 * np.cos(wc)
        a2 = 1 - alpha / a
        sos = dsp.tf2sos([b0, b1, b2], [a0, a1, a2])
        self.audio_data.data = dsp.sosfilt(sos, self.audio_data.data)
        if bode:
            self._bode(sos, "EQ band")

    def noise_reduction(self, thresh_db: float = -50, reduction_db: float = -1) -> None:
        """
        Reduces noise by attenuating frequencies below the dbFS threshold by a decibel amount

            Parameters:
            --------
                thresh_db:
                    Threshold in dbFS, frequencies below this threshold are attenuated (default is -50)
                reduction_db:
                    Amount of gain attenuation(-) to be applied in db (default is -1)
        """
        thresh = np.power(10, thresh_db / 20)
        factor = np.power(10, reduction_db / 20)
        signal_fft = fft.fft(self.audio_data.data)
        n = self.audio_data.get_number_of_samples()
        signal_fft[abs(fft.fft(self.audio_data.data)) * (2 / n) < thresh] *= factor
        self.audio_data.data = fft.ifft(signal_fft).real

    def _gain_computer(self, x: np.ndarray, thresh_db: float, ratio: int) -> np.ndarray:
        side_chain = x * 0
        side_chain[x < thresh_db] = x[x < thresh_db]
        side_chain[x > thresh_db] = thresh_db + (x[x > thresh_db] - thresh_db) / ratio
        control_signal = side_chain - x
        return control_signal

    def _gain_smoothing(
        self, gain: float, attack_ms: int, release_ms: int
    ) -> np.ndarray:
        alpha_a = np.exp(-np.log(9) / (self.audio_data.sample_rate * attack_ms / 1000))
        alpha_r = np.exp(-np.log(9) / (self.audio_data.sample_rate * release_ms / 1000))
        smoothed = np.insert(gain, 0, 0)
        for n, value in enumerate(gain):
            if n == 0:
                continue
            if value <= smoothed[n - 1]:
                smoothed[n] = alpha_a * smoothed[n - 1] + (1 - alpha_a) * value
            if value > smoothed[n - 1]:
                smoothed[n] = alpha_r * smoothed[n - 1] + (1 - alpha_r) * value
        return np.delete(smoothed, 0)

    def _plot_dynamics(
        self, x: np.ndarray, attenuation: np.ndarray, thresh: float
    ) -> None:
        t = np.arange(
            0, len(x) / self.audio_data.sample_rate, 1 / self.audio_data.sample_rate
        )
        y_db = 20 * np.log10(np.maximum(np.abs(self.audio_data.data), 1e-5))
        fig = make_subplots(rows=2, cols=1)

        fig.add_trace(
            go.Scatter(x=t, y=x, name="Input signal"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=t * 0 + thresh, name="Threshold"),
            row=1,
            col=1,
        )
        fig.add_trace(
            go.Scatter(x=t, y=attenuation, name="Attenuation"),
            row=1,
            col=1,
        )
        fig.update_yaxes(range=[-40, 0], row=1, col=1)

        fig.add_trace(
            go.Scatter(x=t, y=y_db, name="Output signal"),
            row=2,
            col=1,
        )
        fig.update_yaxes(range=[-40, 0], row=2, col=1)

        fig.show()

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
        uses a smoothing algorithm for gradual attack and release.

            Parameters:
            --------
                thresh_db:
                    Threshold in dbFS, sounds above this threshold are attenuated (default is -20)
                ratio:
                    Ratio of the gain reduction (default is 2)
                attack_ms:
                    Attack time in milliseconds, the time it takes to reach 90% attenuation (default is 15)
                release_ms:
                    Release time in milliseconds, the time it takes to reach 10% attenuation (default is 50)
                plot:
                    When True, prints the gain reduction curve over the signal dynamics (default is False)
        """
        original_peak = np.max(self.audio_data.data)
        x = 20 * np.log10(np.maximum(np.abs(self.audio_data.data), 1e-5))
        gain = self._gain_computer(x, thresh_db, ratio)
        gain_smooth = self._gain_smoothing(gain, attack_ms, release_ms)
        linear_gain = np.power(10, gain_smooth / 20)
        self.audio_data.data *= linear_gain
        compressed_peak = np.max(self.audio_data.data)
        self.audio_data.data *= original_peak / compressed_peak
        if plot:
            self._plot_dynamics(x, gain_smooth, thresh_db)

    def _zero_padding(self, x: np.ndarray, control: np.ndarray, ms: int) -> None:
        samples = self.audio_data.sample_rate * ms / 1000
        pad = np.zeros(int(samples))
        self.audio_data.data = np.concatenate((pad, self.audio_data.data))
        x = np.concatenate((pad, x))
        control = np.concatenate((control, pad))
        return x, control

    def limiter(
        self,
        thresh_db: float = -10,
        release_ms: int = 250,
        ceiling_db: float = 0,
        lookahead_ms: int = 5,
        plot: bool = False,
    ) -> None:
        """
        Limits the signal by strongly reducing sounds that exceed the dbFS threshold.
        Uses a smoothing algorithm and a lookahead to make sure the transient peaks are reduced with minimal noise.
        Normalizes the signal to the dbFS ceiling

            Parameters:
            --------
                thresh_db:
                    Threshold in dbFS, sounds above this threshold are attenuated (default is -20)
                release_ms:
                    Release time in milliseconds, the time it takes to reach 10% attenuation (default is 50)
                ceiling_db:
                    The dbFS ceiling used to normalize the output signal
                lookahead_ms:
                    Used to shift the gain reduction curve ahead of the signal in milliseconds (default is 5)
                plot:
                    When True, prints the gain reduction curve over the signal dynamics (default is False)
        """
        ratio = 1000
        attack_ms = 10
        x = 20 * np.log10(np.maximum(np.abs(self.audio_data.data), 1e-5))
        gain = self._gain_computer(x, thresh_db, ratio)
        x, gain = self._zero_padding(x, gain, attack_ms + lookahead_ms)
        gain_smooth = self._gain_smoothing(gain, attack_ms, release_ms)
        linear_gain = np.power(10, gain_smooth / 20)
        self.audio_data.data *= linear_gain
        self.normalize(ceiling_db)
        if plot:
            self._plot_dynamics(x, gain_smooth, thresh_db)

    def soft_clipping(self) -> None:
        """
        Performs soft clipping of the signal by using a cubic nonlinearity
        """
        peak_db = 20 * np.log10(np.max(np.abs(self.audio_data.data)))
        for i, x in enumerate(self.audio_data.data):
            if x > 1:
                self.audio_data.data[i] = 2 / 3
            elif x < -1:
                self.audio_data.data[i] = -2 / 3
            else:
                self.audio_data.data[i] = x - np.power(x, 3) / 3
        self.normalize(peak_db)
