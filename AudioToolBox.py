"""
A python module for processing audio.

AudioToolBox provides a user friendly way to process audio in python.
Each channel object is used to import, process and export a single mono wav file.
When processing a stereo file, the left and right chanelles will be summed to a single mono channel.
Another option is to use the provided split_to_mono and join_to_stereo methods to process each channel individualy and then join them.

Dependencies
--------
    numpy, scipy, soundfile, matplotlib

Example
--------
Process a guitar signal using a channel strip::

    import AudioToolBox
    guitar = AudioToolBox.channel('guitar.wav')
    guitar.highpass(fc=100, db_per_octave=6)
    guitar.eq_band(fc=2000, gain_db=3, q=1)
    guitar.lowpass(fc=10000, db_per_octave=12)
    guitar.normalize(target_db=-1)
    guitar.compressor(tresh_db=-20, ratio=4, attack_ms=50, release_ms=100)
    guitar.export('guitar_processed.wav')
"""
import os
import numpy as np
import soundfile as sf
import scipy.signal as dsp
import scipy.fftpack as fft
import matplotlib.pyplot as plt

def split_to_mono(path):
    """
    Split one stereo wav file to seperate mono .wav files
        Parameters:
        --------
            path: str
                The path of the input stereo .wav file.
    """
    y, fs = sf.read(path)
    sf.write(os.path.splitext(path)[0] + '_L.wav', y[:, 0], fs)
    sf.write(os.path.splitext(path)[0] + '_R.wav', y[:, 1], fs)
    return

def join_to_stereo(L_path, R_path, path):
    """
    Join two mono .wav files to one stereo .wav file
        Parameters:
        --------
            L_path: str
                The path of the input left mono .wav file.
            R_path: str
                The path of the input right mono .wav file.
            path: str
                path of the output stereo .wav file
    """
    y_L, fs = sf.read(L_path)
    y_R, fs = sf.read(R_path)
    y = np.column_stack((y_L, y_R))
    sf.write(path, y, fs)
    return

class channel:
    def __init__(self, path):
        """
        Create an audio channel and import a signal from a .wav file
        Sums stereo channels to a single mono channel.
        To process stereo files use the split_to_mono method,
        process the left and right channels individualy, 
        then join them with the join_to_stereo method.

            Parameters:
            --------
                path: str
                    The path of the input .wav file.

            Returns:
            --------
                channel
                    a mono channel object that can call a list of DSP methods

        """
        self.__y, self.__fs = sf.read(path)
        if sf.info(path).channels > 1:
            self.__sum_to_mono()
        return

    def __sum_to_mono(self):
        left = self.__y[:, 0]
        right = self.__y[:, 1]
        ave = (left + right) / 2
        del self.__y
        self.__y = ave
        return

    def export(self, path):
        """
        Exports the current state of the signal, creates a new file if the file does not exist.

            Parameters:
            --------
                path: str
                    The path of the output .wav file.
        """
        sf.write(path, self.__y, self.__fs)
        return

    def plot_signal(self, title=''):
        """
        Plots the waveform and spectrum of the current state of the signal.

            Parameters:
            --------
                title: str, optional
                    Title of the plot, default is an empty string
        """
        n = len(self.__y)
        t = np.arange(0, n/self.__fs, 1/self.__fs)
        psd = np.power(abs(fft.fft(self.__y)),2)  / (self.__fs * n)
        freq = fft.fftfreq(n) * self.__fs
        plt.figure()
        plt.suptitle(title)
        plt.subplot(2,1,1)
        plt.plot(t, self.__y, linewidth=0.5)
        plt.xlim([0, len(self.__y)/self.__fs])
        plt.ylim([-1,1])
        plt.subplot(2,1,2)
        plt.semilogx(freq[:n//2], 10*np.log10(psd[:n//2]), 'tab:gray', linewidth=0.5)
        f = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks = [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k']
        plt.xticks(f, ticks)
        plt.xlim([20, 20000])
        plt.ylim([-140, 0])
        plt.show()
        return

    def __bode(self, sos, title):
        w, h = dsp.filter_design.sosfreqz(sos, fs=self.__fs)
        db = 20 * np.log10(np.maximum(np.abs(h), 1e-5))
        plt.figure()
        plt.suptitle(title)
        plt.subplot(2,1,1)
        plt.semilogx(w, db, 'k', linewidth=1)
        f = [20, 50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000]
        ticks = [20, 50, 100, 200, 500, '1k', '2k', '5k', '10k', '20k']
        plt.xticks(f, ticks)
        plt.xlim([20, 20000])
        plt.subplot(2,1,2)
        plt.semilogx(w, np.angle(h), 'k', linewidth=1)
        plt.xticks(f, ticks)
        plt.xlim([20, 20000])
        plt.yticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi], [r'$-\pi$', r'$-\pi/2$', 0, r'$\pi/2$', r'$\pi$'])
        plt.show()

    def gain(self, gain_db):
        """
        Adjusts the the signal amplitude by a decibel amount.

            Parameters:
            --------
                gain_db: float
                    Amount of gain boost(+) or attenuation(-) to be applied in db
        """
        factor = np.power(10, gain_db/20)
        self.__y *= factor

    def normalize(self, target_db=-0.3):
        """
        Normalizes the signal to a target dbFS peak value

            Parameters:
            --------
                target_db: float
                    Target dbFS peak value
        """
        target = np.power(10, target_db/20)
        peak = np.max(np.abs(self.__y))
        factor = target/peak
        self.__y *= factor
        return

    def fade(self, ms):
        """
        Creates a fade in and a fade out at the start and end of the signal

            Parameters:
            --------
                ms: int
                    Length of the fades in miliseconds
        """
        n = round((ms/1000) * self.__fs)
        self.__y[:n] = self.__y[:n] * np.linspace(0, 1, num=n)
        self.__y[-n:] = self.__y[-n:] * np.linspace(1, 0, num=n)
        return

    def lowpass(self, fc, db_per_octave=12, bode=False):
        """
        Filters frequencies above the cutoff frequency

            Parameters:
            --------
                fc: int
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for downsampled signals)
                db_per_octave: int, optional
                    Slope of the filter transition band, must be a multiple of 6 (default is 12)
                bode: bool, optional
                    When True, prints a bode plot of the filter (default is False)
        """
        order = db_per_octave / 6
        sos = dsp.butter(order, fc, btype='lowpass', output='sos', fs=self.__fs)
        self.__y = dsp.sosfilt(sos, self.__y)
        if bode:
            self.__bode(sos, 'Lowpass')
        return

    def highpass(self, fc, db_per_octave=12, bode=False):
        """
        Filters frequencies below the cutoff frequency

            Parameters:
            --------
                fc: int
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for downsampled signals)
                db_per_octave: int, optional
                    Slope of the filter transition band, must be a multiple of 6 (default is 12)
                bode: bool, optional
                    When True, prints a bode plot of the filter (default is False)
        """
        order = db_per_octave / 6
        sos = dsp.butter(order, fc, btype='highpass', output='sos', fs=self.__fs)
        self.__y = dsp.sosfilt(sos, self.__y)
        if bode:
            self.__bode(sos, 'Highpass')
        return

    def eq_band(self, fc, gain_db, q=1, bode=False):
        """
        Bosts or attenuates frequencies around the cutoff frequency using a a single parametric equalizer band

            Parameters:
            --------
                fc: int
                    Cutoff frequency in Hz, should be between 20Hz and 20000Hz
                    (or 20Hz to fs/2 for downsampled signals)
                gain_db: float
                    Amount of gain boost(+) or attenuation(-) to be applied in db
                q: float, optional
                    quality factor of the equalizer band, higher value results in a narrower band (default is 1)
                bode: bool, optional
                    When True, prints a bode plot of the filter (default is False)
        """
        a = np.power(10, gain_db/40)
        wc = fc * 2 * np.pi / self.__fs
        alpha = np.sin(wc) / (2 * q)
        b0 = 1 + alpha * a
        b1 = -2 * np.cos(wc)
        b2 = 1 - alpha * a
        a0 = 1 + alpha / a
        a1 = -2 * np.cos(wc)
        a2 = 1 - alpha / a
        sos = dsp.tf2sos([b0, b1, b2], [a0, a1, a2])
        self.__y = dsp.sosfilt(sos, self.__y)
        if bode:
            self.__bode(sos, 'EQ band')
        return

    def noise_reduction(self, tresh_db=-50, reduction_db=-1):
        """
        Reduces noise by attenuating frequencies below the dbFS treshold by a decibel amount

            Parameters:
            --------
                tresh_db: float, optional
                    Treshold in dbFS, frequencies below this treshold are attenuated (default is -50)
                reduction_db: float, optional
                    Amount of gain attenuation(-) to be applied in db (default is -1)
        """
        tresh = np.power(10, tresh_db/20)
        factor = np.power(10, reduction_db/20)
        signal_fft = fft.fft(self.__y)
        n = len(self.__y)
        signal_fft[abs(fft.fft(self.__y)) * (2/n) < tresh] *= factor
        self.__y = fft.ifft(signal_fft).real
        return

    def __gain_computer(self, x, tresh_db, ratio):
        side_chain = x * 0
        side_chain[x < tresh_db] = x[x < tresh_db]
        side_chain[x > tresh_db] = tresh_db + (x[x > tresh_db] - tresh_db)/ratio
        control_signal = side_chain - x
        return control_signal

    def __gain_smoothing(self, gain, attack_ms, release_ms):
        alpha_a = np.exp( -np.log(9) / (self.__fs * attack_ms/1000) )
        alpha_r = np.exp( -np.log(9) / (self.__fs * release_ms/1000) )
        smoothed = np.insert(gain, 0, 0)
        for n, value in enumerate(gain):
            if n == 0:
                continue
            if value <= smoothed[n-1]:
                smoothed[n] = alpha_a * smoothed[n-1] + (1 - alpha_a) * value
            if value > smoothed[n-1]:
                smoothed[n] = alpha_r * smoothed[n-1] + (1 - alpha_r) * value
        return np.delete(smoothed, 0)

    def __plot_dynamics(self, x, attenuation, tresh):
        t = np.arange(0, len(x)/self.__fs, 1/self.__fs)
        y_db = 20 * np.log10(np.maximum(np.abs(self.__y), 1e-5))
        plt.figure()
        plt.subplot(2,1,1)
        plt.plot(t, x, linewidth=0.5)
        plt.plot(t, t*0 +tresh, linewidth=1)
        plt.plot(t, attenuation, linewidth=0.5)
        plt.ylim([-40, 0])
        plt.xlim([0, len(self.__y)/self.__fs])
        plt.subplot(2,1,2)
        plt.plot(t, y_db, linewidth=0.5)
        plt.ylim([-40, 0])
        plt.xlim([0, len(self.__y)/self.__fs])
        plt.show()
        return

    def compressor(self, tresh_db=-20, ratio=2, attack_ms=15, release_ms=50, plot=False):
        """
        Compresses the signal by reducing sounds that exceed the dbFS treshold,
        uses a smoothing algorithm for gradual attack and release.

            Parameters:
            --------
                tresh_db: float, optional
                    Treshold in dbFS, sounds above this treshold are attenuated (default is -20)
                ratio: int, optional
                    Ratio of the gain reduction (default is 2)
                attack_ms: int, optional
                    Attack time in miliseconds, the time it takes to reach 90% attenuation (default is 15)
                release_ms: int, optional
                    Release time in miliseconds, the time it takes to reach 10% attenuation (default is 50)
                plot: bool, optional
                    When True, prints the gain reduction curve over the signal dynamics (default is False)
        """
        original_peak = np.max(self.__y)
        x = 20 * np.log10(np.maximum(np.abs(self.__y), 1e-5))
        gain = self.__gain_computer(x, tresh_db, ratio)
        gain_smooth = self.__gain_smoothing(gain, attack_ms, release_ms)
        linear_gain = np.power(10, gain_smooth/20)
        self.__y *= linear_gain
        compressed_peak = np.max(self.__y)
        self.__y *= original_peak/compressed_peak
        if plot:
            self.__plot_dynamics(x, gain_smooth, tresh_db)
        return

    def __zero_padding(self, x, control, ms):
        samples = self.__fs * ms / 1000
        pad = np.zeros(int(samples))
        self.__y = np.concatenate((pad, self.__y))
        x = np.concatenate((pad, x))
        control = np.concatenate((control, pad))
        return x, control

    def limiter(self, tresh_db=-10, release_ms=250, cieling_db=0, lookahead_ms=5, plot=False):
        """
        Limits the signal by strongly reducing sounds that exceed the dbFS treshold.
        Uses a smoothing algorithm and a lookahead to make sure the transient peaks are reduced with minimal noise.
        Normalizes the signal to the dbFS cieling

            Parameters:
            --------
                tresh_db: float, optional
                    Treshold in dbFS, sounds above this treshold are attenuated (default is -20)
                release_ms: int, optional
                    Release time in miliseconds, the time it takes to reach 10% attenuation (default is 50)
                cieling_db: float, optional
                    The dbFS cieling used to normalize the output signal
                lookahead_ms: int, optional
                    Used to shift the gain reduction curve ahead of the signal in miliseconds (default is 5)
                plot: bool, optional
                    When True, prints the gain reduction curve over the signal dynamics (default is False)
        """
        ratio = 1000
        attack_ms = 10
        x = 20 * np.log10(np.maximum(np.abs(self.__y), 1e-5))
        gain = self.__gain_computer(x, tresh_db, ratio)
        x, gain = self.__zero_padding(x, gain, attack_ms + lookahead_ms)
        gain_smooth = self.__gain_smoothing(gain, attack_ms, release_ms)
        linear_gain = np.power(10, gain_smooth/20)
        self.__y *= linear_gain
        self.normalize(cieling_db)
        if plot:
            self.__plot_dynamics(x, gain_smooth, tresh_db)
        return

    def soft_clipping(self):
        """
        Performs soft clipping of the signal by using a cubic nonlinearity
        """
        peak_db = 20 * np.log10(np.max(np.abs(self.__y)))
        for i, x in enumerate(self.__y):
            if x > 1:
                self.__y[i] = 2/3
            elif x < -1:
                self.__y[i] = -2/3
            else:
                self.__y[i] = x - np.power(x, 3)/3
        self.normalize(peak_db)
        return