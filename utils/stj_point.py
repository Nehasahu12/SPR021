import numpy as np
from utils import function as func
import matplotlib.pyplot as plt


def apply_derivative_filter(signal, kernel):
    # Convolve the signal with the kernel
    filtered_signal = np.convolve(signal, kernel, mode='same')

    return filtered_signal


class StjPointer:
    def __init__(self, cleaned_signal, qrs_peak, sampling_rate=400, win=50):
        """
        parameter:
            param: cleaned_signal: waveform
            param: qrs_peak: array of qrs peak
            param: sampling_rate: in Hz
            param: win: length in Milli second
        """
        self.cleaned_signal = cleaned_signal
        self.r_peaks = qrs_peak
        self.sampling_rate = sampling_rate
        self.per_sample_time = 1 / self.sampling_rate
        self.win_time = win
        self.win_length = int((self.win_time / 1000) / self.per_sample_time)
        self.baseline = 0

    def scatter_qrs_peak(self, adjust_sig=True):
        bs = func.baseline_finder(self.cleaned_signal, [self.r_peaks["index"]], margin=0)
        print(f"AVR Base Line: {np.average(bs[1])}")
        self.baseline = np.average(bs[1])
        # Addition setting for baseline wandering
        if adjust_sig:
            self.cleaned_signal = self.cleaned_signal - np.average(bs[1])
            self.r_peaks["value"] = self.r_peaks["value"] - np.average(bs[1])
            self.baseline = 0

        kernel = [-1, 1, -1, 1]
        x = np.arange(len(self.cleaned_signal))
        plt.step(x, self.cleaned_signal, color="green")
        plt.plot(x, self.cleaned_signal, '-.')
        plt.plot(x, apply_derivative_filter(self.cleaned_signal, kernel), color="darkorange")
        plt.scatter(self.r_peaks["index"], self.r_peaks["value"], marker="D", color="red")
        plt.axhline(self.baseline)
        plt.show()
