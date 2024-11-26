#!/usr/bin/env python3
"""
objective: To find and plot ECG R wave progression and visualize ECG data
# TODO : We will add all visualisation related code in this file.
@Auther:
 - Nitish kumar sharma (nitishsharma.era378@gmail.com)

"""

import matplotlib.pyplot as plt
import numpy as np
from utils import function as func


class EcgProgression:
    def __init__(self, cleaned_signal, qrs_peak, sampling_rate=400, anterior_leads=None, rr_interval=800):
        """
        Parameters:
        - cleaned_signal: ECG wave form in the format of dictionary
        - qrs_peak: QRS peaks in the form format of dictionary
        - sampling_rate: sampling rate of the signal/waveform
        - anterior_leads: list of all ecg leads
        Returns:
        - None
        """
        # TODO: need Implementation of all the necessary calculation regarding the Progression
        self.cleaned_signal = cleaned_signal
        self.qrs_peak = qrs_peak
        self.sampling_rate = sampling_rate
        self.r_amp = []
        self.anterior_leads = anterior_leads
        self.prog_result = {}
        print("NU: ", rr_interval)

    def calculate_progression(self, show=True):
        x = []
        y = []
        for index, lead in enumerate(self.anterior_leads):
            self.prog_result[lead] = {}
            val = np.average(self.qrs_peak[lead]["value"])
            x.append(index)
            y.append(val)
        if show:
            plt.title("R Progression")
            plt.xlabel("Leads")
            plt.ylabel("R Amplitude")
            plt.stem(x, y, "-o", label="Leads")
            plt.plot(self.anterior_leads, y, "s-", color="orangered", linewidth=2, markersize=8,
                     label="R wave progression")
            for ind, val in zip(x, y):
                plt.text(ind, val, "{:.3f} mV".format(val), fontsize=12, color='black')
            plt.legend(loc="upper left")
            plt.show()


class NormalVisualizer:
    def __init__(self, df, qrs_peak, pid="124", rr_interval=None, sampling_rate=400):
        """
        param df: signal in the dictionary format
        param qrs_peak: qrs peak in the dictionary format
        param pid: patient id optional, by default it will be 123
        """
        if rr_interval is None:
            rr_interval = [800, 820, 780]
        self.segments = {}
        self.segments_2 = {}
        self.strip = []
        self.x = []
        self.pid = pid
        self.cleaned_signal = df
        self.qrs_peak = qrs_peak
        self.anterior_leads = ["Lead_I", "Lead_II", "Lead_III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.persamtim = 1 / sampling_rate
        tp = []
        for i in rr_interval:
            if not np.isnan(rr_interval[i]):
                print(rr_interval[i])
                tp.append(rr_interval[i])
        self.rr_interval = np.average(tp)
        self.win_length = int(((self.rr_interval / 1000) / self.persamtim) / 2)
        print(f"Window Length: {self.win_length}")
        self.split_all()

    def split_all(self):
        for leads in self.anterior_leads:
            self.segments[leads] = func.split_ecg_signal_p(self.cleaned_signal[leads], self.qrs_peak[leads]["index"],
                                                           left_m=0, right_m=0)

            self.segments_2[leads] = func.segmentation(self.cleaned_signal[leads], self.qrs_peak[leads]["index"],
                                                       win=self.win_length, margin=5)

    def plot_leads(self):
        init = 0
        # plt.cla()
        plt.figure(2)
        plt.rcParams["figure.subplot.left"] = 0.07
        plt.rcParams["figure.subplot.right"] = 0.95
        plt.rcParams["figure.subplot.bottom"] = 0.1
        plt.rcParams["figure.subplot.top"] = 0.97
        plt.rcParams["figure.subplot.wspace"] = 0
        plt.rcParams["figure.subplot.hspace"] = 0.01
        plt.ylim(-3.5, 4.5)

        plt.grid(True, which='both', axis='both', linestyle='dashed')
        for ind, leads in enumerate(self.anterior_leads):
            # print(ind, self.segments[leads])
            if self.segments[leads] is not None:
                if ind >= 0:
                    for noised_segment in self.segments[leads]:
                        self.strip.extend(noised_segment)
                        x = np.linspace(init, init + len(noised_segment), len(noised_segment))
                        self.x.extend(x)
                        plt.text(init + int(len(x) / 4), 1.0, '%s' % leads)
                        plt.axvline(init + len(x), ymin=0.3, ymax=0.6)
                        plt.plot(x, noised_segment, linewidth=1)
                        init += len(noised_segment)
                        break
        plt.title("12 ECG Leads: {}".format(self.pid))
        plt.savefig("AV_{}.png".format(self.pid), transparent=True)
        plt.show()

    def segment_plot(self):
        for pos, seg in enumerate(self.qrs_peak["Lead_II"]["index"]):
            plt.figure(figsize=(5, 8))
            plt.grid()
            for ind, leads in enumerate(self.anterior_leads):
                for i, denoised_segment in enumerate(self.segments_2[leads]):
                    if i == pos:
                        plt.plot(denoised_segment + (ind * 2))
                        # plt.plot(denoised_segment + (ind * 2), label=f"{leads}")
                        plt.axhline(np.average(denoised_segment) + (ind * 2))
                        # break
            # plt.legend()
            plt.show()


if __name__ == "__main__":
    print("Run mian.py")
