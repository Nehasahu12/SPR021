# import numpy as np
# import matplotlib.pyplot as plt
# from utils import function as func
# import pandas as pd
# from scipy import signal
# from scipy.fft import fft
# import neurokit2 as nk
# from utils import qrs_correction_algo as qrs_cor


# class QrsSignalFinder:
#     def __init__(self, x, peaks):
#         self.x = x
#         self.peak = peaks
#         self.q = []
#         self.s = []

#     # Trace QR Peaks
#     def trace_qr(self, baseline=0):
#         num_sample = []
#         for i in self.peak:
#             ln, li = func.left_deep(self.x, i)
#             rn, ri = func.right_deep(self.x, i)
#             self.q.append(li)
#             self.s.append(ri)
#             num_sample.append(ri - li)

#         # print("Q peak: ", self.q)
#         # print("S Peak: ", self.s)
#         # print("Number of Sample: ", num_sample)
#         # print("Remove Outlier: ", func.remove_outlier(num_sample))
#         # print("Base Line", baseline)
#         return func.remove_outlier(num_sample)


# class CalculateQRS:
#     """
#     class is used to clean the signal and find QRS peak globally.
#     """

#     def __init__(self, df, amp_gain=1.4, pid_number="123", low_cutoff_freq=100, high_cutoff_freq=0.5,
#                  filter_mode=1, sampling_rate=400):
#         """

#         :param df:
#         :param amp_gain:
#         :param pid_number:
#         :param low_cutoff_freq:
#         :param high_cutoff_freq:
#         :param filter_mode: 1 -> for wavelate method, 0 -> for simple filter
#         """
#         self.qrs2d = None
#         self.two_peaks = []
#         self.qrs_m = {}
#         self.mode = filter_mode
#         self.qrs_interval = None
#         self.rr_interval = {}
#         self.indices = None
#         self.baseLine = None
#         self.beats = None
#         self.outlier = None
#         self.df = df
#         self.GAIN = amp_gain
#         self.pid = pid_number
#         self.qrs_result = {}
#         self.r_peak = {}
#         self.cleaned_signal = {}
#         self.max_sig = {}
#         self.beat_count = {}
#         self.qrs_segments = {}
#         self.len_of_qrs = []
#         self.number_of_beats = {}
#         self.isoelectric = {}
#         self.qrs_threshold = 6.0
#         self.persamtim = 1 / sampling_rate
#         self.win_time = 350
#         self.qrs_distance = int((self.win_time / 1000) / self.persamtim)
#         # print(f"QRS Margin: {self.qrs_distance}")
#         self.margin = 5
#         self.heart_rate = {}
#         self.lr_qr = [35, 35, 35, 35, 35]
#         self.low_cutoff_freq = low_cutoff_freq
#         self.high_cutoff_freq = high_cutoff_freq
#         self.sampling_rate = sampling_rate
#         # self.anterior_leads = ["Lead_II", "Lead_I", "Lead_III", "V1", "V2", "V3", "V4", "V5", "V6"]
#         self.anterior_leads = ["Lead_II"]
#         # self.anterior_leads = ["Lead_I"]
#         self.tm = np.linspace(0, 12, len(df[self.anterior_leads[0]]))
#         self.win_time = 150
#         self.qrs_width = int((self.win_time / 1000) / self.persamtim)

#     def clean_all_data(self):
#         for leads in self.anterior_leads:
#             if self.mode == 1:
#                 clean_signal = func.highpass(func.denoise_signal(
#                     func.lowpass(func.highpass(self.df[leads], CUT_OFF_FREQUENCY=self.high_cutoff_freq,
#                                                SAMPLING_FREQUENCY=self.sampling_rate),
#                                  CUT_OFF_FREQUENCY=self.low_cutoff_freq,
#                                  SAMPLING_FREQUENCY=self.sampling_rate), 'bior4.4', 9, 1, 7) * self.GAIN)
#             elif self.mode == 2:
#                 # clean_signal = func.bandpass(self.df[leads], FL=1, FH=20)
#                 clean_signal = func.lowpass(func.highpass(self.df[leads], CUT_OFF_FREQUENCY=self.high_cutoff_freq,
#                                                           SAMPLING_FREQUENCY=self.sampling_rate),
#                                             CUT_OFF_FREQUENCY=self.low_cutoff_freq,
#                                             SAMPLING_FREQUENCY=self.sampling_rate)

#             elif self.mode == 3:
#                 clean_signal = nk.ecg_clean(self.df[leads], sampling_rate=self.sampling_rate, method="neurokit")
#             else:
#                 clean_signal = nk.ecg_clean(self.df[leads], sampling_rate=self.sampling_rate, method="neurokit")

#             self.cleaned_signal[leads] = clean_signal
#             self.tm = np.linspace(0, 12, len(clean_signal))
#         #     plt.plot(self.df[leads])
#         #     plt.plot(self.tm, clean_signal)
#         #     plt.plot(np.diff(clean_signal))
#         # plt.legend(self.anterior_leads)
#         # plt.show()

#         # Calculate QRS Peak
#         for leads in self.anterior_leads:
#             maximized_signal = func.maximize_qrs_signal(func.lowpass(func.highpass(self.df[leads],
#                                                                                    SAMPLING_FREQUENCY=self.sampling_rate),
#                                                                      SAMPLING_FREQUENCY=self.sampling_rate,
#                                                                      CUT_OFF_FREQUENCY=15),
#                                                         length=30, sigma=1.8, mode='same', max_height=True)

#             # maximized_signal = func.derivative_filter(maximized_signal)
#             start = 50
#             maximized_signal[0:start] = 0
#             maximized_signal[len(maximized_signal) - 50:len(maximized_signal)] = 0
#             self.max_sig[leads] = maximized_signal
#             # maximized_signal **= 2
#             # Finding the QRS peaks threshold and QRS peaks based on polynomial fit
#             threshold = func.find_threshold(maximized_signal, div_factor=self.qrs_threshold)
#             # print(f"threshold to mean: {threshold}")
#             if threshold > 3.0 or maximized_signal.min() < -3.0:
#                 threshold = maximized_signal.mean() + 0.8

#             # print(f"QRS Threshold: {threshold}, {leads}")
#             # Find peaks of QRS
#             #  if finter mode is neurikits then
#             if self.mode != 3:
#                 qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=self.qrs_distance)
#                 peak_apmlitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])

#                 # print(f"Peaks: {qrs_peaks[0]}")
#                 #
#                 # plt.title(leads)
#                 # plt.plot(self.cleaned_signal[leads])
#                 # plt.plot(maximized_signal)
#                 # plt.axhline(threshold)
#                 # plt.axhline(maximized_signal.mean() + 0.35, color='red')
#                 # plt.show()
#                 # Correct peaks of QRS
#                 corrected_peaks = func.correct_qrs_peaks(self.cleaned_signal[leads], qrs_peaks[0], peak_apmlitude)
#                 # print("Line 143: ", corrected_peaks)
#                 # print(self.cleaned_signal[leads])
#                 self.baseLine = func.baseline_ml_finder(self.cleaned_signal[leads], corrected_peaks)
#                 # print(f"Base Line ML: {self.baseLine}")

#             else:
#                 qrs_peaks = [nk.ecg_findpeaks(self.cleaned_signal[leads], sampling_rate=self.sampling_rate,
#                                               method="neurokit", show=False)["ECG_R_Peaks"]]

#                 peak_apmlitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])
#                 # print("Line 149", qrs_peaks[0], peak_apmlitude)
#                 corrected_peaks = [qrs_peaks[0], peak_apmlitude]
#                 # print("Line 151: ", corrected_peaks)
#                 self.baseLine = func.baseline_ml_finder(self.cleaned_signal[leads], corrected_peaks)

#             self.r_peak[leads] = {}
#             # self.r_peak[leads]["index"] = qrs_peaks[0] # corrected_peaks[0]
#             # self.r_peak[leads]["value"] = peak_apmlitude # corrected_peaks[1]

#             self.r_peak[leads]["index"] = corrected_peaks[0]
#             self.r_peak[leads]["value"] = corrected_peaks[1]
#             self.isoelectric[leads] = self.baseLine[-1]
#         # print("r Peaks: ", self.r_peak)

#         # plt.legend(self.anterior_leads)
#         # plt.show()

#         # # For FFT calculation =====================
#         # for leads in self.anterior_leads:
#         #     # fre = fft(func.lowpass(self.df[leads].values), n=self.df[leads].size)
#         #     fre = func.calculate_fft(self.cleaned_signal[leads], self.sampling_rate)
#         #     # plt.plot(fre.real[1:500])
#         #     plt.plot(fre[0], fre[1])
#         #     # break
#         # plt.legend(self.anterior_leads)
#         # plt.show()

#     def check_m_shape(self):
#         y = np.linspace(1, 1, 5)
#         for leads in self.anterior_leads:
#             self.two_peaks = []
#             temp = []
#             for r in self.r_peak[leads]["index"]:
#                 start = np.int16(np.average(r)) - int(self.qrs_width / 2)
#                 stop = np.int16(np.average(r)) + int(self.qrs_width / 2)
#                 # self.qrs_cor.append([start, stop])
#                 start = 0 if start < 0 else start
#                 stop = len(self.cleaned_signal[leads]) if stop > len(self.cleaned_signal[leads]) else stop
#                 # print(start, stop)

#                 yy = np.convolve(self.cleaned_signal[leads][start:stop], y, 'same')
#                 th = yy.max() - (yy.max() / 2)
#                 lp = signal.find_peaks(yy, distance=4, height=th)
#                 temp.append(len(lp[0]))
#                 plt.plot(yy)
#                 plt.axhline(th)
#             two = temp.count(2)
#             one = temp.count(1)
#             if two > one:
#                 self.qrs_m[leads] = 1
#             else:
#                 self.qrs_m[leads] = 0
#             # print("Number of Peak in QRS complex: ", leads, temp)
#             plt.show()
#             # print("Line 185: ", self.qrs_cor)
#         # if self.qrs_m["V6"]:
#         #     # print("Possible LBBB")
#         # else:
#         #     print("qrs_cal Line 208 Normal")

#     def scatter_r_peak(self):
#         # plt.cla()
#         # plt.figure(3)
#         # plt.ylim(-1, 2)
#         for leads in self.anterior_leads:
#             self.beats = len(self.r_peak[leads]["index"])
#             self.number_of_beats[leads] = self.beats

#             # Testing
#             # print(self.isoelectric[leads])
#             # self.cleaned_signal[leads] = self.cleaned_signal[leads] - self.isoelectric[leads]
#             # testqt = QrsSignalFinder(self.cleaned_signal[leads], self.r_peak[leads]["index"])
#             # self.lr_qr = testqt.trace_qr(baseline=self.isoelectric[leads])
#             # ==================================

#             # print(self.r_peak[leads]["index"])
#             self.qrs_segments[leads] = {}
#             ind = []
#             val = []
#             for i in self.r_peak[leads]["index"]:
#                 esps = func.qrs_signal_finder(self.cleaned_signal[leads], i)
#                 ind.append(esps[1])
#                 val.append(esps[0])
#             self.qrs_segments[leads]["value"] = val
#             self.qrs_segments[leads]["index"] = ind
#             # plt.title(leads+" -> "+str(self.beats))
#             # plt.plot(self.cleaned_signal[leads])
#             # plt.axhline(self.isoelectric[leads])
#             # plt.axhline(0, color='red', linestyle="--")
#             # plt.scatter(self.r_peak[leads]["index"], self.r_peak[leads]["value"],  s=100, color='red')
#         # plt.savefig("ECG_GRAPH/R_PEAK/{}.png".format(self.pid))
#         # plt.show()

#         # check If any Outlier Peaks in the all signal
#         # print(self.r_peak)
#         # self.r_peak = func.validate_peaks(self.r_peak, val=20)
#         # print(self.r_peak)
#         self.fix_qrs_peak()

#     def scatter_qrs(self, ind_v=False, show=False):
#         ind_v = False
#         if show:
#             for leads in self.anterior_leads:
#                 # if ind_v: plt.figure()
#                 # with plt.style.context('dark_background'):
#                 plt.title(leads + " - pid After: " + str(self.pid))
#                 plt.plot(self.cleaned_signal[leads])
#                 for ind, val in zip(self.qrs_segments[leads]["index"], self.qrs_segments[leads]["value"]):
#                     plt.scatter(ind, val, color='red', s=10)
#                 # plt.plot(self.maxsig[leads])
#                 plt.scatter(self.r_peak[leads]["index"], self.r_peak[leads]["value"], s=100, color='red')
#                 # plt.text(self.r_peak[leads]["index"], self.r_peak[leads]["value"], "R")
#                 plt.axhline(self.isoelectric[leads])
#                 plt.axhline(0, color="red", linestyle="--")
#                 if ind_v:plt.show()
#             if not ind_v:plt.show()

#     def fix_qrs_peak(self):
#         self.indices = []
#         # print("R peaks: ", self.number_of_beats)
#         self.outlier = func.outlier_peaks_detector(self.number_of_beats, margin=self.margin)
#         # print("Outlier: ", self.outlier)
#         # local_btc = 15
#         # for ld in self.anterior_leads:
#         #     if ld not in self.outlier:
#         #         local_btc = self.number_of_beats[ld]

#         for leads in self.outlier:
#             maximized_signal = func.maximize_qrs_signal(func.lowpass(func.highpass(self.df[leads],
#                                                                                    SAMPLING_FREQUENCY=self.sampling_rate),
#                                                                      SAMPLING_FREQUENCY=self.sampling_rate,
#                                                                      CUT_OFF_FREQUENCY=15),
#                                                         length=30, sigma=1.8, mode='same', max_height=False)
#             start = 10
#             maximized_signal[0:start] = 0
#             maximized_signal[len(maximized_signal) - 50:len(maximized_signal)] = 0
#             self.max_sig[leads] = maximized_signal
#             # Finding the QRS peaks threshold and QRS peaks based on polynomial fit

#             # adjust threshold for peak detection
#             self.qrs_threshold -= 1
#             threshold = func.find_threshold(maximized_signal, div_factor=self.qrs_threshold)

#             # Find peaks of QRS
#             qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=self.qrs_distance)
#             peak_amplitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])

#             # Correct peaks of QRS
#             corrected_peaks = func.correct_qrs_peaks(self.cleaned_signal[leads], qrs_peaks[0], peak_amplitude)
#             self.r_peak[leads] = {}
#             self.r_peak[leads]["index"] = corrected_peaks[0]
#             self.r_peak[leads]["value"] = corrected_peaks[1]

#         # print("Outlier after: ", self.outlier)
#         if len(self.outlier) > 0:
#             # Adjusting std range
#             self.margin += 0.5
#             self.scatter_r_peak()

#     def get_heart_rate(self, sampling_rate=400):
#         """

#         :param sampling_rate:
#         :return:
#         """
#         per_sample = 1 / sampling_rate
#         for leads in self.anterior_leads:
#             samp = np.average(np.diff(self.r_peak[leads]["index"]))
#             hr = 60 / (samp * per_sample)
#             self.heart_rate[leads] = hr

#             # RR interval in millisecond
#             self.rr_interval[leads] = samp * per_sample * 1000

#         # print("[::] Heart Rate: ", self.heart_rate)
#         return self.heart_rate

#     def get_rr_interval(self, lead=None):
#         # print(self.rr_interval)
#         if lead is not None: return self.rr_interval[lead]
#         return self.rr_interval

#     def get_r_peaks(self, show=False):
#         """
#         return: Dict of all r peaks
#         """
#         # if show: print(self.r_peak)
#         return self.r_peak

#     def get_qrs(self, fs=400, t=10):
#         self.len_of_qrs = []
#         for leads in self.anterior_leads:
#             temp = []
#             for ind, val in zip(self.qrs_segments[leads]["index"], self.qrs_segments[leads]["value"]):
#                 temp.append(len(ind))
#             self.len_of_qrs.extend(temp)
#         # print("[::] Prev: ", self.len_of_qrs)
#         qrs_sample = func.remove_outlier(self.len_of_qrs, margin=1.1)
#         # print("[::] Next: ", list(qrs_sample))
#         # print("[::] QRS: ", np.round(np.mean(qrs_sample)), "sample")
#         self.qrs_interval = round((1 / fs) * np.round(np.mean(qrs_sample)) * 1000)
#         # print("[::] QRS Interval: ", self.qrs_interval, "ms")
#         qrs_interval1 = round((1 / fs) * np.round(np.mean(self.lr_qr)) * 1000)
#         # print("[::] new_method: QRS Interval: ", qrs_interval1, "ms")
#         return self.qrs_interval

#     def type_of_qrs(self):
#         """
#         :return: type of QRS > Narrow, wide
#         """
#         if self.qrs_interval > 120:
#             return "Wide"
#         else:
#             return "Narrow"

#     def find_missing_qrs(self):
#         temp_list = []
#         qrs2d = []
#         # print(self.r_peak)
#         for lead in self.anterior_leads:
#             # print(f"{lead}\t\t{self.r_peak[lead]['index']}")
#             qrs2d.append(self.r_peak[lead]['index'].tolist())
#             temp_list.extend(self.r_peak[lead]['index'])
#         temp_list.sort()
#         # print(qrs2d)
#         # plt.plot(temp_list, '.-')
#         # plt.show()

#         # QRS Correction
#         abcd = qrs_cor.QRS2DCorrection(self.cleaned_signal, self.r_peak)
#         abcd.correct_qrs_matrix()
#         abcd.update_false_r_peak()
#         # abcd.correct_qrs_peak()
#         # abcd.correct_r_peak()
#         self.r_peak = abcd.r_peak
#         # print("Missing QRS peaks are Updated by False R peak")

#     def convert_dict_to_2d_list(self):
#         self.qrs2d = []
#         for lead in self.anterior_leads:
#             # print(f"{lead}\t\t{self.r_peak[lead]['index']}")
#             self.qrs2d.append(self.r_peak[lead]['index'].tolist())
#         # print(self.qrs2d)


# if __name__ == "__main__":
#     print(pid := "123", "qrs_cal.py file called")
#     # print(name := "Nitish Kumar")


import numpy as np
import matplotlib.pyplot as plt
from utils import function as func
import pandas as pd
from scipy import signal
from scipy.fft import fft
import neurokit2 as nk
from utils import qrs_correction_algo as qrs_cor


class QrsSignalFinder:
    def __init__(self, x, peaks):
        self.x = x
        self.peak = peaks
        self.q = []
        self.s = []

    # Trace QR Peaks
    def trace_qr(self, baseline=0):
        num_sample = []
        for i in self.peak:
            ln, li = func.left_deep(self.x, i)
            rn, ri = func.right_deep(self.x, i)
            self.q.append(li)
            self.s.append(ri)
            num_sample.append(ri - li)

        # print("Q peak: ", self.q)
        # print("S Peak: ", self.s)
        # print("Number of Sample: ", num_sample)
        # print("Remove Outlier: ", func.remove_outlier(num_sample))
        # print("Base Line", baseline)
        return func.remove_outlier(num_sample)


class CalculateQRS:
    """
    class is used to clean the signal and find QRS peak globally.
    """

    def __init__(self, df, amp_gain=1, pid_number="123", low_cutoff_freq=50, high_cutoff_freq=0.5,
                 filter_mode=1, sampling_rate=100):
        """

        :param df:
        :param amp_gain:
        :param pid_number:
        :param low_cutoff_freq:
        :param high_cutoff_freq:
        :param filter_mode: 1 -> for wavelate method, 0 -> for simple filter
        """
        self.qrs2d = None
        self.two_peaks = []
        self.qrs_m = {}
        self.mode = filter_mode
        self.qrs_interval = None
        self.rr_interval = {}
        self.indices = None
        self.baseLine = None
        self.beats = None
        self.outlier = None
        self.df = df
        self.GAIN = amp_gain
        self.pid = pid_number
        self.qrs_result = {}
        self.r_peak = {}
        self.cleaned_signal = {}
        self.max_sig = {}
        self.beat_count = {}
        self.qrs_segments = {}
        self.len_of_qrs = []
        self.number_of_beats = {}
        self.isoelectric = {}
        self.qrs_threshold = 6.0
        self.persamtim = 1 / sampling_rate
        self.win_time = 350
        self.qrs_distance = int((self.win_time / 1000) / self.persamtim)
        # print(f"QRS Margin: {self.qrs_distance}")
        self.margin = 5
        self.heart_rate = {}
        self.lr_qr = [35, 35, 35, 35, 35]
        self.low_cutoff_freq = low_cutoff_freq
        self.high_cutoff_freq = high_cutoff_freq
        self.sampling_rate = sampling_rate
        # self.anterior_leads = ["Lead_II", "Lead_I", "Lead_III", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.anterior_leads = ["Lead_II"]
        # self.anterior_leads = ["Lead_I"]
        self.tm = np.linspace(0, 12, len(df[self.anterior_leads[0]]))
        self.win_time = 150
        self.qrs_width = int((self.win_time / 1000) / self.persamtim)

    def clean_all_data(self):
        for leads in self.anterior_leads:
            if self.mode == 1:
                clean_signal = func.highpass(func.denoise_signal(
                    func.lowpass(func.highpass(self.df[leads], CUT_OFF_FREQUENCY=self.high_cutoff_freq,
                                               SAMPLING_FREQUENCY=self.sampling_rate),
                                 CUT_OFF_FREQUENCY=self.low_cutoff_freq,
                                 SAMPLING_FREQUENCY=self.sampling_rate), 'bior4.4', 9, 1, 7) * self.GAIN)
            elif self.mode == 2:
                # clean_signal = func.bandpass(self.df[leads], FL=1, FH=20)
                clean_signal = func.lowpass(func.highpass(self.df[leads], CUT_OFF_FREQUENCY=self.high_cutoff_freq,
                                                          SAMPLING_FREQUENCY=self.sampling_rate),
                                            CUT_OFF_FREQUENCY=self.low_cutoff_freq,
                                            SAMPLING_FREQUENCY=self.sampling_rate)

            elif self.mode == 3:
                clean_signal = nk.ecg_clean(self.df[leads], sampling_rate=self.sampling_rate, method="neurokit")
            else:
                clean_signal = nk.ecg_clean(self.df[leads], sampling_rate=self.sampling_rate, method="neurokit")

            self.cleaned_signal[leads] = clean_signal
            self.tm = np.linspace(0, 12, len(clean_signal))
        #     plt.plot(self.df[leads])
        #     plt.plot(self.tm, clean_signal)
        #     plt.plot(np.diff(clean_signal))
        # plt.legend(self.anterior_leads)
        # plt.show()

        # Calculate QRS Peak
        for leads in self.anterior_leads:
            maximized_signal = func.maximize_qrs_signal(func.lowpass(func.highpass(self.df[leads],
                                                                                   SAMPLING_FREQUENCY=self.sampling_rate),
                                                                     SAMPLING_FREQUENCY=self.sampling_rate,
                                                                     CUT_OFF_FREQUENCY=15),
                                                        length=30, sigma=1.8, mode='same', max_height=True)

            # maximized_signal = func.derivative_filter(maximized_signal)
            start = 50
            maximized_signal[0:start] = 0
            maximized_signal[len(maximized_signal) - 50:len(maximized_signal)] = 0
            self.max_sig[leads] = maximized_signal
            # maximized_signal **= 2
            # Finding the QRS peaks threshold and QRS peaks based on polynomial fit
            threshold = func.find_threshold(maximized_signal, div_factor=self.qrs_threshold)
            # print(f"threshold to mean: {threshold}")
            if threshold > 3.0 or maximized_signal.min() < -3.0:
                threshold = maximized_signal.mean() + 0.8

            # print(f"QRS Threshold: {threshold}, {leads}")
            # Find peaks of QRS
            #  if finter mode is neurikits then
            if self.mode != 3:
                qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=self.qrs_distance)
                peak_apmlitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])

                # print(f"Peaks: {qrs_peaks[0]}")
                #
                # plt.title(leads)
                # plt.plot(self.cleaned_signal[leads])
                # plt.plot(maximized_signal)
                # plt.axhline(threshold)
                # plt.axhline(maximized_signal.mean() + 0.35, color='red')
                # plt.show()
                # Correct peaks of QRS
                corrected_peaks = func.correct_qrs_peaks(self.cleaned_signal[leads], qrs_peaks[0], peak_apmlitude)
                # print("Line 143: ", corrected_peaks)
                # print(self.cleaned_signal[leads])
                self.baseLine = func.baseline_ml_finder(self.cleaned_signal[leads], corrected_peaks)
                # print(f"Base Line ML: {self.baseLine}")

            else:
                qrs_peaks = [nk.ecg_findpeaks(self.cleaned_signal[leads], sampling_rate=self.sampling_rate,
                                              method="neurokit", show=False)["ECG_R_Peaks"]]

                peak_apmlitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])
                # print("Line 149", qrs_peaks[0], peak_apmlitude)
                corrected_peaks = [qrs_peaks[0], peak_apmlitude]
                # print("Line 151: ", corrected_peaks)
                self.baseLine = func.baseline_ml_finder(self.cleaned_signal[leads], corrected_peaks)

            self.r_peak[leads] = {}
            # self.r_peak[leads]["index"] = qrs_peaks[0] # corrected_peaks[0]
            # self.r_peak[leads]["value"] = peak_apmlitude # corrected_peaks[1]

            self.r_peak[leads]["index"] = corrected_peaks[0]
            self.r_peak[leads]["value"] = corrected_peaks[1]
            self.isoelectric[leads] = self.baseLine[-1]
        # print("r Peaks: ", self.r_peak)

        # plt.legend(self.anterior_leads)
        # plt.show()

        # # For FFT calculation =====================
        # for leads in self.anterior_leads:
        #     # fre = fft(func.lowpass(self.df[leads].values), n=self.df[leads].size)
        #     fre = func.calculate_fft(self.cleaned_signal[leads], self.sampling_rate)
        #     # plt.plot(fre.real[1:500])
        #     plt.plot(fre[0], fre[1])
        #     # break
        # plt.legend(self.anterior_leads)
        # plt.show()

    def check_m_shape(self):
        y = np.linspace(1, 1, 5)
        for leads in self.anterior_leads:
            self.two_peaks = []
            temp = []
            for r in self.r_peak[leads]["index"]:
                start = np.int16(np.average(r)) - int(self.qrs_width / 2)
                stop = np.int16(np.average(r)) + int(self.qrs_width / 2)
                # self.qrs_cor.append([start, stop])
                start = 0 if start < 0 else start
                stop = len(self.cleaned_signal[leads]) if stop > len(self.cleaned_signal[leads]) else stop
                # print(start, stop)

                yy = np.convolve(self.cleaned_signal[leads][start:stop], y, 'same')
                th = yy.max() - (yy.max() / 2)
                lp = signal.find_peaks(yy, distance=4, height=th)
                temp.append(len(lp[0]))
                plt.plot(yy)
                plt.axhline(th)
            two = temp.count(2)
            one = temp.count(1)
            if two > one:
                self.qrs_m[leads] = 1
            else:
                self.qrs_m[leads] = 0
            # print("Number of Peak in QRS complex: ", leads, temp)
            plt.show()
            # print("Line 185: ", self.qrs_cor)
        # if self.qrs_m["V6"]:
        #     # print("Possible LBBB")
        # else:
        #     print("qrs_cal Line 208 Normal")

    def scatter_r_peak(self):
        # plt.cla()
        # plt.figure(3)
        # plt.ylim(-1, 2)
        for leads in self.anterior_leads:
            self.beats = len(self.r_peak[leads]["index"])
            self.number_of_beats[leads] = self.beats

            # Testing
            # print(self.isoelectric[leads])
            # self.cleaned_signal[leads] = self.cleaned_signal[leads] - self.isoelectric[leads]
            # testqt = QrsSignalFinder(self.cleaned_signal[leads], self.r_peak[leads]["index"])
            # self.lr_qr = testqt.trace_qr(baseline=self.isoelectric[leads])
            # ==================================

            # print(self.r_peak[leads]["index"])
            self.qrs_segments[leads] = {}
            ind = []
            val = []
            for i in self.r_peak[leads]["index"]:
                esps = func.qrs_signal_finder(self.cleaned_signal[leads], i)
                ind.append(esps[1])
                val.append(esps[0])
            self.qrs_segments[leads]["value"] = val
            self.qrs_segments[leads]["index"] = ind
            # plt.title(leads+" -> "+str(self.beats))
            # plt.plot(self.cleaned_signal[leads])
            # plt.axhline(self.isoelectric[leads])
            # plt.axhline(0, color='red', linestyle="--")
            # plt.scatter(self.r_peak[leads]["index"], self.r_peak[leads]["value"],  s=100, color='red')
        # plt.savefig("ECG_GRAPH/R_PEAK/{}.png".format(self.pid))
        # plt.show()

        # check If any Outlier Peaks in the all signal
        # print(self.r_peak)
        # self.r_peak = func.validate_peaks(self.r_peak, val=20)
        # print(self.r_peak)
        self.fix_qrs_peak()

    def scatter_qrs(self, ind_v=False, show=False):
        ind_v = False
        if show:
            for leads in self.anterior_leads:
                # if ind_v: plt.figure()
                # with plt.style.context('dark_background'):
                plt.title(leads + " - pid After: " + str(self.pid))
                plt.plot(self.cleaned_signal[leads])
                for ind, val in zip(self.qrs_segments[leads]["index"], self.qrs_segments[leads]["value"]):
                    plt.scatter(ind, val, color='red', s=10)
                # plt.plot(self.maxsig[leads])
                plt.scatter(self.r_peak[leads]["index"], self.r_peak[leads]["value"], s=100, color='red')
                # plt.text(self.r_peak[leads]["index"], self.r_peak[leads]["value"], "R")
                plt.axhline(self.isoelectric[leads])
                plt.axhline(0, color="red", linestyle="--")
                if ind_v:plt.show()
            if not ind_v:plt.show()

    def fix_qrs_peak(self):
        self.indices = []
        # print("R peaks: ", self.number_of_beats)
        self.outlier = func.outlier_peaks_detector(self.number_of_beats, margin=self.margin)
        # print("Outlier: ", self.outlier)
        # local_btc = 15
        # for ld in self.anterior_leads:
        #     if ld not in self.outlier:
        #         local_btc = self.number_of_beats[ld]

        for leads in self.outlier:
            maximized_signal = func.maximize_qrs_signal(func.lowpass(func.highpass(self.df[leads],
                                                                                   SAMPLING_FREQUENCY=self.sampling_rate),
                                                                     SAMPLING_FREQUENCY=self.sampling_rate,
                                                                     CUT_OFF_FREQUENCY=15),
                                                        length=30, sigma=1.8, mode='same', max_height=False)
            start = 10
            maximized_signal[0:start] = 0
            maximized_signal[len(maximized_signal) - 50:len(maximized_signal)] = 0
            self.max_sig[leads] = maximized_signal
            # Finding the QRS peaks threshold and QRS peaks based on polynomial fit

            # adjust threshold for peak detection
            self.qrs_threshold -= 1
            threshold = func.find_threshold(maximized_signal, div_factor=self.qrs_threshold)

            # Find peaks of QRS
            qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=self.qrs_distance)
            peak_amplitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])

            # Correct peaks of QRS
            corrected_peaks = func.correct_qrs_peaks(self.cleaned_signal[leads], qrs_peaks[0], peak_amplitude)
            self.r_peak[leads] = {}
            self.r_peak[leads]["index"] = corrected_peaks[0]
            self.r_peak[leads]["value"] = corrected_peaks[1]

        # print("Outlier after: ", self.outlier)
        if len(self.outlier) > 0:
            # Adjusting std range
            self.margin += 0.5
            self.scatter_r_peak()

    def get_heart_rate(self, sampling_rate=100):
        """

        :param sampling_rate:
        :return:
        """
        per_sample = 1 / sampling_rate
        for leads in self.anterior_leads:
            samp = np.average(np.diff(self.r_peak[leads]["index"]))
            hr = 60 / (samp * per_sample)
            self.heart_rate[leads] = hr

            # RR interval in millisecond
            self.rr_interval[leads] = samp * per_sample * 1000

        # print("[::] Heart Rate: ", self.heart_rate)
        return self.heart_rate

    def get_rr_interval(self, lead=None):
        # print(self.rr_interval)
        if lead is not None: return self.rr_interval[lead]
        return self.rr_interval

    def get_r_peaks(self, show=False):
        """
        return: Dict of all r peaks
        """
        # if show: print(self.r_peak)
        return self.r_peak

    def get_qrs(self, fs=100, t=10):
        self.len_of_qrs = []
        for leads in self.anterior_leads:
            temp = []
            for ind, val in zip(self.qrs_segments[leads]["index"], self.qrs_segments[leads]["value"]):
                temp.append(len(ind))
            self.len_of_qrs.extend(temp)
        # print("[::] Prev: ", self.len_of_qrs)
        qrs_sample = func.remove_outlier(self.len_of_qrs, margin=1.1)
        # print("[::] Next: ", list(qrs_sample))
        # print("[::] QRS: ", np.round(np.mean(qrs_sample)), "sample")
        self.qrs_interval = round((1 / fs) * np.round(np.mean(qrs_sample)) * 1000)
        # print("[::] QRS Interval: ", self.qrs_interval, "ms")
        qrs_interval1 = round((1 / fs) * np.round(np.mean(self.lr_qr)) * 1000)
        # print("[::] new_method: QRS Interval: ", qrs_interval1, "ms")
        return self.qrs_interval

    def type_of_qrs(self):
        """
        :return: type of QRS > Narrow, wide
        """
        if self.qrs_interval > 120:
            return "Wide"
        else:
            return "Narrow"

    def find_missing_qrs(self):
        temp_list = []
        qrs2d = []
        # print(self.r_peak)
        for lead in self.anterior_leads:
            # print(f"{lead}\t\t{self.r_peak[lead]['index']}")
            qrs2d.append(self.r_peak[lead]['index'].tolist())
            temp_list.extend(self.r_peak[lead]['index'])
        temp_list.sort()
        # print(qrs2d)
        # plt.plot(temp_list, '.-')
        # plt.show()

        # QRS Correction
        abcd = qrs_cor.QRS2DCorrection(self.cleaned_signal, self.r_peak)
        abcd.correct_qrs_matrix()
        abcd.update_false_r_peak()
        # abcd.correct_qrs_peak()
        # abcd.correct_r_peak()
        self.r_peak = abcd.r_peak
        # print("Missing QRS peaks are Updated by False R peak")

    def convert_dict_to_2d_list(self):
        self.qrs2d = []
        for lead in self.anterior_leads:
            # print(f"{lead}\t\t{self.r_peak[lead]['index']}")
            self.qrs2d.append(self.r_peak[lead]['index'].tolist())
        # print(self.qrs2d)


if __name__ == "__main__":
    print(pid := "123", "qrs_cal.py file called")
    # print(name := "Nitish Kumar")
