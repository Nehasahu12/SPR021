import numpy as np
import matplotlib.pyplot as plt
from utils import function as func
from wfdb import processing


def max_occurrence(arr):
    # Create a dictionary to store the count of each element
    counts = {}
    for num in arr:
        if num in counts:
            counts[num] += 1
        else:
            counts[num] = 1

    # Find the element with the highest count
    max_count = 0
    max_num = None
    for num, count in counts.items():
        if count > max_count:
            max_count = count
            max_num = num

    return max_num


def find_outlier_elements(data, lm=25, um=75):
    q1, q3 = np.percentile(data, [5, 95])
    iqr = q3 - q1
    # print("Q1:", q1)
    # print("Q3:", q3)
    # print("IQR:", iqr)

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = [x for x in data if x < lower_bound or x > upper_bound]
    return outliers


class QRS2DCorrection:
    def __init__(self, cleaned_signal, qrs_peaks,
                 sampling_rate=400, baseline=0) -> None:
        self.cleaned_signal = cleaned_signal
        self.qrs2d = []
        self.r_peak = qrs_peaks
        self.sampling_rate = sampling_rate
        self.baseline = baseline
        self.anterior_leads = ["Lead_I", "Lead_II", "Lead_III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        self.max_qrs_interval = 200  # milli second
        self.reference_row = 0
        self.outlier_leads = []
        self.persamtim = 1 / sampling_rate

        self.win_time = 150
        self.qrs_width = int((self.win_time / 1000) / self.persamtim)
        self.again_find = False
        self.__convert_dict_to_2d_list()

    def __convert_dict_to_2d_list(self):
        self.qrs2d = []
        for lead in self.anterior_leads:
            # print(f"{lead}\t\t{self.r_peak[lead]['index']}")
            self.qrs2d.append(self.r_peak[lead]['index'].tolist())
        # print(self.qrs2d)

    def correct_qrs_matrix(self):
        """
        Note: This Function can be optimized to reduce big O.
        """
        number_of_peaks = [len(i) for i in self.qrs2d]
        number_of_peak = max_occurrence(number_of_peaks)
        max_peak_nu = max(number_of_peaks)

        # Finding Reference Row for remove False Positive QRS peak
        self.reference_row = 0
        pv = 0
        for row in range(len(self.qrs2d)):
            if len(self.qrs2d[row]) == number_of_peak:
                dif = np.diff(self.qrs2d[row], 2)
                cv = abs(np.average(dif))
                if cv < pv:
                    self.reference_row = row
                pv = cv

        # print("Reference: ", self.reference_row)
        # print(self.qrs2d[self.reference_row])

        # Fill the blank items with the zero
        for row in range(len(self.qrs2d)):
            if not len(self.qrs2d[row]) == max_peak_nu:
                self.qrs2d[row] = self.qrs2d[row] + [0] * (max_peak_nu - len(self.qrs2d[row]))

        temp_row = self.qrs2d[self.reference_row]
        for i in range(len(self.qrs2d)):
            for current_col in range(max_peak_nu):
                target = self.qrs2d[i][current_col]
                distance = [abs(self.qrs2d[i][current_col] - tr) for tr in temp_row]
                target_col = np.argmin(distance)

                # checking current_ind with target value index
                if target_col != current_col:
                    self.outlier_leads.append(i)
                    d1 = abs(target - temp_row[target_col])
                    d2 = abs(self.qrs2d[i][target_col] - temp_row[target_col])
                    if d1 < d2:
                        # print(f"{target} : More Closer to target need Change")
                        self.qrs2d[i][target_col] = target
                    else:
                        # print(f"{self.qrs2d[i][target_col]} : More Closer to target")
                        pass
                    self.qrs2d[i][current_col] = temp_row[current_col]

        for i in range(len(self.qrs2d)):
            self.qrs2d[i] = self.qrs2d[i][:number_of_peak]

        if len(self.outlier_leads) == 0:
            # print("All Leads have equal qrs peak detected")
            self.again_find = False
        else:
            self.again_find = True
            # print("Missing or extra qrs peaks are filled and removed")

    def update_false_r_peak(self):
        for ind, lead in enumerate(self.anterior_leads):
            a = func.correct_qrs_peaks(self.cleaned_signal[lead], self.qrs2d[ind],
                                       np.array([self.cleaned_signal[lead][val] for val in self.qrs2d[ind]]))
            # print(a)
            # self.r_peak[lead]["index"] = self.qrs2d[ind]
            self.r_peak[lead]["index"] = a[0]
            # self.r_peak[lead]["value"] = np.array([self.cleaned_signal[lead][val] for val in self.qrs2d[ind]])
            self.r_peak[lead]["value"] = a[1]

    def correct_qrs_peak(self):
        """

        Not Working well
        """
        min_bpm = 20
        max_bpm = 230
        search_radius = int(self.sampling_rate * 60 / max_bpm)
        for ind, lead in enumerate(self.anterior_leads):
            # Correct the peaks shifting them to local maxima
            # min_gap = record.fs * 60 / min_bpm
            # Use the maximum possible bpm as the search radius
            corrected_peak_inds = processing.peaks.correct_peaks(self.cleaned_signal[lead],
                                                                 peak_inds=self.r_peak[lead]["index"],
                                                                 search_radius=search_radius,
                                                                 smooth_window_size=150)
            self.r_peak[lead]["index"] = corrected_peak_inds
            self.r_peak[lead]["value"] = np.array([self.cleaned_signal[lead][val] for val in corrected_peak_inds])

    def correct_r_peak(self, win_length=120):
        """

        Not Working well
        """
        for ind, lead in enumerate(self.anterior_leads):
            # c_ind, c_val = func.find_correct_qrs_peaks(self.cleaned_signal[lead], self.r_peak[lead]["index"],
            #                                            window_size=150, sampling_rate=500)

            c_ind, c_val = func.take_a_look_on_qrs(self.cleaned_signal[lead], self.r_peak[lead]["index"],
                                                   window_size=150, sampling_rate=500)

            self.r_peak[lead]["index"] = np.array(c_ind)
            self.r_peak[lead]["value"] = np.array([self.cleaned_signal[lead][i] for i in c_ind])


if __name__ == "__main__":
    print("This is module file")
