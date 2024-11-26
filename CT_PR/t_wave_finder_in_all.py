from utils import function as func
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from copy import deepcopy
from utils.custom_print import print

closest_point = 0


def find_maximum_occurrence(lst):
    # Use a Counter to count the occurrences of each item
    counter = Counter(lst)
    # Get a list of the items with the maximum occurrence
    most_common = counter.most_common()
    # The first item in the list is the one with the maximum occurrence
    maximum_occurrence_item = most_common[0]
    return maximum_occurrence_item[0], maximum_occurrence_item[1]


class TDetector:
    def __init__(self, cleaned_signal, qrs_peak, sampling_rate=400, margin=0,
                 baseline=None):
        self.j_point_2d = None
        self.j_value_2d = None
        self.qrs_ver_segments = None
        self.qrs_hor_segments = None
        self.ver_segments = None
        self.hor_segments = None
        self.qrs_cor = None
        self.segments = None
        self.cor = None
        self.st_segments = None
        self.avg_line = None
        self.center_point = None
        if baseline is None:
            baseline = {"Lead_II": 0}
        self.cleaned_signal = deepcopy(cleaned_signal)
        self.r_peaks = deepcopy(qrs_peak)
        self.sampling_rate = sampling_rate
        self.margin = margin
        self.persamtim = 1 / sampling_rate
        self.ecg_leads = ["Lead_II"]
        self.baseline = baseline
        self.win_time = 200
        self.qrs_width = int((self.win_time / 1000) / self.persamtim)
        # print(self.cleaned_signal["aVL"])

    def adjust_baseline(self, show=False, adjust_sig=True):
        for lead in self.ecg_leads:
            bs = func.baseline_finder(self.cleaned_signal[lead], [self.r_peaks[lead]["index"]], margin=self.margin)
            # print(f"{lead} Base Line: {np.average(bs[1])}")
            self.baseline = np.average(bs[0])

            if adjust_sig:
                if len(self.r_peaks[lead]["value"]) > 0:
                    self.cleaned_signal[lead] = self.cleaned_signal[lead] - np.average(bs[1])
                    self.r_peaks[lead]["value"] = self.r_peaks[lead]["value"] - np.average(bs[1])
                    self.baseline = 0

        temp = []
        self.center_point = []

        # Correct selection
        a = []
        for lead in self.ecg_leads:
            a.append(len(self.r_peaks[lead]['index']))
        max_occur_val, _ = find_maximum_occurrence(a)
        new_ecg_lead = []
        for i, v in enumerate(a):
            if v == max_occur_val:
                # print(i, v, max_occur_val, self.ecg_leads[i])
                new_ecg_lead.append(self.ecg_leads[i])

        for lead in new_ecg_lead:
            # print(f"Leads: {lead}, len: {len(self.r_peaks[lead]['index'])}")
            if len(self.r_peaks[lead]["index"]) > 0:
                temp.append(self.r_peaks[lead]["index"])  # QRS peak index
                self.center_point.append(func.find_center_point(self.cleaned_signal[lead],
                                                                self.r_peaks[lead]["index"])[0])  # center point
            else:
                print(f"Signal Error in: {lead}")
        self.center_point = np.array(self.center_point).T
        self.center_point = [np.int16(np.average(i)) for i in self.center_point]
        # print(f"Center Point: {self.center_point}")
        self.avg_line = np.array(temp).T
        np.save(r"qrs_peak.npy", self.avg_line)

        if show:
            shift = 3
            plt.title("QRS Signal Peak", fontsize=14, color="teal")
            plt.grid()
            for ind, lead in enumerate(self.ecg_leads):
                plt.plot(self.cleaned_signal[lead] + (ind * shift))
                plt.text(0, ind * shift + 1, lead)
                # plt.axhline((ind * shift), color='black')
                if len(self.r_peaks[lead]["index"]) > 0:
                    plt.scatter(self.r_peaks[lead]["index"], self.r_peaks[lead]["value"] + (ind * shift))

                for i in self.avg_line:
                    plt.axvline(np.average(i), alpha=0.6)
                    plt.axvline(np.average(i) - int(self.qrs_width / 2), color="coral", alpha=0.5)
                    plt.axvline(np.average(i) + int(self.qrs_width / 2), color="coral", alpha=0.5)

                for j in self.center_point:
                    plt.axvline(j, color='red', alpha=0.6)
                # plt.legend()
                # plt.show()

    def split_t_segments(self, show=False):
        self.cor = []
        self.hor_segments = []
        self.ver_segments = []
        for r, c in zip(self.avg_line, self.center_point):
            self.cor.append([np.int16(np.average(r)) + int(self.qrs_width / 2), c])
            # print(c - np.int16(np.average(r)))
        # print(self.cor)

        # Horizontal Split
        for lead in self.ecg_leads:
            sg = []
            for co in self.cor:
                sg.append(self.cleaned_signal[lead][co[0]:co[1]])
            self.hor_segments.append(sg)

        # Vertical split
        for co in self.cor:
            sg = []
            for lead in self.ecg_leads:
                sg.append(self.cleaned_signal[lead][co[0]:co[1]])
            self.ver_segments.append(sg)

        # np.save("horizontal.npy", self.hor_segments)
        if show:
            for i, svg in enumerate(self.hor_segments):
                plt.figure(figsize=(5, 5))
                plt.title(f"H-{i}")
                for sg in svg:
                    plt.plot(sg)
                plt.axhline()

            # np.save("vertical.npy", self.ver_segments)
            for i, svg in enumerate(self.ver_segments):
                plt.figure(figsize=(5, 5))
                plt.title(f"V-{i}")
                for sg in svg:
                    plt.plot(sg)
                plt.legend(self.ecg_leads)
                plt.axhline()
            plt.show()

    def split_qrs_segments(self, show=False):
        self.qrs_cor = []
        self.qrs_hor_segments = []
        self.qrs_ver_segments = []
        for r in self.avg_line:
            start = np.int16(np.average(r)) - int(self.qrs_width / 2)
            if start < 0: start = 0
            self.qrs_cor.append([start, np.int16(np.average(r)) + int(self.qrs_width / 2)])
        # print(self.cor)

        # Horizontal Split
        for lead in self.ecg_leads:
            sg = []
            for co in self.qrs_cor:
                if len(self.cleaned_signal[lead][co[0]:co[1]]) > 10:
                    sg.append(self.cleaned_signal[lead][co[0]:co[1]])
                # print(f"len of qrs H: {len(self.cleaned_signal[lead][co[0]:co[1]])}")
            self.qrs_hor_segments.append(sg)

        '''
        [170] QRS Cor : [[226, 326], [569, 669], [930, 1030], [1283, 1383], [1618, 1718], [1960, 2060], 
        [2337, 2437],
        [2686, 2786], [3072, 3172], [3455, 3555], [3817, 3917], [4178, 4278], [4553, 4653], [4902, 5002]]
        '''
        # Vertical split
        # print(f"[170] QRS Cor : {self.qrs_cor}")
        for co in self.qrs_cor:
            sg = []
            for lead in self.ecg_leads:
                if len(self.cleaned_signal[lead][co[0]:co[1]]) > 10:
                    sg.append(self.cleaned_signal[lead][co[0]:co[1]])
                # print(f"len of qrs V: {len(self.cleaned_signal[lead][co[0]:co[1]])}")

            self.qrs_ver_segments.append(sg)

        # print(f"length: {len(self.qrs_ver_segments)}")
        # print(self.qrs_ver_segments)
        np.save(r"qrs_horizontal.npy", self.qrs_hor_segments)
        np.save(r"qrs_vertical.npy", self.qrs_ver_segments)
        # print("Data saved!")
        if show:
            for i, svg in enumerate(self.qrs_hor_segments):
                plt.figure(figsize=(5, 5))
                plt.title(f"H-{i}")
                for sg in svg:
                    plt.plot(sg)
                plt.axhline()
            plt.show()

            for i, svg in enumerate(self.qrs_ver_segments):
                plt.figure(figsize=(5, 5))
                plt.title(f"V-{i}")
                for sg in svg:
                    plt.plot(sg)
                # plt.legend(self.ecg_leads)
                plt.axhline()
            plt.show()

    def get_j_point_from_2d(self, show=False):
        global closest_point
        closest_ind_val = []
        self.j_point_2d = []
        self.j_value_2d = []
        qrs_point = []
        # self.qrs_ver_segments = np.array(self.qrs_ver_segments)
        # print(f"Length of ecg data: {len(self.qrs_ver_segments)}")
        for lead_ind in range(len(self.qrs_ver_segments)):
            upper = []
            lower = []
            for ind, segm in enumerate(np.array(self.qrs_ver_segments[lead_ind]).T):
                upper.append(segm.max() * 10)
                lower.append(segm.min() * 10)
            # plt.plot(upper)
            # plt.plot(lower)
            # plt.show()
            closest_point, closest_value = func.find_closest_points(upper, lower, min_distance=self.qrs_width / 8,
                                                                    th=2.5)
            closest_ind_val.append((closest_point, closest_value))
            # print(f"SJ0 : {np.int16(np.average(self.avg_line[lead_ind]) + closest_point / 2)}")
            self.j_point_2d.append(np.int16(np.average(self.avg_line[lead_ind]) - (self.qrs_width / 2) + closest_point))
            # self.j_point_2d.append(closest_point)
            self.j_value_2d.append(closest_value)
            qrs_point.append(np.int16(np.average(self.avg_line[lead_ind])))
            if show:
                plt.figure()
                plt.plot(upper, '.-', linewidth=3, color="black")
                plt.plot(lower, '.-', linewidth=3, color="teal")
                # sub = np.subtract(upper, lower)
                # add = np.add(upper, lower)wq
                # avg = func.avg_finder(np.array(upper), np.array(lower))
                # plt.plot(sub, "-o", color="red")
                # plt.plot(avg, "-o", color="blue")
                # plt.plot(sub)
                # plt.show()
                plt.axhline(max(upper) / 7)
                plt.axvline(closest_point)
                plt.show()
                # break
        return self.j_point_2d, self.j_value_2d

    def find_t_peak(self, show=False):
        pass
