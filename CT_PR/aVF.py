# importing modules
import numpy as np
import matplotlib.pyplot as plt
from utils import function as func


# Lead I calculation
class AVF(func.PWave):
    def __init__(self, cleaned_signal, sampling_rate=400, anterior_leads=None):
        """
        params: cleaned_signal, sampling_rate, anterior_leads
        """
        self.win_length = None
        self.stm = []
        self.ste = []
        self.ste_a = []
        self.ste_i = []
        self.stm_i = []
        self.stm_a = []
        self.prt = None
        self.rp_interval = None
        self.qt_class = None
        self.qtc_interval = None
        self.qt_interval = None
        self._qrs_duration = None
        self.t_duration = None
        self.ventricular_rate = None
        self.beat_count = None
        self.rr_type = None
        self.hr_type = None
        self.heart_rate = None
        self.rr_interval = None
        self.pr_sample = None
        self.pq_interval = []
        self.qt_int = None
        self.atrial_rate = None
        self.t_peak = None
        self.t_amp = None
        self.j_point = None
        self.t_offset = None
        self.t_onset = None
        self.p_amp = None
        self.q_onset = None
        self.r_peaks = None
        self.on_off_p = None
        self.p_peak = None
        self.p_wave = None
        self.p_onset = None
        self.pr_interval = None
        self.VAT = None
        if anterior_leads is None:
            anterior_leads = ["AVF"]
        self.ecg_result = {anterior_leads[0]: {}}
        self.sampling_rate = sampling_rate
        self.persamtim = 1 / sampling_rate
        self.cleaned_signal = cleaned_signal
        self.anterior_leads = anterior_leads[0]
        self.baseline = 0
        self.win_time = 40 # milliseconds

        # A Object For T wave Analysis
        self.t_obj = None

    # Setting R peaks
    def set_r_peaks(self, r_pears, baseline=0, show=True, adjust_sig=True):
        """
        Param:
            r_peak: Dictionary of Index and value
            if show == True,
                display graph
            else:
                only set value in the class
        Return: None
        """

        self.r_peaks = r_pears
        self.baseline = baseline

        # Addition setting for baseline wandering
        if adjust_sig:
            self.cleaned_signal = self.cleaned_signal - baseline
            self.r_peaks["value"] = self.r_peaks["value"] - baseline
            self.baseline = 0
            # initializing super class PWave
            super().__init__(self.cleaned_signal, self.r_peaks["index"], self.baseline)
            self.t_obj = func.TWave(self.cleaned_signal, self.r_peaks["index"],
                                    self.baseline, conv_len=35, per_sample=self.persamtim)
        else:
            # initializing super class PWave
            super().__init__(self.cleaned_signal, self.r_peaks["index"], self.baseline)
            # initializing super class TWave
            self.t_obj = func.TWave(self.cleaned_signal, self.r_peaks["index"],
                                    self.baseline, conv_len=35, per_sample=self.persamtim)

        # If Show then plot graph
        if show:
            plt.title(f"QRS Peak Scatter Graph, {self.anterior_leads}")
            plt.xlabel("Time")
            plt.ylabel("amplitude (mV)")
            plt.scatter(self.r_peaks["index"], self.r_peaks["value"], color='red')
            plt.plot(self.cleaned_signal, label="Clean signal")
            plt.axhline(self.baseline, color='red')
            plt.show()

    def rate_classification(self):
        __heart = func.AnalysePeaks(self.cleaned_signal, self.r_peaks["index"], sampling_rate=self.sampling_rate)
        __qrs = func.QRSComplex(self.cleaned_signal, self.r_peaks["index"], margin=0.0005, base_line=self.baseline)

        self.rr_interval = __heart.cal_rr_interval()
        self.heart_rate = __heart.cal_heart_rate()
        self.hr_type = __heart.rate_analysis(show_out=False)
        self.rr_type = __heart.rythm_analysis(show_out=False)
        self.beat_count = __heart.get_beat_count(show_out=False)
        self.ventricular_rate = self.heart_rate
        self._qrs_duration = __qrs.get_qrs_duration(show_out=False)[1]
        r_amp, q_amp, s_amp = __qrs.get_r_duration()
        qrs_type = __qrs.get_qrs_duration_type(show_out=False)
        r_dur = __qrs.get_r_dur()
        self._Q_duration = __qrs.Q_duration
        av_relationship = round(self.atrial_rate / self.ventricular_rate)
        self.p_onset[1: self.beat_count]
        self.rp_interval = round(np.average([p - r for p, r in zip(self.p_onset[1:], self.qrs_peak)]) * self.persamtim * 1000)
        self.VAT = round(self._Q_duration + np.max(r_dur).round(3))

        self.ecg_result[self.anterior_leads]["R_Amplitude"] = {"V": round(r_amp, 4), "U": "mV"}
        self.ecg_result[self.anterior_leads]["Q_Amplitude"] = {"V": round(q_amp, 4), "U": "mV"}
        self.ecg_result[self.anterior_leads]["S_Amplitude"] = {"V": round(s_amp, 4), "U": "mV"}
        self.ecg_result[self.anterior_leads]["QRS_Duration"] = {"V": self._qrs_duration, "U": "ms"}
        self.ecg_result[self.anterior_leads]["Q_Duration"] = {"V": __qrs.Q_duration, "U": "ms"}
        self.ecg_result[self.anterior_leads]["R_Duration"] = {"V": __qrs.R_duration, "U": "ms"}
        self.ecg_result[self.anterior_leads]["S_Duration"] = {"V": __qrs.S_duration, "U": "ms"}
        self.ecg_result[self.anterior_leads]["QRS_Type"] = {"V": qrs_type, "U": ""}
        self.ecg_result[self.anterior_leads]["R_peak_time"] = {"V": np.max(r_dur).round(3), "U": "ms"}
        self.ecg_result[self.anterior_leads]["AV_relationship"] = {"V": av_relationship}
        self.ecg_result[self.anterior_leads]["RP_Interval"] = {"V": self.rp_interval, "U": "ms"}
        self.ecg_result[self.anterior_leads]["RR_Interval"] = {"V": round(self.rr_interval), "U": "ms"}
        self.ecg_result[self.anterior_leads]["Heart_Rate_Type"] = {"V": self.hr_type, "U": ""}
        self.ecg_result[self.anterior_leads]["Ventricular_rate"] = {"V": self.ventricular_rate, "U": "bpm"}
        self.ecg_result[self.anterior_leads]["Heart_Rate"] = {"V": self.heart_rate, "U": "Bpm"}
        self.ecg_result[self.anterior_leads]["RR_Rhythm"] = {"V": self.rr_type, "U": ""}
        self.ecg_result[self.anterior_leads]["Total_Beat"] = {"V": self.beat_count, "U": ""}
        self.ecg_result[self.anterior_leads]["Total_QRS_amp"] = {"V": round((r_amp + s_amp), 3), "U": ""}
        self.ecg_result[self.anterior_leads]["Net_QRS_amp"] = {"V": round((r_amp - s_amp), 3), "U": ""}
        self.ecg_result[self.anterior_leads]["VAT"] ={"V": self.VAT ,"U":"ms"}

        # TODO: Need to implement more if there is any bug found in the interval
        __qt = []
        print("===========N")
        print(__qrs.get_q_onset())
        print(self.t_obj.get_t_offset())
        for q, t in zip(__qrs.get_q_onset(), self.t_obj.get_t_offset()):
            if t > q:
                __qt.append((t - q) * self.per_sample * 1000)
        self.qt_interval = round(np.mean(__qt))
        self.qtc_interval = self.qt_interval + ((1000 - __heart.get_rr_interval()) / 7)

    # P wave analysis Finding P peak and scatter optional
    def calculate_p_peaks(self, show=False, r=False):
        # print(self.r_peaks)
        if len(self.r_peaks["index"]) <= 1: print("Artifact Present"); return "Artifact Present"
        self.p_peak, self.p_amp, amps = self.find_p_peaks(polarity=1)
        self.on_off_p = self.p_onset_offset()

        if show:
            plt.title("P Wave")
            plt.xlabel("Time")
            plt.ylim(-3, 3.5)
            plt.ylabel("amplitude (mV)")
            plt.plot(self.cleaned_signal, label="Clean signal")
            for item in self.p_peak:
                plt.text(item[0], self.cleaned_signal[item[0]], "P", color="blue", fontsize=12)
                plt.scatter(item[0], self.cleaned_signal[item[0]], color="black")

            # Scatter R if True
            if r: plt.scatter(self.r_peaks["index"], self.r_peaks["value"], color='red')
            # Scatter p onset and p offset
            plt.scatter(self.on_off_p[0], self.on_off_p[2], color='magenta', s=50)
            plt.scatter(self.on_off_p[1], self.on_off_p[3], color='blue', s=50)

            plt.axhline()
            plt.show()
        self.p_wave_analysis()

    def get_atrial_rate(self):
        """

        :return: Atrial Rate
        """
        print(f"P peak: {self.p_peak}")
        self.atrial_rate = round(60 / (np.average(np.diff([i[0] for i in self.p_peak])) * self.persamtim))
        pp_interval = round(np.average(np.diff([i[0] for i in self.p_peak])) * self.persamtim)
        print(f"Atrial Rate: {self.atrial_rate} bpm")

        # Storing to global Result
        self.ecg_result[self.anterior_leads]["Atrial_rate"] = {"V": self.atrial_rate, "U": "bpm"}
        self.ecg_result[self.anterior_leads]["pp_interval"] = {"V": pp_interval, "U": "mV"}
        return self.atrial_rate

    def get_pr_interval(self, q_onset, per_sample=0.0025):
        """
        :param q_onset:
        :param per_sample:
        :return: pr duration
        """

        _P_duration = func.PWave(self.cleaned_signal, self.r_peaks["index"], margin=0, s_rate=400,conv_len=25, base_line=self.baseline, per_sample=0.0025)      
            
        # TODO: maybe need to improve the process of detection if required.
        self.q_onset = [i[0] for i in q_onset if len(i) > 0]
        self.ecg_result[self.anterior_leads]["P_Amplitude"] = {"V": round(np.average(self.avg), 4), "U": ""}
        self.P_duration= round(np.average(self.time_dur) * 1000,4)
        self.P_Amplitude = round(np.average(self.avg), 4)
        self.P_area =  self.P_duration* self.P_Amplitude
        self.ecg_result[self.anterior_leads]["P_area"] = {"V": self.P_area  , "U": "mV-ms"}
        # self.ecg_result[self.anterior_leads]["P_duration"] = {"V": self.P_duration  , "U": "ms"}
        if np.average(self.avg) > 0.01:
            print("Bc")
            self.p_onset = [i for i in self.on_off_p[0]]
            for ind, val in enumerate(self.p_onset):
                if self.q_onset[ind] < val:
                    self.p_onset.pop(0)
                else:
                    break

            # Calculating PR interval
            self.pr_sample = []
            print("Q onset: ", self.q_onset)
            print("P onset: ", self.p_onset)
            for q, p in zip(self.q_onset, self.p_onset):
                if q > p: self.pr_sample.append(q - p)
            print("A Q onset: ", self.q_onset)
            print("A P onset: ", self.p_onset)

            # Q
            # onset: [84, 374, 669, 973, 1270, 1573, 1862, 2165, 2470, 2774, 3078, 3373, 3670, 3974, 4278, 4574]
            # P
            # onset: [2408, 2716, 3015, 3309, 3514, 4221, 4511]

            if len(self.pr_sample) > 2:
                # Duration in milliseconds
                pr_int = np.average(self.pr_sample) * per_sample * 1000
                print(f"Average PR interval: {pr_int:.4} ms")
                self.ecg_result[self.anterior_leads]["PR_Interval"] = {"V": round(pr_int), "U": "ms"}
                return pr_int
            else:
                self.ecg_result[self.anterior_leads]["PR_Interval"] = {"V": 0, "U": "ms"}
                return 0
        else:
            print("Cb")
            self.ecg_result[self.anterior_leads]["PR_Interval"] = {"V": 0, "U": "ms"}
            print(f"There is very low amplitude of p wave, ie: {np.average(self.avg):.4} mv")
            return 0

    # T wave analysis
    def get_basic_t(self, show=True, j_point=None):
        self.win_length = int((self.win_time / 1000) / self.persamtim)
        ttw = self.t_obj.get_t_wave(j_point=j_point)
        self.t_peak, self.t_amp = self.t_obj.find_t_peak()
        self.ecg_result[self.anterior_leads]["T_Amplitude"] = {"V": round(self.t_amp, 4), "U": "mv"}
        self.t_obj.t_on_off()
        print(f"T peaks index: {self.t_peak}, {len(ttw)}")

        on_off_t = self.t_obj.t_onset_offset(show=True)
        self.t_offset = on_off_t[1]
        self.t_onset = on_off_t[0]

        # J point is also known sas STJ point
        if j_point is None:
            self.j_point = self.t_obj.find_j_point()
        else:
            self.j_point = [[i, self.cleaned_signal[i]] for i in j_point]
            print("j_point Not None")
        self.t_obj.st_wave_classification()

        # T time Duration of ECG signal
        self.t_duration = np.mean(self.t_obj.get_time_dur())
        # print({"V": f"{t_amp:.4}", "U": "mV"})

        # verify j point and t onset point
        # Check if s onset is before j point then assign j point to the s onset.
        # Also define STM and STE point in between j point and S onset for st segment analysis
        self.stm_i = []
        self.stm_a = []
        self.ste_i = []
        self.ste_a = []
        self.j40 = []
        self.j80 = []
        for ind, val in enumerate(self.j_point):
            if ind < len(self.t_onset) and self.t_onset[ind] < val[0]:
                self.t_onset[ind] = val[0]
            if ind < len(self.t_onset):
                dif_step = int((self.t_onset[ind] - val[0]) / 3)
                # append index and amplitude
                self.stm_i.append(val[0] + dif_step)
                self.stm_a.append(self.cleaned_signal[self.stm_i[ind]])
                # append index and amplitude
                self.ste_i.append(val[0] + dif_step + dif_step)
                self.ste_a.append(self.cleaned_signal[self.ste_i[ind]])

            # For J40 and J80
            if (val[0] + self.win_length) < len(self.cleaned_signal):
                self.j40.append(self.cleaned_signal[val[0] + self.win_length])
            if (val[0] + self.win_length + self.win_length) < len(self.cleaned_signal):
                self.j80.append(self.cleaned_signal[val[0] + self.win_length + self.win_length])

            # J40 and J80 Amplitude
        self.ecg_result[self.anterior_leads]["J40"] = {"V": np.average(self.j40), "U": "mv"}
        self.ecg_result[self.anterior_leads]["J80"] = {"V": np.average(self.j80), "U": "mv"}

        # Calculate STJ amplitude, STM amplitude and STE amplitude
        self.ecg_result[self.anterior_leads]["STJ_Amp"] = {"V": np.average([i[1] for i in self.j_point]).round(3),
                                                           "U": "mv"}
        self.ecg_result[self.anterior_leads]["STM_Amp"] = {"V": np.average(self.stm_a).round(3), "U": "mv"}
        self.ecg_result[self.anterior_leads]["STE_Amp"] = {"V": np.average(self.ste_a).round(3), "U": "mv"}

        # TODO: Neet to calculate these parameter, now i am assuming ST_Min = avg(j,m,e)
        self.ecg_result[self.anterior_leads]["ST_Min"] = {"V": np.average([np.average([i[1] for i in self.j_point]),
                                                                           np.average(self.stm_a),
                                                                           np.average(self.ste_a)]).round(3), "U": "mv"}
        if round(self.t_amp, 4) > 0:
            self.ecg_result[self.anterior_leads]["ST_Pol"] = {"V": 1, "U": "mv"}
        else:
            self.ecg_result[self.anterior_leads]["ST_Pol"] = {"V": -1, "U": "mv"}

        if show:
            plt.title("T Components")
            plt.plot(self.cleaned_signal)
            for one in self.t_peak:
                plt.scatter(one[0], one[1], color='red', s=50)
            for one in self.j_point:
                plt.scatter(one[0], one[1], color='magenta')
            for i in self.t_offset:
                plt.scatter(i, self.cleaned_signal[i], color='black', s=100)
            for i in self.t_onset:
                plt.scatter(i, self.cleaned_signal[i], color='black')
            for i in self.stm_i:
                plt.scatter(i, self.cleaned_signal[i], color='brown')
            for i in self.ste_i:
                plt.scatter(i, self.cleaned_signal[i], color='brown')
            plt.axhline()
            plt.show()

    def get_qt_interval(self, per_sample=0.0025, rr_interval=1000):
        """

        :param rr_interval:
        :param per_sample:
        :return: Qt interval and Qtc interval
        """
        for ind, val in enumerate(self.q_onset):
            if self.t_offset[ind] < val:
                self.t_offset.pop(0)
            else:
                break

        qt = []
        for t, q in zip(self.t_offset, self.q_onset):
            # print("QT: ", t, q)
            if t > q: qt.append(t - q)

        print(f"QT Segments: {qt}, {len(qt)}")
        if len(qt) > 2:
            # Duration in milliseconds
            self.qt_int = np.average(qt) * per_sample * 1000
            self.qtc_interval = round(np.mean(qt) + ((1000 - rr_interval) / 7))
            self.ecg_result[self.anterior_leads]["QT_Interval"] = {"V": round(self.qt_int), "U": "ms"}
            self.ecg_result[self.anterior_leads]["QTc_Interval"] = {"V": self.qtc_interval, "U": "ms"}
            print(f"Average QT interval: {self.qt_int:.4} ms")
            print(f"Average QTc interval: {self.qtc_interval} ms")
            return self.qt_int
        else:
            return 0

    def get_qtc_interval(self):
        return self.qtc_interval

    def get_qt_class(self):
        __qtc = func.QtClassification(round(self.qt_int))
        _, qtr = __qtc.get_result()
        print("[::] QT Classification: ", qtr)
        self.qt_class = qtr
        self.ecg_result[self.anterior_leads]["QT_Class"] = {"V": self.qt_class, "U": ""}
        return self.qt_class

    # ======= Calculating PQ interval =====================
    def get_pr_type(self):
        if self.pr_sample is not None:
            a = [i * self.persamtim * 1000 for i in self.pr_sample]
            print(a)
            self.prt = self.pr_type(a)  # prt -> pr interval type
            self.ecg_result[self.anterior_leads]["Pr_type"] = {"V": self.prt, "U": ""}
            print(f"PR Type: {self.pr_type(a)}")
            return self.pr_type(a)
        else:
            print(f"PR type Not defined")
            self.ecg_result[self.anterior_leads]["Pr_type"] = {"V": "NA", "U": ""}
            return "NA"

    def get_report(self):
        return self.ecg_result


if __name__ == "__main__":
    print("it's not a main file. run: python main.py")
    pass
