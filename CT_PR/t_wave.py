from utils.function import *


class TWave:
    def __init__(self, x, qrs_peak, margin=0, sampling_rate=400,
                 conv_len=25, base_line=0, per_sample=0.0025):
        self.st_area = None
        self.left_prog = None
        self.t_x = x
        self.t_margin = margin
        self.t_qrs_peak = qrs_peak
        self.t_sampling_rate = sampling_rate
        self.t_conv_len = conv_len
        self.t_result = []
        self.s_peak_index = []
        self.center_point = find_center_point(self.t_x, self.t_qrs_peak)[0]
        self.t_slope = np.gradient(self.t_x)
        self.t_y = np.sin(np.linspace(0, np.pi, self.t_conv_len))
        self.t_amp_slop = np.convolve(self.t_slope, self.t_y, mode='same')
        self.t_base_line = base_line
        self.t_bthm = 0.5
        self.t_peak = []
        self.j_point = []
        self.per_sample = per_sample
        self.ST_type = "None"

    def get_t_wave(self, show=True, baseline=0):
        """
        Return: T wave signal, from s-offset to center point
        """
        # print("R-T Wave: {0:.3f}".format(self.center_point))
        self.left_prog = []
        self.t_result = []
        self.s_peak_index = []

        # ========== DATE: 09-10-2022 ===========

        self.j_point = j_point_finder(self.t_x, win_legth=4, qrs_peak=self.t_qrs_peak)
        # print("====", self.j_point)
        # print("====", self.t_qrs_peak)

        if self.t_qrs_peak[0] > self.j_point[0]:
            self.j_point.pop(0)

        # print("New R peak:\t", len(self.t_qrs_peak))
        # print("New J-Point:\t", len(self.j_point))
        # print("New C-Point:\t", len(self.center_point))

        # print("New R peak:\t", self.t_qrs_peak)
        # print("New J-Point:\t", self.j_point)
        # print("New C-Point:\t", self.center_point)
        self.polarity = []
        for jp, cp in zip(self.j_point, self.center_point):
            # print("J-point: {}, C-point: {} ".format(jp, cp))
            self.t_result.append([self.t_x[jp:cp], jp])
            derivative = integration(np.diff(self.t_x[jp:cp]))  # np.convolve(np.diff(self.t_x[jp:cp]), np.ones(5))
            self.polarity.append(area_under_curve(derivative))
            # print("St Segments: ", list(self.t_x[jp:cp]))
            # Ratan Sir
            # nd2 = abs(np.diff(self.t_x[jp:cp], 2))
            # ind = np.argmax(abs(np.diff(self.t_x[jp:cp], 2)))
            # ==========

        # print("Polarity: ", self.polarity)
        if self.polarity.count(1) > self.polarity.count(0):
            self.ST_type = "positive"
            print("Positive: ", self.polarity)
        elif self.polarity.count(1) < self.polarity.count(0):
            self.ST_type = "negative"
            print("Negative: ", self.polarity)
        else:
            self.ST_type = "None"
            print("Not Found: ", self.polarity)

        self.s_peak_index = self.j_point

        if show:
            plt.title("T wave Segments")
            for segment in [i[0] for i in self.t_result]:
                if len(segment) > 0:
                    # plt.plot(segment, color='red')
                    # y = np.cos(np.linspace(0, np.pi, 20))
                    y = np.ones(4)  # For moving Average
                    d = np.diff(segment)
                    # plt.plot(integration(d), color='green')
                    # print("T wave segment len: ", len(segment))
                    plt.plot(lowpass(segment, CUT_OFF_FREQUENCY=10), color='green')
            plt.axhline(0)
            plt.legend(["Original", "Integrated"])
            plt.show()

            plt.title("T wave average polling")
            for segment in [i[0] for i in self.t_result]:
                y = np.ones(10)  # For moving Average, we use linilear response for convolution. [10 -> window size]
                d = integration(np.diff(average_polling(segment, n=2, order=2)))
                # print("Nature: ", sum(d),"\t", area_under_curve(d))
                plt.plot(d, color='green')
            plt.axhline(0)
            plt.show()

        # print("Result: ", self.t_result)
        # for i in self.t_result:
        #     print(len(i))
        return self.t_result

    # TODO: correct t peak index
    def correct_t_peak_index(self):
        for ind in self.tp:
            # print("correct_t_peak_index: ", ind)
            pass

    def find_t_peak(self, show=False, baseline=0, polarity=1):
        """
        Return [[peak, amp], [peak, amp], ...]
        """
        self.tp = []
        avg = []
        self.s_wave = self.get_t_wave()
        for i in self.s_wave:
            if len(i[0]) > 0:
                # temp_peak = signal.find_peaks(i[0], prominence=0.001, distance=4)[0] + +i[1]
                # finding the maximum peak value for t peak
                tem = i[0] - baseline

                # classification by area under the curve
                # temp = integration(np.diff(tem))
                if polarity == 1:
                    temp_peak = np.argmax(i[0]) + i[1]

                else:
                    temp_peak = np.argmin(i[0]) + i[1]

                # neg = np.argmin(i[0]) + i[1]
                # print("ABCD", i[0][np.argmax(i[0])], i[0][np.argmin(i[0])])

                # if abs(i[0][np.argmin(tem)]) > abs(i[0][np.argmax(tem)]):
                #     print("True", baseline)
                # else:
                #     print("False", baseline)
                #     temp_peak = np.argmin(i[0]) + i[1]

                # print("S-Peak: ", temp_peak)
                self.tp.append([temp_peak, self.t_x[temp_peak]])
                avg.append(self.t_x[temp_peak])

        print("[::] T-amp:\t --> {0:.3f} mv".format(np.average(avg)))
        if np.average(avg) > 0.05:
            print("[::] Hypercute T wave: 1. Vasospasm, 2. Early Stemi")

        if show:
            plt.plot([i[1] for i in self.tp])
            plt.show()
        self.correct_t_peak_index()
        return self.tp, np.average(avg)

    def t_on_off(self, n_points=20):
        temp = []
        t_peaks = [i[0] for i in self.tp]
        for start, i in zip(t_peaks, self.s_wave):
            # print("ABC", start)
            ind, val = find_tangent_line(i[0],
                                         prev_n=n_points,
                                         t_peak=start, baseline_stop=True)
            # print("Offset: ", len(ind), ind[-1])
            temp.append([ind[-1]][0])
        # print("T Tangent OffSet: ", temp)

    def find_j_point(self):
        """
        Return: J point and amplitude [[peak, amp], [peak, amp], ...]
        """
        self.j_point = []
        for i in j_point_finder(self.t_x, win_legth=4, qrs_peak=self.t_qrs_peak, show=True):
            if i:
                self.j_point.append([i, self.t_x[i]])

        # print("J-Point:\t --> {}".format(self.j_point))
        # print("QRS Peak: {}".format(self.t_qrs_peak))
        if self.t_qrs_peak[0] > self.j_point[0][0]:
            self.j_point.pop(0)
        return self.j_point

    # TODO: Find T duration
    def t_duration(self, t_index, lbsr=True, rbsr=True):
        """
        params: peaks, lbsr-> left baseline, rbsr-> right baseline
        Return : [t_duration, onset, offset]
        """
        if lbsr and rbsr:
            onset = t_index - left_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        elif lbsr and not rbsr:
            onset = t_index - left_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, slop=0.001)
            t = t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        elif rbsr and not lbsr:
            onset = t_index - left_deep(self.t_x, t_index, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, lth=self.t_base_line, slop=0.001)
            t = t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

        else:
            onset = t_index - left_deep(self.t_x, t_index, slop=0.001)
            offset = t_index + right_deep(self.t_x, t_index, slop=0.001)
            t = (offset[0] - onset[0]) * self.per_sample
            return t, onset[0], offset[0]

    # TODO: Find T onset offset
    def t_onset_offset(self):
        """
        Return: P onset and offset
        """

        # Search for P-wave onset and offset index
        self.t_onset = []
        self.t_offset = []

        # Store P onset and offset Amplitute
        self.t_onset_val = []
        self.t_offset_val = []

        self.time_dur = []
        # print("[::] T-wave Amplitude:", [i[1] for i in self.tp])
        # plt.xlabel("Amplitude (mV)")
        # plt.ylabel("Contribution")
        # plt.hist([i[1]*10 for i in self.pp], bins=10)
        # plt.savefig("result/p_wave_amp.png")
        # temp = []
        t_peaks = [i[0] for i in self.tp]
        for start, i in zip(t_peaks, self.s_wave):
            stp = np.argmax(i[0]) + 1
            # print("ABC", start, "LEN:", len(i[0][stp:]))
            ind, val = find_tangent_line(i[0][stp:],
                                         prev_n=20,
                                         t_peak=start, baseline_stop=True)
            oti, otv = find_tangent_p_onset(i[0][:stp],
                                            prev_n=20,
                                            t_peak=start, baseline_stop=True)
            self.t_onset.append(oti[-1])
            self.t_onset_val.append(self.t_x[oti[-1]])
            # print("OTI: ", oti[-1])
            # print("Offset: ", ind[-1], val[-1])
            self.t_offset.append(ind[-1])
            # temp.append([ind[-1]][0])
            self.t_offset_val.append(self.t_x[ind[-1]])

            plt.title("ON OFF")
            plt.plot(i[0][:stp])
        plt.show()
        # print("T Tangent OffSet: ", temp)

        for i in self.tp:
            # onset = i[0] - left_deep(self.t_x, i[0], lth=self.t_base_line)
            # offset = i[0] + right_deep(self.t_x, i[0], lth=self.t_base_line)
            ti = self.t_duration(i[0], lbsr=True, rbsr=True)
            # print("---------->", ti)
            if ti[0] < 0.1:
                print("First Trial")
                # self.t_onset.append(ti[1])
                # self.t_offset.append(ti[2])
                # self.t_onset_val.append(self.t_x[ti[1]])
                # self.t_offset_val.append(self.t_x[ti[2]])
                self.time_dur.append(ti[0])
            else:
                ti = self.t_duration(i[0], lbsr=True, rbsr=True)
                if ti[0] < 0.15:
                    print("Second Trial")
                    # self.t_onset.append(ti[1])
                    # self.t_offset.append(ti[2])
                    # self.t_onset_val.append(self.t_x[ti[1]])
                    # self.t_offset_val.append(self.t_x[ti[2]])
                    self.time_dur.append(ti[0])
                else:
                    print("Third Trial")
                    ti = self.t_duration(i[0], lbsr=True, rbsr=False)
                    # self.t_onset.append(ti[1])
                    # self.t_offset.append(ti[2])
                    # self.t_onset_val.append(self.t_x[ti[1]])
                    # self.t_offset_val.append(self.t_x[ti[2]])
                    self.time_dur.append(ti[0])

        # print("[::] T-onset:\t --> {0:.3f}".format(self.t_onset))
        # print("[::] T-offset:\t --> {0:.3f}".format(self.t_offset))

        gap = [i[1] - i[0] for i in zip(self.t_onset, self.t_offset)]
        # print("[::] T-onset to P-offset:\t --> {0:.3f}".format(gap))
        print("[::] T-duration(ms):\t --> {0:.3f}".format(np.average(self.time_dur) * 1000))
        print("[::] T-mean duration(ms):\t --> {0:.3f}".format(np.mean(self.time_dur) * 1000))
        print("[::] T-std duration(ms):\t --> {0:.3f}".format(np.std(self.time_dur) * 1000))
        print("[::] T-max duration(ms):\t --> {0:.3f}".format(np.max(self.time_dur) * 1000))
        print("[::] T-min duration(ms):\t --> {0:.3f}".format(np.min(self.time_dur) * 1000))

        print("[::] T-onset Amplitute:\t --> {0:.3f} mv".format(np.average(self.t_onset_val)))
        print("[::] T-offset Amplitute:\t --> {0:.3f} mv".format(np.average(self.t_offset_val)))
        print("[::] T-Amplitute:\t --> {0:.3f} mv".format(np.average(self.t_onset_val) - np.average(self.t_offset_val)))
        print("[::] T-mean:\t --> {0:.3f} mv".format(np.mean(self.t_x)))
        print("[::] T-std:\t --> {0:.3f} mv".format(np.std(self.t_x)))
        print("[::] T-max:\t --> {0:.3f} mv".format(np.max(self.t_x)))
        print("[::] T-min:\t --> {0:.3f} mv".format(np.min(self.t_x)))

        return self.t_onset, self.t_offset, self.t_onset_val, self.t_offset_val, self.time_dur

    def get_t_onset(self):
        return self.t_onset

    def get_t_offset(self):
        return self.t_offset

    # ST Wave Classification
    def st_wave_classification(self, show_out=True):
        self.find_j_point()
        self.st_area = []
        # print("[::] J-Point:\t --> {}".format(self.j_point))
        self.j_value = [i[1] for i in self.j_point]
        print("[::] ", len(self.t_result), "value")
        for dt, j in zip(self.t_result, self.j_value):
            # print("######")
            y_up = np.array([i for i in dt[0] if i >= j])
            y_down = np.array([i for i in dt[0] if i < j])
            self.st_area.append([sum(abs(y_down - j)), sum(abs(y_up - j))])
        return self.st_area



'''
rename the all pdf file with the PID name, you will get PID from the same PDF file.
File Should be: {PID}_file_name.pdf and upload all the pdf on the google drive that shared with you.
Drive Link: https://drive.google.com/drive/folders/1UCkf0jIpQ-CZ9T8JMP-7dwrwzQ6SANJu?usp=sharing

'''