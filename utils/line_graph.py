from utils import function as func
import matplotlib.pyplot as plt
import numpy as np


class NormalVisualizer:
    def __init__(self, df, PID="123") -> None:
        self.df = df
        self.pid = PID
        self.len_of_qrs = []
        self.anterior_leads = ["Lead_I", "Lead_II", "Lead_III", "aVL", "aVR", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
        # self.anterior_leads = ["Lead_II"]
        self.segments = {}
        self.maxsig = {}
        self.x = []
        self.cleaned_signal = {}
        self.peaks = {}
        self.strip = []
        self.qrs_threshold = 6.0
        self.qrs_distance = 100
        self.GAIN = 1
        self.clean_all_data()

    def clean_all_data(self):
        for leads in self.anterior_leads:
            clean_signal = func.highpass(func.denoise_signal(
                func.lowpass(func.highpass((self.df[leads]), CUT_OFF_FREQUENCY=0.56), CUT_OFF_FREQUENCY=100), 'bior4.4',
                9, 1, 7) * self.GAIN)
            self.cleaned_signal[leads] = clean_signal
        #     plt.plot(clean_signal)
        # plt.show()

    def split_all(self):
        self.df["add_all"] = self.df[self.anterior_leads[0]]
        for leads in self.anterior_leads:
            maximized_signal = func.maximize_qrs_signal(func.lowpass(func.highpass(self.df[leads])), length=30, sigma=2,
                                                        mode='same', max_height=False)
            # Finding the QRS peaks threshold and QRS peaks based on polynomial fit
            threshold = func.find_threshold(maximized_signal, div_factor=self.qrs_threshold)

            # Find peaks of QRS
            qrs_peaks = signal.find_peaks(maximized_signal, height=threshold, distance=self.qrs_distance)
            peak_apmlitude = np.array(self.cleaned_signal[leads][qrs_peaks[0]])

            # Correct peaks of QRS
            corrected_peaks = func.correct_qrs_peaks(self.cleaned_signal[leads], qrs_peaks[0], peak_apmlitude)
            # Split the cleaned ecg into multiple segments with respect to the peaks
            denoised_segment = func.split_ecg_signal_p(self.cleaned_signal[leads], corrected_peaks[0], left_m=0,
                                                       right_m=40)
            self.segments[leads] = denoised_segment
            # print(self.segments[leads])

    def plot_leads(self):
        init = 0
        # plt.cla()
        plt.figure(2)
        plt.rcParams["figure.subplot.left"] = 0.07
        plt.rcParams["figure.subplot.right"] = 0.95
        # plt.rcParams["figure.subplot.bottom"] = 0.1
        # plt.rcParams["figure.subplot.top"] = 0.97
        plt.rcParams["figure.subplot.wspace"] = 0
        plt.rcParams["figure.subplot.hspace"] = 0.01
        plt.ylim(-3, 3)

        # plt.rcParams['axes.spines.left'] = False
        # plt.rcParams['axes.spines.right'] = False
        # plt.rcParams['axes.spines.top'] = False
        # plt.rcParams['axes.spines.bottom'] = False
        # plt.tick_params(labelbottom=False, labelleft=False)
        plt.grid(True, which='both', axis='both', linestyle='dashed')

        for leads in self.anterior_leads:
            for denoised_segment in self.segments[leads]:
                # print(denoised_segment)
                self.strip.extend(denoised_segment)
                x = np.linspace(init, init + len(denoised_segment), len(denoised_segment))
                self.x.extend(x)
                plt.text(init + int(len(x) / 4), 1.0, '%s' % leads)
                plt.axvline(init + len(x), ymin=0.3, ymax=0.6)
                plt.plot(x, denoised_segment, linewidth=2)
                init += len(denoised_segment)
                break
        # plt.plot(self.x, self.strip, linewidth=1)
        # plt.grid(True)
        plt.title("12 ECG Leads: {}".format(self.pid))
        # plt.savefig("ECG_GRAPH/SINGLE/{}.png".format(self.pid))
        plt.show()