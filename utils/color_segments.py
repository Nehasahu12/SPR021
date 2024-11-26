import numpy as np
import matplotlib.pyplot as plt


class ColorQrs:
    def __init__(self, cleaned_signal, qrs_peak, sampling_rate=400, anterior_leads=None,
                 rr_interval=800, qrs_interval=80, *args, **kwargs):
        """

        :param cleaned_signal:
        :param qrs_peak:
        :param sampling_rate:
        :param anterior_leads:
        :param rr_interval:
        """
        # TODO: Program to visualize QRS interval if there is short duration or low amplitude.
        self.cleaned_signal = cleaned_signal
        self.qrs_peak = qrs_peak
        self.sampling_rate = sampling_rate
        self.r_amp = []
        self.anterior_leads = anterior_leads
        self.prog_result = {}
        self.qrs_interval = qrs_interval
        self.win_time = qrs_interval
        self.persamtim = 1 / self.sampling_rate
        self.qrs_width = int((self.win_time / 1000) / self.persamtim)
        print(f"QRS Width window: {self.qrs_width}")

    def __scatter_qrs(self):
        for lead in self.anterior_leads:
            print(self.qrs_peak[lead])

    def __plot_data(self):
        for lead in self.anterior_leads:
            plt.plot(self.cleaned_signal[lead])
        plt.show()

    def __split_signal_and_highlight(self):
        boundary = np.int16(self.qrs_width / 2)
        fig, axs = plt.subplots(12, 1, figsize=(13.69, 8.27))
        fig.tight_layout(pad=3.0)
        for ind, lead in enumerate(self.anterior_leads):
            ax = axs[ind]
            ax.grid()
            plt.xlabel("(Wide QRS) Time ->")
            plt.ylabel("Amplitude (mV)")
            ax.plot(self.cleaned_signal[lead])
            for qrs_ind in self.qrs_peak[lead]["index"]:
                ax.axvspan(qrs_ind - boundary, qrs_ind + boundary, facecolor='red', alpha=0.5)
            ax.legend([lead])
        plt.show()

    def __split_signal_and_color(self):
        boundary = np.int16(self.qrs_width)
        # boundary = np.int16(self.qrs_width / 2)
        fig, axs = plt.subplots(12, 1, figsize=(13.69, 8.27))
        fig.tight_layout(pad=3.0)
        shift = 3
        for ind, lead in enumerate(self.anterior_leads):
            start = 0
            ax = axs[ind]
            ax.grid()
            plt.xlabel("(Wide QRS) Time ->")
            plt.ylabel("Amplitude (mV)")
            # self.cleaned_signal[lead] = self.cleaned_signal[lead] + (shift * ind)
            # print("QRs Width: ", boundary)
            x = np.linspace(0, len(self.cleaned_signal[lead]), len(self.cleaned_signal[lead]))
            for qrs_ind in self.qrs_peak[lead]["index"]:
                ax.plot(x[start:qrs_ind - boundary + 1], self.cleaned_signal[lead][start:qrs_ind - boundary + 1],
                        color='green', linewidth=1.5)
                ax.plot(x[qrs_ind - boundary:qrs_ind + boundary],
                        self.cleaned_signal[lead][qrs_ind - boundary:qrs_ind + boundary],
                        color='red', linewidth=1.5)
                start = qrs_ind + boundary - 1
            ax.legend([lead])
        plt.show()

    def __single_lead_plot(self):
        boundary = np.int16(self.qrs_width)
        for ind, lead in enumerate(self.anterior_leads):
            plt.figure(figsize=(7, 3))
            plt.cla()
            plt.grid()
            plt.xlabel("(Wide QRS) Time ->")
            plt.ylabel("Amplitude (mV)")
            self.cleaned_signal[lead] = self.cleaned_signal[lead][:1300]
            start = 0
            x = np.linspace(0, len(self.cleaned_signal[lead]), len(self.cleaned_signal[lead]))
            for qrs_ind in self.qrs_peak[lead]["index"]:
                if (qrs_ind + boundary) < len(self.cleaned_signal[lead]):
                    plt.plot(x[start:qrs_ind - boundary + 1], self.cleaned_signal[lead][start:qrs_ind - boundary + 1],
                             color='green', linewidth=1.5)
                    plt.plot(x[qrs_ind - boundary:qrs_ind + boundary],
                             self.cleaned_signal[lead][qrs_ind - boundary:qrs_ind + boundary],
                             color='red', linewidth=1.5)
                    # plt.text(qrs_ind - boundary, 0, "Q")
                    # plt.text(qrs_ind, 0, "R")
                    # plt.text(qrs_ind + boundary, 0, "S")
                    start = qrs_ind + boundary - 1
                else:
                    break
            plt.legend([lead])
            plt.savefig(f"qrs_color_image/{lead}.png")
            # plt.show()

    def run(self):
        # self.__plot_data()
        # self.__scatter_qrs()
        # self.__split_signal_and_highlight()
        self.__split_signal_and_color()
        self.__single_lead_plot()
