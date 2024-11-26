"""
Subject: By using all leads together or individual to find out any PVC is there or not,

Date: 11-03-2023
Author: Nitish sharma (10042)

"""

from utils import function as func
import numpy as np
import matplotlib.pyplot as plt

print("Hello World! Need to implement PVC Detection Algorithm")


def matching_temp():
    ht = np.array([2, 5, 1])
    xt = np.array([1, 2, 3, 4, 5, 6, 2, 3, 1, 2, 4, 1, 2, 1, 1, 2, 5, 1, 2, 1, 2, 1, 1, 2, 1, 1, 3])
    yt = []
    step = len(ht)
    for i in range(len(xt) - len(ht)):
        yt.append(sum(xt[i:i+step] - ht))
    plt.plot(ht, '.-', label="HT")
    plt.plot(xt, '.-', label="XT")
    plt.plot(yt, '.-', label="YT")
    plt.legend()
    plt.show()


class PVC:
    def __init__(self, cleaned_signal, qrs_peaks, baseline, debug=True):
        self.cleaned_signal = cleaned_signal
        self.qrs_peaks = qrs_peaks
        self.baseline = baseline
        self.debug = debug
        pass

    def _print(self, *args, **kwargs):
        if self.debug:
            print(*args, **kwargs)

    def test_1(self):
        self._print("hello World")
        matching_temp()


a = PVC(1, 2, 2, debug=True)
a.test_1()
