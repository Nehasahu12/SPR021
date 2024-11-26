"""
Subject: By using all leads of the median complex to define the end of
        ventricular repolarization (QT Interval),

Date: 11-03-2023
Author: Nitish sharma (10042)

Note: Need to implement for finding T offset correctly to find correct QT interval

"""

from utils import function as func
import numpy as np
import matplotlib.pyplot as plt

print("Hello World! Need to implement QT Detection with all lead")


class QtEndpoint:
    def __init__(self, cleaned_signal, qrs_peak, sampling_rate=400, baseline=0):
        self.cleaned_signal = cleaned_signal
        self.qrs_peak = qrs_peak
        self.sampling_rate = sampling_rate
        self.baseline = baseline

    def _split_signal(self):
        print("Hello World", self.qrs_peak)
