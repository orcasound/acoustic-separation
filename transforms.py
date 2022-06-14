import numpy as np
import pandas as pd
from scipy.signal import lfilter, savgol_filter, medfilt
from hampel import hampel

class Fourier_transforms:
    def _init_(self, signal):
        self.signal = signal #in frequency mode

    def moving_average(self):
        step = 100
        coeff = np.ones((step), dtype=int)/ step
        #print(self.signal.shape, coeff.shape)
        self.filtered_signal = lfilter(coeff, 1, self.signal)
        fDelay = (len(coeff) - 1)/2/1000
        return self.filtered_signal

    def binomial_weighted_moving_average(self):
        h = np.array([0.5, 0.5])
        binomialCoeff = np.convolve(h, h)
        for n in range(0,3):
            binomialCoeff = np.convolve(binomialCoeff, h)
        bDelay = (len(binomialCoeff) - 1)/2;
        self.filtered_signal = lfilter(binomialCoeff, 1, self.signal)
        return self.filtered_signal

    def gaussian_expansion_moving_average(self):
        alpha = 0.45
        self.filtered_signal = lfilter([alpha], [1, alpha - 1], self.signal)
        return self.filtered_signal

    def cubic_sgfir_filter(self):
        self.filtered_signal = savgol_filter(abs(self.signal), 7,3)
        return self.filtered_signal

    def quartic_sgfir_filter(self):
        self.filtered_signal = savgol_filter(abs(self.signal), 7,4)
        return self.filtered_signal

    def quintic_sgfir_filter(self):
        self.filtered_signal = savgol_filter(abs(self.signal), 7,5)
        return self.filtered_signal

    def median_filter(self):
        # cannot specify  nth-order one-dimensional median filter to x
        self.filtered_signal = medfilt(abs(self.signal))
        return self.filtered_signal

    def hampel_filter(self):
        self.filtered_signal = hampel(pd.Series(abs(self.signal)), window_size=5, n=3, imputation=True)
        return self.filtered_signal.to_numpy()