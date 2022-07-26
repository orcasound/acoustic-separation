import pywt
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

class Wavelet_transforms:
    def _init_(self, original):
        self.original = original # original signal resized

    def denoise(self):
        # w = pywt.Wavelet('sym6')
        # maxlev = pywt.dwt_max_level(len(self.original), w.dec_len)
        # # Get detail coefficients
        # coeffs = pywt.wavedec(self.original, 'sym6', level=maxlev)
        # for i in range(1, len(coeffs)):
        #     coeffs[i] = pywt.threshold(coeffs[i], 0.5 * max(coeffs[i]))
        # self.filtered_signal = pywt.waverec(coeffs, 'sym6')
        (cA5, cD5, cD4, cD3, cD2, cD1) = pywt.wavedec(self.original, 'sym6', level=5)
        # Apply soft thresholding
        threshold = 0.25  # Threshold for filtering
        cA5t = pywt.threshold(cA5, threshold * max(cA5), mode='garrote')
        cD5t = pywt.threshold(cD5, threshold * max(cD5), mode='garrote')
        cD4t = pywt.threshold(cD4, threshold * max(cD4), mode='garrote')
        cD3t = pywt.threshold(cD3, threshold * max(cD3), mode='garrote')
        cD2t = pywt.threshold(cD2, threshold * max(cD2), mode='garrote')
        cD1t = pywt.threshold(cD1, threshold * max(cD1), mode='garrote')
        # Multilevel reconstruction from coefficients
        XD = pywt.waverec([cA5t, cD5t, cD4t, cD3t, cD2t, cD1t], 'sym6')
        # Match original size with reconstruction size
        if self.original.shape[0] < XD.shape[0]:
            num = XD.shape[0] - self.original.shape[0]
            for i in range(num):
                self.original = np.append(self.original, 0)
        # Filtered signal is the subtraction between the original signal and the boat sound XD
        self.filtered_signal = np.fft.fft(self.original-XD)
        return self.filtered_signal, cA5, cD5, cD4, cD3, cD2, cD1, cA5t, cD5t, cD4t, cD3t, cD2t, cD1t, XD
