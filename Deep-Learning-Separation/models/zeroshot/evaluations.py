import numpy as np
import pandas as pd
import os
import math
import librosa
from transforms import *
from sklearn.metrics import mean_squared_error



def signalPower(x):
    return np.mean(x**2)


def SNRsystem(inputSig, outputSig):
    noise = outputSig - inputSig
    powS = signalPower(outputSig)
    powN = signalPower(noise)
    snr = (powS-powN)/powN
    return snr

filename = "20181202_25sec.mp3"
#C:\Users\Ambra\Desktop\orcasound\MATLAB_filtered_audios\Binomial_weighted_average.wav
y, fs = librosa.load(filename)
#os.system(filename) #play sound
#Rescale input
m = len(y)
n = pow(2, math.ceil(math.log(m)/math.log(2)))
signal = np.fft.fft(y,n=n)
dfty = Fourier_transforms()
dfty.signal = signal
Fourier_transforms.hampel_filter(dfty)
filtered = np.real(np.fft.ifft(dfty.filtered_signal))
i = 1
Fourier_transforms.binomial_weighted_moving_average(dfty)
cleaner = np.real(np.fft.ifft(dfty.filtered_signal))
snr1 = SNRsystem(np.real(signal), filtered)
print("SNR with hampel: {} ".format(snr1)) #-0.9999981602044624
snr2 = SNRsystem(np.real(signal), cleaner)
print("SNR with binomial: {} ".format(snr2)) # -0.999999956536924 higher hence better

mse1= mean_squared_error(filtered, np.real(signal))
mse2= mean_squared_error(cleaner, np.real(signal))
print("MSE with hampel: {} ".format(mse1))
print("MSE with binomial: {} ".format(mse2))
# print("NMSE with hampel: {} ".format(mse1/np.mean(np.real(signal))))
# print("NMSE with binomial: {} ".format(mse2/np.mean(np.real(signal)))) #mean signal is 0.005375520326197149
print("RMSE with hampel: {} ".format(math.sqrt(mse1)))
print("RMSE with binomial: {} ".format(math.sqrt(mse2)))
# print("NRMSE with hampel: {} ".format(math.sqrt(mse1/np.mean(np.real(signal)))))
# print("NRMSE with binomial: {} ".format(math.sqrt(mse2/np.mean(np.real(signal)))))
#write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.float32)) #y is float32