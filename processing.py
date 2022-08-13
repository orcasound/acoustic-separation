import numpy as np
import math
import os
import librosa
import PIL.Image
import scipy
import pywt
from PIL import ImageTk
from scipy.io.wavfile import write
from transforms import *
from plots import *

from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.filedialog import asksaveasfilename

from collections import defaultdict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import subprocess
from spleeterfunc.spleeter_separator import Separator
from zeroshot.inference import inference



SPLEETER_MODEL_PARAMS = 'spleeterfunc/2stems-finetune.json'
SPLEETER_MODEL = '2stems-finetune'
ZEROSHOT_CHECKPOINT = 'checkpoints/zeroshot_asp_full.ckpt'
HTSAT_CHECKPOINT = 'checkpoints/htsat_audioset_2048d.ckpt'


#Open file
def ask_input():
    global y, fs, signals_dict, fsignals_dict, originals_dict, fs_dict, fft_dict, preprocessed_dict
    root.filename = filedialog.askopenfilenames(initialdir="/orcasound",
                                               title="Select audios file",
                                               filetypes=(("Wav files", ".wav"), ("mp3 files", ".mp3")))
    #print(type(root.filename), root.filename) #root.filename is a tuple
    signals_dict = defaultdict(lambda: "Not present") #contains filename, not actual signals
    originals_dict = defaultdict(lambda: "Not present") #contains original rezised in time
    fft_dict = defaultdict(lambda: "Not present") #contains original resized signals in fourier
    fsignals_dict = defaultdict(lambda: "Not present")#contains filtered signals in fourier
    fs_dict = defaultdict(lambda: "Not present")#contains original signals sampling frequency
    preprocessed_dict = defaultdict(lambda: "Not present")
    for index, file in enumerate(root.filename):#save tuple of filenames in dict
        signals_dict[index]= file
    for i in range(len(signals_dict)): #filter each file of dict
        y, fs = librosa.load(signals_dict[i], sr=44100)
        # Rescale input
        m = len(y)
        n = pow(2, math.ceil(math.log(m) / math.log(2)))
        signal = np.fft.fft(y, n=n)
        dfty.signal= signal
        fft_dict[i] = signal
        originals.original = y
        originals_dict[i] = y  # no need of resampling in time
        preprocessed_dict[i] = y
        fs_dict[i] = fs


def store_files(): #
    global cA5, cD5, cD4, cD3, cD2, cD1, cA5t, cD5t, cD4t, cD3t, cD2t, cD1t, XD
    for i in range(len(signals_dict)): #filter each file of dict
        if filter == "Moving average":
            Fourier_transforms.moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal #filtered signals saved in fsignals_dict
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Binomial weighted moving average":
            Fourier_transforms.binomial_weighted_moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Gaussian expansion moving average":
            Fourier_transforms.gaussian_expansion_moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Cubic-Weighted Savitzky-Golay":
            Fourier_transforms.cubic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Quartic-Weighted Savitzky-Golay":
            Fourier_transforms.quartic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Quintic-Weighted Savitzky-Golay":
            Fourier_transforms.quintic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Median filter":
            Fourier_transforms.median_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Hampel filter":
            Fourier_transforms.hampel_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered
        if filter == "Wavelet":
            (originals.filtered_signal, cA5, cD5, cD4, cD3, cD2, cD1, cA5t, cD5t, cD4t, cD3t, cD2t, cD1t, XD) = Wavelet_transforms.denoise(originals)
            fsignals_dict[i] = originals.filtered_signal
            filtered = np.real(np.fft.ifft(fsignals_dict[i]))
            preprocessed_dict[i] = filtered

#metrics
def signalPower(x):
    return np.mean(x**2)

def SNRsystem(inputSig, outputSig):
    noise = outputSig - inputSig
    powS = signalPower(outputSig)
    powN = signalPower(noise)
    snr = (powS-powN)/powN
    return snr

def metrics(name):
    global metric
    metric = name
    choose_metric()

def choose_metric():
    global m_value #evaluation in time space
    for i in range(len(originals_dict)): #filter each file of dict
        if metric == "SNR":
            m_value = SNRsystem(np.real(originals_dict[i]), np.real(np.fft.ifft(fsignals_dict[i])))
        if metric == "MSE":
            m_value = mean_squared_error(np.real(np.fft.ifft(fsignals_dict[i])), np.real(originals_dict[i]))
        if metric == "RMSE":
            m_value = math.sqrt(mean_squared_error(np.real(np.fft.ifft(fsignals_dict[i])), np.real(originals_dict[i])))
        print(m_value)
#pseudocode
# 1. pass root.filename to command
# 2. command calls second function
# 3. second function uses the filter on each element of the list(creates a new object)

def switch(name):
    global filter
    filter = name
    store_files()

def downloading():
    for i in range(len(signals_dict)):
        output = preprocessed_dict[i]
        filename = asksaveasfilename(initialdir="/", title="Save as",
                                     filetypes=(("audio file", "*.wav"), ("all files", "*.*")),
                                     defaultextension=".wav")
        write(str(filename), 44100, output.astype(np.float32))

def spleeter():
    model = Separator(params_descriptor=SPLEETER_MODEL_PARAMS, model_path=SPLEETER_MODEL)
    for i in range(len(signals_dict)): #filter each file of dict
        waveform = preprocessed_dict[i]
        waveform = waveform.reshape((waveform.shape[0], 1)).astype(np.float32)
        orca_vocals = model.return_source_dictionary(waveform, 44100)['orca_vocals']
        orca_vocals = orca_vocals[:,1]
        preprocessed_dict[i] = orca_vocals

def zeroshot():
    for i in range(len(signals_dict)): #filter each file of dict
        waveform = preprocessed_dict[i]
        output_dict = inference(waveform,zeroshot_checkpoint=ZEROSHOT_CHECKPOINT, htsat_checkpoint=HTSAT_CHECKPOINT)
        print(output_dict)
        orca_vocals = output_dict['orca']
        preprocessed_dict[i] = orca_vocals




if __name__=='__main__':
    # Open file
    root = Tk()
    root.title("Signal processing")
    root.geometry("1080x700")
    dfty = Fourier_transforms()
    originals = Wavelet_transforms()

    # Add vertical and horizontal sliders
    main_frame = Frame(root)  # Create main frame
    main_frame.pack(fill=BOTH, expand=1)

    my_canvas = Canvas(main_frame)  # Create background canvas
    my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

    # Add scrollbar to canvas
    y_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
    y_scrollbar.pack(side=RIGHT, fill=Y)
    # Configure background Canvas
    my_canvas.configure(yscrollcommand=y_scrollbar.set)
    my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

    second_frame = Frame(my_canvas)  # Create another frame inside canvas
    second_frame.grid(row=0, column=0, sticky="nsew")
    # Add new frame to window inside canvas
    my_canvas.create_window((0, 0), window=second_frame, anchor="nw")

    # Create background
    image = PIL.Image.open("Orca_background.jpeg")
    backGroundImage = ImageTk.PhotoImage(image.resize((1080, 800)))
    backGroundImageLabel = Label(second_frame, image=backGroundImage)
    backGroundImageLabel.place(x=0, y=0)

    # Create labels
    ask_input_label = Label(second_frame, text="Upload audio file")
    choose_filter_label = Label(second_frame, text="Choose processing technique")
    choose_ft_label = Label(second_frame, text="Fourier transforms")
    # signals_plots = Label(root, text="Power density plot")
    wavelet_label = Label(second_frame, text="Discrete wavelet transform")
    download_label = Label(second_frame, text="Download filtered files")
    choose_metric_label = Label(second_frame, text="Choose evaluation metric")
    separator_label = Label(second_frame, text="Separate Orca vocals")
    signals_plots = Label(second_frame, text="Power density plot")

    # Put labels on the screen
    ask_input_label.grid(row=0, column=0, columnspan=4, padx=10, pady=10)
    signals_plots.grid(row=0, column=3, columnspan=4, padx=10, pady=10)
    choose_filter_label.grid(row=2, column=0, columnspan=4, padx=10, pady=10)
    choose_ft_label.grid(row=3, column=0, columnspan=4, padx=10, pady=10)
    wavelet_label.grid(row=6, column=0, columnspan=4, padx=10, pady=10)
    download_label.grid(row=8, column=0, columnspan=4, padx=10, pady=10)
    choose_metric_label.grid(row=10, column=0, columnspan=4, padx=10, pady=10)
    separator_label.grid(row=12, column=0, columnspan=4, padx=10, pady=10)

    # Create buttons
    ask_input = Button(second_frame, text="Choose file from directory", padx=60, pady=20, command=ask_input)
    spectrogram = Button(second_frame, text="Plot", padx=100, pady=20,
                         command=lambda: plot(originals_dict, second_frame, fft_dict,fs_dict))
    ftmoving_average = Button(second_frame, text="Moving average", padx=85, pady=20,
                              command=lambda: switch("Moving average"))
    ftbinomial_weighted_moving_average = Button(second_frame, text="Binomial weighted moving average", padx=40, pady=20,
                                                command=lambda: switch("Binomial weighted moving average"))
    ftgaussian_expansion_moving_average = Button(second_frame, text="Gaussian expansion moving average", padx=40,
                                                 pady=20,
                                                 command=lambda: switch("Gaussian expansion moving average"))
    ftcubic_sgfir_filter = Button(second_frame, text="Cubic-Weighted Savitzky-Golay", padx=39, pady=20,
                                  command=lambda: switch("Cubic-Weighted Savitzky-Golay"))
    ftquartic_sgfir_filter = Button(second_frame, text="Quartic-Weighted Savitzky-Golay", padx=40, pady=20,
                                    command=lambda: switch("Quartic-Weighted Savitzky-Golay"))
    ftquintic_sgfir_filter = Button(second_frame, text="Quintic-Weighted Savitzky-Golay", padx=46, pady=20,
                                    command=lambda: switch("Quintic-Weighted Savitzky-Golay"))
    ftmedian_filter = Button(second_frame, text="Median filter", padx=103, pady=20,
                             command=lambda: switch("Median filter"))
    fthampel_filter = Button(second_frame, text="Hampel filter", padx=88, pady=20,
                             command=lambda: switch("Hampel filter"))
    wavelet_button = Button(second_frame, text="Wavelet denoising", padx=88, pady=20, command=lambda: switch("Wavelet"))
    display_wdetails = Button(second_frame, text="Plot wavelet details", padx=88, pady=20,
                              command= lambda: plot_wavelet_details(originals_dict, fsignals_dict, second_frame, fft_dict,fs_dict, cD1, cD1t, cD2, cD2t, cD3, cD3t, cD4, cD4t, cD5, cD5t, XD))
    plot_wavelet = Button(second_frame, text="Plot denoised signal", padx=88, pady=20, command=lambda: plot_denoising_result(originals_dict, second_frame, fft_dict,fs_dict,fsignals_dict))
    download = Button(second_frame, text="Download", padx=120, pady=20, command=downloading, relief=SUNKEN)

    snr = Button(second_frame, text="Signal-Noise Ratio", padx=85, pady=20,
                 command=lambda: metrics("SNR"))
    mse = Button(second_frame, text="Mean Squared Error", padx=40, pady=20,
                 command=lambda: metrics("MSE"))
    rmse = Button(second_frame, text="Root Mean Squared Error", padx=40, pady=20,
                  command=lambda: metrics("RMSE"))

    separator1 = Button(second_frame, text="Separate using Spleeter", padx=120, pady=20, command=spleeter,
                        relief=SUNKEN)
    separator2 = Button(second_frame, text="Separate using Zero-shot model", padx=120, pady=20, command=zeroshot,
                        relief=SUNKEN)

    # Put buttons on screen
    ask_input.grid(row=1, column=1, columnspan=1, sticky=E)
    spectrogram.grid(row=1, column=3, columnspan=1, sticky=W)
    ftmoving_average.grid(row=4, column=1, sticky=SE)
    ftbinomial_weighted_moving_average.grid(row=4, column=2, sticky=EW)
    ftgaussian_expansion_moving_average.grid(row=4, column=3, sticky=EW)
    ftcubic_sgfir_filter.grid(row=4, column=4, sticky=SW)
    ftquartic_sgfir_filter.grid(row=5, column=1, sticky=NE)
    ftquintic_sgfir_filter.grid(row=5, column=2, sticky=EW)
    ftmedian_filter.grid(row=5, column=3, sticky=EW)
    fthampel_filter.grid(row=5, column=4, sticky=NW)
    wavelet_button.grid(row=7, column=1, sticky=E)
    display_wdetails.grid(row=7, column=2, sticky=EW)
    plot_wavelet.grid(row=7, column=3, sticky=W)
    download.grid(row=9, column=1, columnspan=3, padx=10, pady=10)
    snr.grid(row=11, column=1, sticky=E)
    mse.grid(row=11, column=2, sticky=EW)
    rmse.grid(row=11, column=3, sticky=W)
    separator1.grid(row=13, column=0, columnspan=3, sticky=E)
    separator2.grid(row=13, column=3, columnspan=3, sticky=W)

    root.mainloop()


# # Test
# filename = "20181202_25sec.mp3"
# filename= "OS_9_27_2017_08_57_00__0000.wav"
# y, fs = librosa.load(filename)
# # #os.system(filename) #play sound
# # #Rescale input
# m = len(y)
# n = pow(2, math.ceil(math.log(m)/math.log(2)))
# print(n)
#
# # Decompose signal using Discrete Wavelet Transform
# w = pywt.Wavelet('sym6')
# maxlev = pywt.dwt_max_level(len(y), w.dec_len)
# # Get detail coefficients
# coeffs = pywt.wavedec(y, 'sym8', level=5)
# plt.figure()
# for i in range(1, len(coeffs)):
#     plt.subplot(len(coeffs), 1, i)
#     plt.title(str("Level "+str(i)+" details"))
#     plt.plot(coeffs[i], label="Original coefficients")
#     coeffs[i] = pywt.threshold(coeffs[i], 0.25 * max(coeffs[i]), mode='garrote')
#     plt.plot(coeffs[i], label="Thresholded coefficients")
#     plt.legend(loc="upper right")
# XD = pywt.waverec(coeffs, 'sym8')
# if y.shape[0] < XD.shape[0]:
#     num = XD.shape[0]-y.shape[0]
#     for i in range(num):
#         y = np.append(y,0)
# subtraction = y-XD
# write(str("filtered"+".wav"), fs, subtraction.astype(np.float32))
#
# (cA5, cD5, cD4, cD3, cD2, cD1) = pywt.wavedec(y, 'sym6', level=5)
# fig, axs = plt.subplots(6)
# fig.suptitle('Wavelet details')
# axs[0].plot(y)
# axs[0].set_title('Original signal')
# axs[0].set_ylim([np.min(y), np.max(y)])
# axs[0].set_xlim([0, len(y)])
# axs[1].plot(cD1)
# axs[1].set_title('Level 1 details')
# axs[1].set_ylim([np.min(cD1), np.max(cD1)])
# axs[1].set_xlim([0, len(cD1)])
# axs[2].plot(cD2)
# axs[2].set_title('Level 2 details')
# axs[2].set_ylim([np.min(cD2), np.max(cD2)])
# axs[2].set_xlim([0, len(cD2)])
# axs[3].plot(cD3)
# axs[3].set_title('Level 3 details')
# axs[3].set_ylim([np.min(cD3), np.max(cD3)])
# axs[3].set_xlim([0, len(cD3)])
# axs[4].plot(cD4)
# axs[4].set_title('Level 4 details')
# axs[4].set_ylim([np.min(cD4), np.max(cD4)])
# axs[4].set_xlim([0, len(cD4)])
# axs[5].plot(cD5)
# axs[5].set_title('Level 5 details')
# axs[5].set_ylim([np.min(cD5), np.max(cD5)])
# axs[5].set_xlim([0, len(cD5)])
# # XD = pywt.waverec([cA5, cD5, cD4, cD3, cD2, cD1], 'sym6')
# fig, axs = plt.subplots(2)
# fig.suptitle('Wavelet details')
# axs[0].plot(y)
# axs[0].set_title('Original signal')
# axs[0].set_ylim([np.min(y), np.max(y)])
# axs[0].set_xlim([0, len(y)])
# axs[1].plot(XD, label='boat')
# axs[1].plot(subtraction, label='subtraction')
# axs[1].legend()
# axs[1].set_title('Denoised signal')
# axs[1].set_ylim([np.min(XD), np.max(XD)])
# axs[1].set_xlim([0, len(XD)])
# write(str("filtered"+".wav"), fs, XD.astype(np.float32)) #y is float32

# signal = np.fft.fft(y,n=n)
# dfty = Fourier_transforms()
# dfty.signal = signal
# Fourier_transforms.hampel_filter(dfty)
# filtered = np.real(np.fft.ifft(dfty.filtered_signal))
# i = 1
# #filtered = np.real(np.fft.ifft(np.fft.fft(y)))
# print(filtered.dtype, filtered.shape, fs)
# write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.float32)) #y is float32


