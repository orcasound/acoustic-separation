import numpy as np
import math
import os
import librosa
from scipy.io.wavfile import write
from transforms import *
from tkinter import *
from tkinter import filedialog
from collections import defaultdict

#import soundfile as sf

#Open file
root = Tk()
root.title("Signal processing")
dfty = Fourier_transforms()
#Create labels
ask_input_label = Label(root, text="Upload audio file")
choose_filter_label = Label(root, text="Choose processing technique")
choose_ft_label = Label(root, text="Fourier transforms")
#signals_plots = Label(root, text="Power density plot")

#Put labels on the screen
ask_input_label.grid(row=0,column=0, columnspan=4, padx=10, pady=10)
choose_filter_label.grid(row=2,column=0, columnspan=4, padx=10, pady=10)
choose_ft_label.grid(row=3,column=0, columnspan=4, padx=10, pady=10)

#Open file
def ask_input():
    root.filename = filedialog.askopenfilenames(initialdir="/orcasound",
                                               title="Select audios file",
                                               filetypes=(("Wav files", ".wav"), ("mp3 files", ".mp3")))

def store_files(): #
    global y, fs, signals_dict, fsignals_dict
    #print(type(root.filename), root.filename) #root.filename is a tuple
    signals_dict = defaultdict(lambda: "Not present") #contains filename, not actual signals
    fsignals_dict = defaultdict(lambda: "Not present")#contains signals
    for index, file in enumerate(root.filename):#save tuple of filenames in dict
        signals_dict[index]= file
    for i in range(len(signals_dict)): #filter each file of dict
        y, fs = librosa.load(signals_dict[i])
        # Rescale input
        m = len(y)
        n = pow(2, math.ceil(math.log(m) / math.log(2)))
        signal = np.fft.fft(y, n=n)
        dfty.signal= signal
        if filter == "Moving average":
            Fourier_transforms.moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal #filtered signals saved in fsignals_dict
        if filter == "Binomial weighted moving average":
            Fourier_transforms.binomial_weighted_moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Gaussian expansion moving average":
            Fourier_transforms.gaussian_expansion_moving_average(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Cubic-Weighted Savitzky-Golay":
            Fourier_transforms.cubic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Quartic-Weighted Savitzky-Golay":
            Fourier_transforms.quartic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Quintic-Weighted Savitzky-Golay":
            Fourier_transforms.quintic_sgfir_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Median filter":
            Fourier_transforms.median_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal
        if filter == "Hampel filter":
            Fourier_transforms.hampel_filter(dfty)
            fsignals_dict[i] = dfty.filtered_signal



#pseudocode
# 1. pass root.filename to command
# 2. command calls second function
# 3. second function uses the filter on each element of the list(creates a new object)

def switch(name):
    global filter
    filter = name
    store_files()

def downloading():
    for i in range(len(fsignals_dict)):
        filtered = np.real(np.fft.ifft(fsignals_dict[i]))
        name = signals_dict[i]
        write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.int16))


#Create buttons
ask_input = Button(root, text="Choose file from directory", padx=40, pady=20, command=ask_input)
ftmoving_average = Button(root, text="Moving average", padx=40, pady=20,
                          command=lambda: switch("Moving average"))
ftbinomial_weighted_moving_average = Button(root, text="Binomial weighted moving average", padx=40, pady=20,
                          command=lambda: switch("Binomial weighted moving average"))
ftgaussian_expansion_moving_average = Button(root, text="Gaussian expansion moving average", padx=40, pady=20,
                          command=lambda: switch("Gaussian expansion moving average"))
ftcubic_sgfir_filter = Button(root, text="Cubic-Weighted Savitzky-Golay", padx=40, pady=20,
                          command=lambda: switch("Cubic-Weighted Savitzky-Golay"))
ftquartic_sgfir_filter = Button(root, text="Quartic-Weighted Savitzky-Golay", padx=40, pady=20,
                          command=lambda: switch("Quartic-Weighted Savitzky-Golay"))
ftquintic_sgfir_filter = Button(root, text="Quintic-Weighted Savitzky-Golay", padx=40, pady=20,
                          command=lambda: switch("Quintic-Weighted Savitzky-Golay"))
ftmedian_filter = Button(root, text="Median filter", padx=40, pady=20,
                          command=lambda: switch("Median filter"))
fthampel_filter = Button(root, text="Hampel filter", padx=40, pady=20,
                          command=lambda: switch("Hampel filter"))
download = Button(root, text="Download", padx=40, pady=20, command=downloading)

#Put buttons on screen
ask_input.grid(row=1,column=0, columnspan=3, padx=10, pady=10)
ftmoving_average.grid(row=4, column=1)
ftbinomial_weighted_moving_average.grid(row=4, column=2)
ftgaussian_expansion_moving_average.grid(row=4, column=3)
ftcubic_sgfir_filter.grid(row=4, column=4)
ftquartic_sgfir_filter.grid(row=5, column=1)
ftquintic_sgfir_filter.grid(row=5, column=2)
ftmedian_filter.grid(row=5, column=3)
fthampel_filter.grid(row=5, column=4)
download.grid(row=6,column=0, columnspan=3, padx=10, pady=10)
root.mainloop()


# filename = "20181202_25sec.mp3"
# y, fs = librosa.load(filename)
# #y, fs = sf.read(filename)
# #os.system(filename) #play sound
# #Rescale input
# m = len(y)
# n = pow(2, math.ceil(math.log(m)/math.log(2)))
# signal = np.fft.fft(y,n=n)
# dfty.signal = signal
# #Fourier_transforms.hampel_filter(dfty)
# #filtered = np.real(np.fft.ifft(np.real(dfty.filtered_signal)))
# i = 1
# #filtered = np.fft.ifft(np.real(signal))
# filtered = np.fft.ifft(np.fft.fft(y))
# print(filtered.dtype, filtered.shape, fs)
# write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.int16))


