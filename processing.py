import numpy as np
import math
import os
import librosa
import PIL.Image
import scipy
from PIL import ImageTk
from scipy.io.wavfile import write
from transforms import *
from tkinter import *
from tkinter import ttk, filedialog, messagebox
from collections import defaultdict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import subprocess



#Open file
root = Tk()
root.title("Signal processing")
root.geometry("1080x700")
dfty = Fourier_transforms()

#Add vertical and horizontal sliders
main_frame = Frame(root) #Create main frame
main_frame.pack(fill=BOTH, expand=1)

my_canvas = Canvas(main_frame)#Create background canvas
my_canvas.pack(side=LEFT, fill=BOTH, expand=1)

#Add scrollbar to canvas
y_scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=my_canvas.yview)
y_scrollbar.pack(side=RIGHT, fill=Y)
#Configure background Canvas
my_canvas.configure(yscrollcommand=y_scrollbar.set)
my_canvas.bind('<Configure>', lambda e: my_canvas.configure(scrollregion=my_canvas.bbox("all")))

second_frame = Frame(my_canvas) #Create another frame inside canvas
second_frame.grid(row=0,column=0,sticky="nsew")
#Add new frame to window inside canvas
my_canvas.create_window((0,0), window=second_frame, anchor="nw")

#Create background
image = PIL.Image.open("Orca_background.jpeg")
backGroundImage = ImageTk.PhotoImage(image.resize((1080,800)))
backGroundImageLabel = Label(second_frame, image=backGroundImage)
backGroundImageLabel.place(x=0,y=0)


#Create labels
ask_input_label = Label(second_frame, text="Upload audio file")
choose_filter_label = Label(second_frame, text="Choose processing technique")
choose_ft_label = Label(second_frame, text="Fourier transforms")
#signals_plots = Label(root, text="Power density plot")
download_label = Label(second_frame, text="Download filtered files")
choose_metric_label = Label(second_frame, text="Choose evaluation metric")
separator_label = Label(second_frame, text="Separate Orca vocals")
signals_plots = Label(second_frame, text="Power density plot")

#Put labels on the screen
ask_input_label.grid(row=0,column=0, columnspan=4, padx=10, pady=10)
signals_plots.grid(row=0,column=3, columnspan=4, padx=10, pady=10)
choose_filter_label.grid(row=2,column=0, columnspan=4, padx=10, pady=10)
choose_ft_label.grid(row=3,column=0, columnspan=4, padx=10, pady=10)
download_label.grid(row=6,column=0, columnspan=4, padx=10, pady=10)
choose_metric_label.grid(row=8,column=0, columnspan=4, padx=10, pady=10)
separator_label.grid(row=10,column=0, columnspan=4, padx=10, pady=10)
#Open file
def ask_input():
    global y, fs, signals_dict, fsignals_dict,originals_dict, fs_dict
    root.filename = filedialog.askopenfilenames(initialdir="/orcasound",
                                               title="Select audios file",
                                               filetypes=(("Wav files", ".wav"), ("mp3 files", ".mp3")))
    #print(type(root.filename), root.filename) #root.filename is a tuple
    signals_dict = defaultdict(lambda: "Not present") #contains filename, not actual signals
    originals_dict = defaultdict(lambda: "Not present") #contains original resized signals
    fsignals_dict = defaultdict(lambda: "Not present")#contains filtered signals
    fs_dict = defaultdict(lambda: "Not present")#contains original signals sampling frequency
    for index, file in enumerate(root.filename):#save tuple of filenames in dict
        signals_dict[index]= file
    for i in range(len(signals_dict)): #filter each file of dict
        y, fs = librosa.load(signals_dict[i])
        # Rescale input
        m = len(y)
        n = pow(2, math.ceil(math.log(m) / math.log(2)))
        signal = np.fft.fft(y, n=n)
        dfty.signal= signal
        originals_dict[i] = signal
        fs_dict[i] = fs


def store_files(): #
    for i in range(len(signals_dict)): #filter each file of dict
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


# Plot spectrogram
def plot():
    if len(signals_dict)>1:
        messagebox.showerror('Plot Error', 'Error: Plots available only for 1 file at a time!')
    else:
        # Toplevel object which willbe treated as a new window
        newWindow = Toplevel(second_frame)
        # sets the title of the Toplevel widget
        newWindow.title("Spectrogram")
        # A Label widget to show in toplevel
        Label(newWindow).pack()
        # the figure that will contain the plot
        fig = plt.figure(figsize=(5, 5),dpi=100)
        # Plot
        f, Pxx_den = scipy.signal.welch(np.real(originals_dict[0]), fs_dict[0])
        plt.semilogy(f / 1000, Pxx_den)
        plt.title('Welch’s power spectral density estimate')
        plt.xlabel('Frequency [kHz]')
        plt.ylabel('PSD log scale(dB/kHz)')
        plt.ylim([np.min(Pxx_den), np.max(Pxx_den)])
        plt.xlim([np.min(f / 1000), np.max(f / 1000)])
        # plt.text(0, 100, '$y=x^3$', fontsize=22)

        # creating the Tkinter canvas containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=newWindow)
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas,newWindow)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()


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
    global m_value
    for i in range(len(originals_dict)): #filter each file of dict
        if metric == "SNR":
            m_value = SNRsystem(np.real(originals_dict[i]), np.real(fsignals_dict[i]))
        if metric == "MSE":
            m_value = mean_squared_error(np.real(fsignals_dict[i]), np.real(originals_dict[i]))
        if metric == "RMSE":
            m_value = math.sqrt(mean_squared_error(np.real(fsignals_dict[i]), np.real(originals_dict[i])))
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
    for i in range(len(fsignals_dict)):
        filtered = np.real(np.fft.ifft(fsignals_dict[i]))
        name = signals_dict[i]
        write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.float32))

def spleeter():
    global signals_dict
    signals_dict = defaultdict(lambda: "Not present")  # contains filename, not actual signals
    for index, file in enumerate(root.filename):  # save tuple of filenames in dict
        signals_dict[index] = file

    for i in range(len(signals_dict)): #filter each file of dict
        directory = signals_dict[i].split("/")
        for i in range(len(directory)):
            directory[i] = '"'+directory[i]+'"'
        filename = "/".join(directory)
        command_to_execute = "spleeter separate -i "+ filename +" -o spleeter_output -p 2stems-finetune.json"
        subprocess.run(command_to_execute)

def zeroshot():
    global signals_dict
    signals_dict = defaultdict(lambda: "Not present")  # contains filename, not actual signals
    for index, file in enumerate(root.filename):  # save tuple of filenames in dict
        signals_dict[index] = file

    for i in range(len(signals_dict)): #filter each file of dict
        directory = signals_dict[i].split("/")
        output = directory[-1].split(".")[0]

        for j in range(len(directory)):
            directory[j] = '"' + directory[j] + '"'
        filename = "/".join(directory)

        command_to_execute = "python inference.py -i " + filename + " -o zeroshot_output/" + output
        subprocess.run(command_to_execute)


#Create buttons
ask_input = Button(second_frame, text="Choose file from directory", padx=60, pady=20, command=ask_input)
ftmoving_average = Button(second_frame, text="Moving average", padx=85, pady=20,
                          command=lambda: switch("Moving average"))
ftbinomial_weighted_moving_average = Button(second_frame, text="Binomial weighted moving average", padx=40, pady=20,
                          command=lambda: switch("Binomial weighted moving average"))
ftgaussian_expansion_moving_average = Button(second_frame, text="Gaussian expansion moving average", padx=40, pady=20,
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
download = Button(second_frame, text="Download", padx=120, pady=20, command=downloading, relief=SUNKEN)

snr = Button(second_frame, text="Signal-Noise Ratio", padx=85, pady=20,
                          command=lambda: metrics("SNR"))
mse = Button(second_frame, text="Mean Squared Error", padx=40, pady=20,
                          command=lambda: metrics("MSE"))
rmse = Button(second_frame, text="Root Mean Squared Error", padx=40, pady=20,
                          command=lambda: metrics("RMSE"))
spectrogram = Button(second_frame, text="Plot",padx=100, pady=20,
                          command=plot)

separator1 = Button(second_frame, text="Separate using Spleeter", padx=120, pady=20, command=spleeter, relief=SUNKEN)
separator2 = Button(second_frame, text="Separate using Zero-shot model", padx=120, pady=20, command=zeroshot, relief=SUNKEN)


#Put buttons on screen
ask_input.grid(row=1,column=1, columnspan=1,sticky=E)
spectrogram.grid(row=1, column=3, columnspan=1, sticky=W)
ftmoving_average.grid(row=4, column=1, sticky=SE)
ftbinomial_weighted_moving_average.grid(row=4, column=2,sticky=EW)
ftgaussian_expansion_moving_average.grid(row=4, column=3,sticky=EW)
ftcubic_sgfir_filter.grid(row=4, column=4, sticky=SW)
ftquartic_sgfir_filter.grid(row=5, column=1,sticky=NE)
ftquintic_sgfir_filter.grid(row=5, column=2,sticky=EW)
ftmedian_filter.grid(row=5, column=3,sticky=EW)
fthampel_filter.grid(row=5, column=4, sticky=NW)
download.grid(row=7,column=1, columnspan=3, padx=10, pady=10)
snr.grid(row=9, column=1, sticky=E)
mse.grid(row=9, column=2,sticky=EW)
rmse.grid(row=9, column=3, sticky=W)
separator1.grid(row=11, column=0, columnspan=3, sticky=E)
separator2.grid(row=11, column=3, columnspan=3, sticky=W)



root.mainloop()

# Test
# filename = "20181202_25sec.mp3"
# y, fs = librosa.load(filename)
# #os.system(filename) #play sound
# #Rescale input
# m = len(y)
# n = pow(2, math.ceil(math.log(m)/math.log(2)))
# signal = np.fft.fft(y,n=n)
# dfty = Fourier_transforms()
# dfty.signal = signal
# Fourier_transforms.hampel_filter(dfty)
# filtered = np.real(np.fft.ifft(dfty.filtered_signal))
# i = 1
# #filtered = np.real(np.fft.ifft(np.fft.fft(y)))
# print(filtered.dtype, filtered.shape, fs)
# write(str("filtered"+str(i)+".wav"), fs, filtered.astype(np.float32)) #y is float32


