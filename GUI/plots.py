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

from tkinter import *
from tkinter import ttk, filedialog, messagebox
from tkinter.filedialog import asksaveasfilename

from collections import defaultdict
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)

# Plot spectrogram
def plot(originals_dict, second_frame, fft_dict,fs_dict):
    if len(originals_dict)>1:
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
        f, Pxx_den = scipy.signal.welch(np.real(fft_dict[0]), fs_dict[0])
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

# Plot wavelet details
def plot_wavelet_details(originals_dict, fsignals_dict, second_frame, fft_dict,fs_dict, cD1, cD1t, cD2, cD2t, cD3, cD3t, cD4, cD4t, cD5, cD5t, XD):
    if len(originals_dict)>1:
        messagebox.showerror('Plot Error', 'Error: Plots available only for 1 file at a time!')
    else:
        # First window
        # Toplevel object which will be treated as a new window
        newWindow = Toplevel(second_frame)
        # sets the title of the Toplevel widget
        newWindow.title("Wavelet details")
        # A Label widget to show in toplevel
        Label(newWindow).pack()
        # the figure that will contain the plot
        fig, axs = plt.subplots(6)
        fig.suptitle('Wavelet details')
        axs[0].plot(originals_dict[0])
        axs[0].set_title('Original signal')
        axs[0].set_ylim([np.min(originals_dict[0]), np.max(originals_dict[0])])
        axs[0].set_xlim([0, len(originals_dict[0])])
        axs[1].plot(cD1, label='Coefficients')
        axs[1].plot(cD1t, label='Thresholded coefficients')
        axs[1].legend()
        axs[1].set_title('Level 1 details')
        axs[1].set_ylim([np.min(cD1), np.max(cD1)])
        axs[1].set_xlim([0, len(cD1)])
        axs[2].plot(cD2, label='Coefficients')
        axs[2].plot(cD2t, label='Thresholded coefficients')
        axs[2].legend()
        axs[2].set_title('Level 2 details')
        axs[2].set_ylim([np.min(cD2), np.max(cD2)])
        axs[2].set_xlim([0, len(cD2)])
        axs[3].plot(cD3, label='Coefficients')
        axs[3].plot(cD3t, label='Thresholded coefficients')
        axs[3].set_title('Level 3 details')
        axs[3].legend()
        axs[3].set_ylim([np.min(cD3), np.max(cD3)])
        axs[3].set_xlim([0, len(cD3)])
        axs[4].plot(cD4, label='Coefficients')
        axs[4].plot(cD4t, label='Thresholded coefficients')
        axs[4].legend()
        axs[4].set_title('Level 4 details')
        axs[4].set_ylim([np.min(cD4), np.max(cD4)])
        axs[4].set_xlim([0, len(cD4)])
        axs[5].plot(cD5, label='Coefficients')
        axs[5].plot(cD5t, label='Thresholded coefficients')
        axs[5].legend()
        axs[5].set_title('Level 5 details')
        axs[5].set_ylim([np.min(cD5), np.max(cD5)])
        axs[5].set_xlim([0, len(cD5)])

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

        #Second window
        secondWindow = Toplevel(second_frame)
        # sets the title of the Toplevel widget
        secondWindow.title("Denoising result")
        # A Label widget to show in toplevel
        Label(secondWindow).pack()
        # the figure that will contain the plot
        fig, axs = plt.subplots(2)
        fig.suptitle('Wavelet denoising details')
        axs[0].plot(originals_dict[0])
        axs[0].set_title('Original signal')
        axs[0].set_ylim([np.min(originals_dict[0]), np.max(originals_dict[0])])
        axs[0].set_xlim([0, len(originals_dict[0])])
        axs[1].plot(XD, label='Extracted loud noises')
        axs[1].plot(np.real(np.fft.ifft(fsignals_dict[0])), label='Subtracted signal from loud noises')
        axs[1].legend()
        axs[1].set_title('Denoised signal')
        axs[1].set_ylim([np.min(XD), np.max(XD)])
        axs[1].set_xlim([0, len(XD)])
        # creating the Tkinter canvas containing the Matplotlib figure
        canvas = FigureCanvasTkAgg(fig, master=secondWindow)
        canvas.draw()
        # placing the canvas on the Tkinter window
        canvas.get_tk_widget().pack()
        # creating the Matplotlib toolbar
        toolbar = NavigationToolbar2Tk(canvas, secondWindow)
        toolbar.update()
        # placing the toolbar on the Tkinter window
        canvas.get_tk_widget().pack()

# Plot denoised wavelet
def plot_denoising_result(originals_dict, second_frame, fft_dict,fs_dict,fsignals_dict):
    if len(originals_dict)>1:
        messagebox.showerror('Plot Error', 'Error: Plots available only for 1 file at a time!')
    else:
        # Toplevel object which willbe treated as a new window
        newWindow = Toplevel(second_frame)
        # sets the title of the Toplevel widget
        newWindow.title("Denoising result")
        # A Label widget to show in toplevel
        Label(newWindow).pack()
        # the figure that will contain the plot
        fig, axs = plt.subplots()
        axs.plot(originals_dict[0], label="Original audio")
        axs.plot(np.real(np.fft.ifft(fsignals_dict[0])), label="Denoised audio")
        axs.legend(loc="upper right")
        axs.set_ylim([np.min(originals_dict[0]), np.max(originals_dict[0])])
        axs.set_xlim([0, len(originals_dict[0])])
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