import socket

from playsound import playsound

import matplotlib.pyplot as plt
import numpy as np
import time

from scipy.io import wavfile
from scipy.signal import butter,filtfilt
from scipy.signal import find_peaks

from statistics import mean

import timeit

FFT_SIZE = 1024 * 4
FFT_OVERLAP = 0.2

# Número de picos do espectrograma
NUM_PEAKS = 1

LOW_FREQ_MIN = 1 # Hz
LOW_FREQ_MAX = 200 # Hz
MID_FREQ_MIN = 201 # Hz
MID_FREQ_MAX = 800 # Hz
HIGH_FREQ_MIN = 801 # Hz
HIGH_FREQ_MAX = 2000 # Hz

TENS_MAX_MOD = 10 # Duty (%)
TENS_LOW_FREQ_MIN = 1 # Hz
TENS_LOW_FREQ_MAX = 30 # Hz
TENS_MID_FREQ_MIN = 20 # Hz
TENS_MID_FREQ_MAX = 60 # Hz
TENS_HIGH_FREQ_MIN = 40 # Hz
TENS_HIGH_FREQ_MAX = 90 # Hz

ITER_JUMP = 1000

def shift_with_padding(array, n):
    shifted = array[n:]
    return np.append(shifted, np.zeros(n))

def get_peaks(ampArray, freqArray, top):
    # ampArray = ampArray[10:len(ampArray)]
    # freqArray = freqArray[10:len(freqArray)]

    sortedAmp = np.sort(ampArray)

    topAmps = sortedAmp[-top:] # Pega 'top' últimos elementos
    topAmps = topAmps[::-1] # Inverte

    topFreqs = []

    for amp in topAmps:
        topAmpIndex = np.where(ampArray == amp)
        topFreqs.append(freqArray[topAmpIndex[0][0]])

    return topAmps, topFreqs

def get_freq_index(freq_array, freq):
    
    last_i = 0

    for i in range(0, len(freq_array)):
        if freq_array[i] <= freq:
            last_i = i
            continue

        return last_i    

def pos(lst):
    return [x for x in lst if x > 0] or None

def butter_lowpass_filter(data, cutoff, fs, order):
    normal_cutoff = cutoff / (0.5 * fs) # 0.5 * fs = nyquist cutoff frequency
    # Get the filter coefficients 
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = filtfilt(b, a, data)
    return y

def log(channel, message, linebreaks=0):
    breaks = ""
    for i in range(linebreaks):
        breaks += '\n'

    print("[{0}] - {1}{2}".format(channel, message, breaks))