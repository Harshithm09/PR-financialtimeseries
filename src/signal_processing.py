import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import spectrogram

def generate_spectrogram(signal):
    f, t, Sxx = spectrogram(signal, fs=1)

    plt.figure()
    plt.pcolormesh(t, f, Sxx)
    plt.title("Spectrogram")
    plt.savefig("results/spectrogram.png")

    return Sxx


def plot_fft(signal):
    fft = np.fft.fft(signal)
    freq = np.fft.fftfreq(len(signal))

    plt.figure()
    plt.plot(freq, np.abs(fft))
    plt.title("Frequency Spectrum")
    plt.savefig("results/frequency.png")