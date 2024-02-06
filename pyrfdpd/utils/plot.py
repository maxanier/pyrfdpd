from .metrics import rms
import matplotlib.pyplot as plt
import numpy as np


def psd(signals: dict, fs: int = 122.88e6, filename: str = "psd.png") -> None:
    for name, signal in signals.items():
        signal = signal / rms(signal)
        plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=False, label=name)
    plt.ylim(-100, 0)
    plt.savefig(filename)
    plt.close()


def amam(x, y, filename="amam.png"):
    x = x / max(abs(x))
    y = y / max(abs(y))
    plt.scatter(abs(x), abs(y), marker=".")
    plt.savefig(filename)
    plt.close()


def ampm(x, y, filename="ampm.png"):
    x = x / max(abs(x))
    y = y / max(abs(y))
    z = np.angle(x / y, deg=True)
    plt.scatter(abs(x), z, marker=".")
    plt.savefig(filename)
    plt.close()
