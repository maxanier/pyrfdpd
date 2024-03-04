from cProfile import label
from tkinter import font
from .metrics import rms
import matplotlib.pyplot as plt
import numpy as np


def psd(signals: dict, fs: float = 122.88e6, filename: str = "psd.png", lang="en") -> None:
    for name, signal in signals.items():
        signal = signal / rms(signal)
        plt.psd(signal, NFFT=1024, Fs=fs, scale_by_freq=False, linewidth=3, label=name)
    plt.ylim(-100, 0)
    if (lang == "en"):
        plt.xlabel("Frequency Offset (Hz)")
        plt.ylabel("Normalized Power Spectral Density (dB/Hz)")
    elif (lang == "zh"):
        plt.xlabel("频偏 (Hz)", fontproperties='SimHei')
        plt.ylabel("归一化功率谱密度 (dB/Hz)", fontproperties='SimHei')
    else:
        raise ValueError("Language not supported")
    plt.legend(loc="upper right", fontsize=15)
    plt.savefig(filename)
    plt.close()


def amam(x, y: dict, filename="amam.png", lang="en"):
    x = x / max(abs(x))
    for name, signal in y.items():
        signal = signal / max(abs(signal))
        plt.scatter(abs(x), abs(signal), marker=".", s = 10, label=name)
    if (lang == "en"):
        plt.xlabel("Normalized Input Amplitude")
        plt.ylabel("Normalized Output Amplitude")
    elif (lang == "zh"):
        plt.xlabel("归一化输入幅度", fontproperties='SimHei')
        plt.ylabel("归一化输出幅度", fontproperties='SimHei')
    else:
        raise ValueError("Language not supported")
    plt.legend(loc = "upper left")
    plt.savefig(filename)
    plt.close()


def ampm(x, y: dict, filename="ampm.png", lang="en"):
    x = x / max(abs(x))
    for name, signal in y.items():
        signal = signal / max(abs(signal))
        plt.scatter(abs(x), np.angle(x / signal, deg=True), marker=".", s = 10, label=name)
    if (lang == "en"):
        plt.xlabel("Normalized Input Amplitude")
        plt.ylabel("Phase Difference (degree)")
    elif (lang == "zh"):
        plt.xlabel("归一化输入幅度", fontproperties='SimHei')
        plt.ylabel("相位差 (度)", fontproperties='SimHei')
    else:
        raise ValueError("Language not supported")
    plt.legend(loc="upper right")
    plt.savefig(filename)
    plt.close()
