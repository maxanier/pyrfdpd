import numpy as np
from scipy import signal


def rms(x):
    return np.sqrt(np.mean(np.square(abs(x))))


def nmse(x, y, logger=None):
    # Power normalization first
    rmsx = x / rms(x)
    rmsy = y / rms(y)
    mse_err = np.mean(np.square(np.abs(rmsx - rmsy)))
    mse_x = np.mean(np.square(np.abs(rmsx)))
    nmse = 10 * np.log10(mse_err / mse_x)
    if logger:
        logger.info(f"NMSE: {nmse:.3f} dB")
    else:
        print(f"NMSE: {nmse:.3f} dB")
    return nmse


def papr(x):
    papr = (max(abs(x)) / rms(x)) ** 2
    return 10 * np.log10(papr)


def acpr(x, fs, offset, bw, logger=None):
    """
    fs and offset and bw should be in Hz for example, 122.88MHz fs , give fs=122.88e6
    """
    # Spectrum
    f, Pxx = signal.welch(x, fs=fs, nperseg=2048, return_onesided=False)
    import matplotlib.pyplot as plt

    f, Pxx = np.fft.fftshift(f), 10 * np.log10(np.fft.fftshift(Pxx))

    # Calculate ACPR
    # intra-band: Center power
    fc_band = 0
    in_indx = np.where((f > (fc_band - bw / 2)) & (f < (fc_band + bw / 2)))
    B_pwr = 10 * np.log10(np.sum(10 ** (Pxx[in_indx] / 10)) / len(in_indx))

    # intra-band: ACPR1_Lower
    fc_band_lower = fc_band - offset
    lower1_indx = np.where(
        (f > (fc_band_lower - bw / 2)) & (f < (fc_band_lower + bw / 2))
    )
    ACPR1_Lower = (
        10 * np.log10(np.sum(10 ** (Pxx[lower1_indx] / 10)) / len(lower1_indx)) - B_pwr
    )

    # intra-band: ACPR1_Upper
    fc_band_upper = fc_band + offset
    upper1_indx = np.where(
        (f > (fc_band_upper - bw / 2)) & (f < (fc_band_upper + bw / 2))
    )
    ACPR1_Upper = (
        10 * np.log10(np.sum(10 ** (Pxx[upper1_indx] / 10)) / len(upper1_indx)) - B_pwr
    )

    # intra-band: ACPR2_Lower
    fc_band_lower = fc_band - offset * 2
    lower2_indx = np.where(
        (f > (fc_band_lower - bw / 2)) & (f < (fc_band_lower + bw / 2))
    )
    ACPR2_Lower = (
        10 * np.log10(np.sum(10 ** (Pxx[lower2_indx] / 10)) / len(lower2_indx)) - B_pwr
    )

    # intra-band: ACPR2_Upper
    fc_band_upper = fc_band + offset * 2
    upper2_indx = np.where(
        (f > (fc_band_upper - bw / 2)) & (f < (fc_band_upper + bw / 2))
    )
    ACPR2_Upper = (
        10 * np.log10(np.sum(10 ** (Pxx[upper2_indx] / 10)) / len(upper2_indx)) - B_pwr
    )

    if logger:
        logger.info(f"ACPR1_L: {ACPR1_Lower:.3f} dBc, ACPR1_U: {ACPR1_Upper:.3f} dBc")
        logger.info(f"ACPR2_L: {ACPR2_Lower:.3f} dBc, ACPR2_U: {ACPR2_Upper:.3f} dBc")
    else:
        print(f"ACPR1_L: {ACPR1_Lower:.3f} dBc, ACPR1_U: {ACPR1_Upper:.3f} dBc")
        print(f"ACPR2_L: {ACPR2_Lower:.3f} dBc, ACPR2_U: {ACPR2_Upper:.3f} dBc")

    return ACPR1_Lower, ACPR1_Upper, ACPR2_Lower, ACPR2_Upper
