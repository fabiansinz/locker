import numpy as np
from scipy import stats
from scipy.signal import butter, filtfilt
import warnings

from .data import peakdet


def val_at(w, f, w0, tol=2):
    return np.max(f[np.abs(w - w0) < tol])


def second_order_critical_vector_strength(spikes, alpha=0.001):
    spikes_per_trial = [len(s) for s in spikes]
    poiss_rate = np.mean(spikes_per_trial)
    r = np.linspace(0, 2, 10000)
    dr = r[1] - r[0]
    mu = np.sum(2 * poiss_rate * r ** 2 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
            1 - np.exp(-poiss_rate))) * dr
    s = np.sum(2 * poiss_rate * r ** 3 * np.exp(poiss_rate * np.exp(-r ** 2) - poiss_rate - r ** 2) / (
            1 - np.exp(-poiss_rate))) * dr
    s2 = np.sqrt(s - mu ** 2.)
    threshold = stats.norm.ppf(1 - alpha, loc=mu,
                               scale=s2 / np.sqrt(len(spikes_per_trial)))  # use central limit theorem

    return threshold


def butter_lowpass_filter(data, highcut, fs, order=5):
    b, a = butter_lowpass(highcut, fs, order=order)
    y = filtfilt(b, a, data)
    return y


def butter_lowpass(highcut, fs, order=5):
    nyq = 0.5 * fs
    high = highcut / nyq
    b, a = butter(order, high, btype='low')
    return b, a


def normalize_signal(eod, samplerate, norm_window=.5):
    max_time = len(eod) / samplerate

    if norm_window > max_time * .5:
        warnings.warn("norm_window is larger than trace. Not normalizing anything!")
        return eod

    w = np.ones(samplerate * norm_window)
    w[:] /= len(w)
    local_std = np.sqrt(np.correlate(eod ** 2., w, mode='same') - np.correlate(eod, w, mode='same') ** 2.)
    local_mean = np.correlate(eod, w, mode='same')
    return (eod - local_mean) / local_std


def amplitude_spec(dat, samplerate):
    return np.abs(np.fft.fft(dat)), np.fft.fftfreq(len(dat), 1. / samplerate)


def estimate_fundamental(dat, samplerate, highcut=3000, normalize=-1, four_search_range=(-20, 20)):
    """
    Estimates the fundamental frequency in the data.

    :param dat: one dimensional array
    :param samplerate: sampling rate of that array
    :param highcut: highcut for the filter
    :param normalize: whether to normalize the data or not
    :param four_search_range: search range in the Fourier domain in Hz
    :return: fundamental frequency
    """
    filtered_data = butter_lowpass_filter(dat, highcut, samplerate, order=5)

    if normalize > 0:
        filtered_data = normalize_signal(filtered_data, samplerate, norm_window=normalize)

    n = len(filtered_data)
    t = np.arange(n) / samplerate

    _, eod_peak_idx, _, eod_trough_idx = peakdet(filtered_data)

    diff_eod_peak_t = np.diff(t[eod_peak_idx])
    freq_from_median = 1 / np.median(diff_eod_peak_t)
    f, w = amplitude_spec(filtered_data, samplerate)

    f[(w < freq_from_median + four_search_range[0]) & (w > freq_from_median + four_search_range[1])] = -np.Inf
    freq_from_fourier = np.argmax(f)

    return abs(w[freq_from_fourier])


def get_best_time_window(data, samplerate, fundamental_frequency, eod_cycles):
    eod_peaks1, eod_peak_idx1, _, _ = peakdet(data)

    max_time = len(data) / samplerate
    time_for_eod_cycles_in_window = eod_cycles / fundamental_frequency

    if time_for_eod_cycles_in_window > max_time * .2:
        time_for_eod_cycles_in_window = max_time * .2
        warnings.warn("You are reqeusting a window that is too long. Using T=%f" % (time_for_eod_cycles_in_window,))

    sample_points_in_window = int(fundamental_frequency * time_for_eod_cycles_in_window)

    tApp = np.arange(len(data)) / samplerate
    w1 = np.ones(sample_points_in_window) / sample_points_in_window

    local_mean = np.correlate(eod_peaks1, w1, mode='valid')
    local_std = np.sqrt(np.correlate(eod_peaks1 ** 2., w1, mode='valid') - local_mean ** 2.)
    COV = local_std / local_mean

    mi = min(COV)
    for ind, j in enumerate(COV):
        if j == mi:
            v = (eod_peak_idx1[ind])

    idx = (tApp >= tApp[v]) & (tApp < tApp[v] + time_for_eod_cycles_in_window)
    tApp = tApp[idx]
    dat_app = data[idx]
    tApp = tApp - tApp[0]

    return tApp, dat_app


def get_harm_coeff(time, dat, fundamental_freq, harmonics):
    ret = np.zeros((harmonics, 2))
    VR = fundamental_freq * 2. * np.pi
    # combCoeff = np.zeros((harmonics, 1))

    rec = 0 * time

    for i, ti in enumerate(np.arange(1, harmonics + 1)):
        V1 = np.sin(time * ti * VR)
        V2 = np.cos(time * ti * VR)
        V1 = V1 / np.sqrt(sum(V1 ** 2.))
        V2 = V2 / np.sqrt(sum(V2 ** 2.))

        coeff_sin, coeff_cos = np.dot(V1, dat), np.dot(V2, dat)

        VS = coeff_sin * V1
        VC = coeff_cos * V2
        rec = rec + VS + VC

        ret[i, :] = [coeff_sin, coeff_cos]

    return ret  # combCoeff
