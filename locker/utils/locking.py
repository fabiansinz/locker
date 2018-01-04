import functools
import itertools
import numpy as np
from scipy import optimize, stats, signal


import pycircstat as circ

from .data import peakdet


def vector_strength_at(f, trial, alpha=None):
    if alpha is None:
        return 1 - circ.var((trial % (1. / f)) * f * 2 * np.pi)
    else:
        return 1 - circ.var((trial % (1. / f)) * f * 2 * np.pi), np.sqrt(- np.log(alpha) / len(trial))


def _neg_vs_at(f, spikes):
    return -np.mean([1 - circ.var((trial % (1. / f)) * f * 2 * np.pi) for trial in spikes])


def find_best_locking(spikes, f0, tol=3):
    """
    Locally searches for a maximum in vector strength for a collection of spikes.

    The vector strength is locally maximized with fminbound within f0+-tol. There are two exceptions
    to the search range:

    * if two initial guesses are closer then tol, then their mean is taken as search boundary
    * if all initial guesses are negative or positive, the search intervals are chosen such that the result is
      again negative or positive, respectively.

    :param spikes: array of spike times or list thereof
    :param f0: list of initial guesses
    :param tol: search range is +-tol in Hz
    :return: best locking frequencies, corresponding vector strength
    """
    max_w, max_v = [], []
    if type(spikes) is not list:
        spikes = [spikes]

    # at an initial and end value to fundamental to generate the search intervals
    f0 = np.array(f0)
    f0.sort()

    # --- make sure that fundamentals + boundaries stay negative positive if they were before
    if f0[0] > 0:
        f0 = np.hstack((max(f0[0] - tol, 0), f0, f0[-1] + tol))
    elif f0[-1] < 0:
        f0 = np.hstack((f0[0] - tol, f0, min(f0[-1] + tol, 0)))
    else:
        f0 = np.hstack((f0[0] - tol, f0, f0[-1] + tol))

    for freq_before, freq, freq_after in zip(f0[:-2], f0[1:-1], f0[2:]):
        # search in freq +- tol unless we get too close to another fundamental.
        upper = min(freq + tol, (freq + freq_after) / 2)
        lower = max(freq - tol, (freq_before + freq) / 2)
        obj = functools.partial(_neg_vs_at, spikes=spikes)
        f_opt = optimize.fminbound(obj, lower, upper)
        max_w.append(f_opt)
        max_v.append(-obj(f_opt))

    return np.array(max_w), np.array(max_v)


def find_significant_peaks(spikes, w, spectrum, peak_dict, threshold, tol=3.,
                           upper_cutoff=2000):
    if not threshold > 0:
        print("Threshold value %.4f is not allowed" % threshold)
        return []
    # find peaks in spectrum that are greater or equal than the threshold
    max_vs, max_idx, _, _ = peakdet(spectrum, delta=threshold * .9)
    max_vs, max_idx = max_vs[threshold <= max_vs], max_idx[threshold <= max_vs]
    max_w = w[max_idx]

    # get rid of everythings that is above the frequency cutoff
    idx = np.abs(max_w) < upper_cutoff
    if idx.sum() == 0:  # no sigificant peak was found
        print('No significant peak found')
        return []
    max_w = max_w[idx]
    max_vs = max_vs[idx]

    # refine the found maxima
    max_w_ref, max_vs_ref = find_best_locking(spikes, max_w, tol=tol)

    # make them all sorted in the right order
    idx = np.argsort(max_w)
    max_w, max_vs = max_w[idx], max_vs[idx]
    idx = np.argsort(max_w_ref)
    max_w_ref, max_vs_ref = max_w_ref[idx], max_vs_ref[idx]

    for name, freq in peak_dict.items():
        idx = np.argmin(np.abs(max_w - freq))
        if np.abs(max_w[idx] - freq) < tol:
            print("\t\tAdjusting %s: %.2f --> %.2f" % (name, freq, max_w[idx]))
            peak_dict[name] = max_w[idx]

    coeffs = [(name, peak_dict[name], np.arange(-5, 6)) for name in peak_dict]
    coeff_names, coeff_f, coeff_facs = zip(*coeffs)

    ret = []

    for maw, ma, maw_r, ma_r in zip(max_w, max_vs, max_w_ref, max_vs_ref):
        for facs in itertools.product(*coeff_facs):
            cur_freq = np.dot(facs, coeff_f)
            if np.abs(cur_freq) > upper_cutoff:
                continue

            if np.abs(maw - cur_freq) < tol:
                tmp = dict(zip(coeff_names, facs))
                tmp['frequency'] = maw
                tmp['vector_strength'] = ma
                tmp['tolerance'] = tol
                tmp['refined'] = 0
                ret.append(tmp)

            if np.abs(maw_r - cur_freq) < tol:
                tmp = dict(zip(coeff_names, facs))
                tmp['frequency'] = maw_r
                tmp['vector_strength'] = ma_r
                tmp['tolerance'] = tol
                tmp['refined'] = 1
                ret.append(tmp)
    return ret
