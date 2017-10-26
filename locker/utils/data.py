import numpy as np
import sys

def peakdet(v, delta=None):
    """
    Peak detection. Modified version from https://gist.github.com/endolith/250860

    A point counts as peak if it is maximal and is preceeded by a value lower by delta.

    :param v: array of values
    :param delta: threshold; set to 99.9%ile - median of v if None
    :return: maxima, maximum indices, minima, minimum indices
    """
    maxtab = []
    maxidx = []

    mintab = []
    minidx = []
    v = np.asarray(v)
    if delta is None:
        up = int(np.min([1e5, len(v)]))
        tmp = np.abs(v[:up])
        delta = np.percentile(tmp, 99.9) - np.percentile(tmp, 50)

    if not np.isscalar(delta):
        sys.exit('Input argument delta must be a scalar')

    if delta <= 0:
        sys.exit('Input argument delta must be positive')

    mn, mx = np.Inf, -np.Inf
    mnpos, mxpos = np.NaN, np.NaN

    lookformax = True
    n = len(v)
    for i in range(len(v)):
        this = v[i]
        if this > mx:
            mx = this
            mxpos = i

        if this < mn:
            mn = this
            mnpos = i

        if lookformax:
            if this < mx - delta:
                maxtab.append(mx)
                maxidx.append(mxpos)
                mn = this
                mnpos = i
                lookformax = False

        else:
            if this > mn + delta:
                mintab.append(mn)
                minidx.append(mnpos)
                mx = this
                mxpos = i
                lookformax = True

    return np.asarray(maxtab), np.asarray(maxidx, dtype=int), np.asarray(mintab), np.asarray(minidx, dtype=int)

