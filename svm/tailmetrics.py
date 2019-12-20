import numpy as np
import peakdetector
import sys
from scipy import fft
from framemetrics import *


def maxangle(tailfit):
    """Maximum tail angle"""
    return np.absolute(tail2angles(tailfit)).max()

def meansumangle(tailfit):
    """Mean tail curvature"""
    return tail2sumangles(tailfit).mean()
def maxsumangle(tailfit):
    """Maximum tail curvature"""
    # print(tailfit)
    return np.absolute(tail2sumangles(tailfit)).max()
def diffangles(tailfit):
    """Mean tail curvature vs tail angle"""
    return np.absolute(tail2angles(tailfit)-tail2sumangles(tailfit)).mean()

def numpeaks(tailfit):
    """Number of peaks in tail angle"""
    peaks=peakdetector.peakdetold(tail2angles(tailfit),4)
    peaks = [p[0] for p in peaks[0]+peaks[1]]
    return len(peaks)

def diffpeaks(tailfit):
    """Mean # of frames between peaks"""
    peaks=peakdetector.peakdetold(tail2angles(tailfit),4)
    peaks = [p[0] for p in peaks[0]+peaks[1]]
    if len(peaks) > 1:
        val = np.diff(peaks).mean()
        if not np.isnan(val) and not np.isinf(val): #nan or inf
            return val
        else:
            return 0
    else:
        return 0

def tipmean(tailfit):
    """Mean tail position"""
    results=np.zeros(len(tailfit))
    for i, fit in enumerate(tailfit):
        results[i]=fit[tailfraction(.3,tailfit[0]):-1,0].mean()
    return results.mean()

def tipmean2(tailfit):
    """Mean near-tip tail position"""
    results=np.zeros(len(tailfit))
    for i, fit in enumerate(tailfit):
        results[i]=fit[tailfraction(.8,tailfit[0]):,0].mean()
    return results.mean()

def tipmax(tailfit):
    """Maximum tailtip deviation"""
    results=np.zeros(len(tailfit))
    for i, fit in enumerate(tailfit):
        results[i]=np.abs(fit[:,1]-fit[0,1]).max()
    return results.max()

def tipvar(tailfit):
    """Variance in tail position"""
    results=np.zeros(len(tailfit))
    for i, fit in enumerate(tailfit):
        results[i]=fit[tailfraction(.3,tailfit[0]):-1,0].mean()
    return results.var()

def freql(tailfit):
    """Low frequency power of tail angles"""
    angles = tail2angles(tailfit)
    n = len(angles) # length of the signal
    Y = fft(angles)/n # fft computing and normalization
    Y = Y[range(n/2)]
    return Y[1:2].mean()
def freqm(tailfit):
    """Medium frequency power of tail angles"""
    angles = tail2angles(tailfit)
    n = len(angles) # length of the signal
    Y = fft(angles)/n # fft computing and normalization
    Y = Y[range(n/2)]
    return Y[3:6].mean()
##def freqh(tailfit):
##    angles = tail2angles(tailfit)
##    n = len(angles) # length of the signal
##    Y = fft(angles)/n # fft computing and normalization
##    Y = Y[range(n/2)]
##    return Y[6:9].mean()

def tipvsbend(tailfit):
    """Tail angle vs tip angle"""
    angles = tail2angles(tailfit)
    tips = tail2tipangles(tailfit)
    return np.abs(angles-tips).mean()

def meantip(tailfit):
    """Mean tail tip angle"""
    return tail2tipangles(tailfit).mean()

def meanabstip(tailfit):
    """Mean tip angle deviation"""
    return np.abs(tail2tipangles(tailfit)).mean()

def maxbendvstip(tailfit):
    """Tail angle vs tip angle at frame of maximum tail angle"""
    i=tail2angles(tailfit).argmax()
    return tail2tipangles(tailfit)[i]/tail2sumangles(tailfit)[i]

# metric_list = [i for i in sys.modules[__name__].__dict__.copy().itervalues() if isfunction(i) and i.__module__ == __name__]

