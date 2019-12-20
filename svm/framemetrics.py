import numpy as np
import math
import pdb


def tailfraction(fraction, fit):
    assert fraction >= 0 and fraction <= 1
    return int(fraction*fit.shape[0])

def tail2mean (tailfit, fraction = .2):
    results=np.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        results[i]=fit[tailfraction(fraction,fit):,0].mean()
    return results   

def tail2angles(tailfit, fraction = .2):
    results=np.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        temp= fit[0,:]-np.mean(fit[tailfraction(1-fraction,fit):,:],0)
        results[i]=math.degrees(np.arctan2(temp[0],temp[1]))+90
    return results


def tail2tipangles(tailfit):
    results=np.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        temp= fit[tailfraction(.8,fit),:]-np.mean(fit[-3:-1,:],0)
        results[i]=math.degrees(np.arctan2(temp[1],-temp[0]))
    return results


def tail2sumangles(tailfit):
    results=np.empty(len(tailfit))
    for i,fit in enumerate(tailfit):
        temp = fit[1:,:]-fit[:-1,:]
        results[i]=np.mean(abs(np.degrees(np.arctan2(temp[:,1],temp[:,0]))))
    return results


def tail2mean2 (tailfit):
    results=np.array([])
    for fit in tailfit:
        results=np.append(results,fit[10:,1].mean())
    return results       

