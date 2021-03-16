import pandas as pd
import numpy as np

def weighted_hs(rets, lam, window, p):
    wT = (1 - lam) / (1 - lam ** window)
    wT = 1

    wt = np.zeros(window)
    wt[window-1] = wT


    for i in range(window-2,0,-1):
        wt[i] = wt[i+1] * lam
    
    n = rets.shape[0]
    var = np.empty(n)
    var.fill(np.nan)
    save = {}

    for i in range(window,n+1):
        modeled_rets = rets[i-window:i]
        modeled_rets.index = range(window)
        modeled_rets = pd.DataFrame([modeled_rets,wt]).T
        modeled_rets.columns = ["rets","weights"]
        modeled_rets = modeled_rets.sort_values("rets")

        cutoff = int(window * p)
        modeled_rets["rets"][cutoff:] = 0

        sums = (modeled_rets["rets"] * modeled_rets["weights"]).sum()

        var[i-1] = sums
        save[i] = modeled_rets
    
    var = pd.Series(var)
    var.index = rets.index

    return  var


# 2 Subsampled RV

def subsampled_rv(rets,k):
    n = len(rets)

    # k is the block size:
    r2 = []
    for i in range(k):
        r2.append((rets.iloc[i::k] ** 2).sum())
    
    r2 = pd.Series(r2).sum()
    num = n / k
    denom = n - k - 1
    multiplier = num / denom

    rv = multiplier * r2
    rv = float(rv)

    return rv




# 3, AR 1 simulation


def ar1_simulate(rho, errors, y0):

    y = []
    y.append(y0)
    t = len(errors)
    for i in range(1,t):
        y.append(y[-1] * rho + errors[i-1])
    
    y = np.asarray(y)
    return y

