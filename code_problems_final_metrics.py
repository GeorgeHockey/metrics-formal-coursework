### MY CODE

import pandas as pd
import numpy as np

def weighted_hs(rets, lam, window, p):
    wT = (1 - lam) / (1 - lam ** window)

    wt = np.zeros(window)
    wt[window-1] = wT

    for t in range(0,window-1):
        wt[t] = wT * (lam ** (window - 1 - t))
    
    n = rets.shape[0]
    v = np.empty(n)
    v.fill(np.nan)

    for i in range(window,n+1):
        modeled_rets = rets[i-window:i]
        modeled_rets.index = range(window)
        modeled_rets = pd.DataFrame([modeled_rets,wt]).T
        modeled_rets.columns = ["rets","weights"]
        modeled_rets = modeled_rets.sort_values("rets")
        modeled_rets["weights"] = modeled_rets["weights"].cumsum()
        modeled_rets = np.asarray(modeled_rets)
        weights = modeled_rets[:,1]
        returns = modeled_rets[:,0]
      
   

        v[i-1] = returns[np.min(np.where(weights > p))]

    v = pd.Series(-v)
    v.index = rets.index

    return v

# 2 Subsampled RV

def subsampled_rv(rets,k):
    n = len(rets)

    # k is the block size:
    r2 = []
    for i in range(0,n-k+1):
        r2.append((rets.iloc[i:k+i] ).sum() ** 2)
    
    r2 = pd.Series(r2).sum()
    num = n / k
    denom = n - k + 1
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

