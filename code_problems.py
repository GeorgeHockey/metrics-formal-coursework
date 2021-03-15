# 2 Subsampled RV
import pandas as pd
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
import numpy as np

def ar1_simulate(rho, errors, y0):

    y = []
    y.append(y0)
    t = len(errors)
    for i in range(1,t):
        y.append(y[-1] * rho + errors[i])
    
    y = np.asarray(y)
    return y

