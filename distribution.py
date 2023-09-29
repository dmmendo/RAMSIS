import numpy as np
import scipy.stats
from numba import jit

def my_pmf(lam,t_a,t_b,k):
    a = lam*t_a
    b = lam*t_b
    assert all(np.array([b]) > np.array([a])) and all(np.array([b]) - np.array([a]) > 1e-9), "b and a not far apart enough!"
    res = (scipy.special.gammainc(k+1,b) - scipy.special.gammainc(k+1,a))/(b-a)
    #res = np.exp(np.log(scipy.special.gammainc(k+1,b) - scipy.special.gammainc(k+1,a)) - np.log(b-a))
    res[np.array(k) < 0] = 0.
    return res

@jit(nopython=True)
def erlang_sf(k,lam,t):
    frac = lam*t
    p = 0.
    for n in range(k):
        cur_p = 1.
        for i in range(1,n+1):
            cur_p *= (frac/i)
        p += cur_p
    p *= np.exp(-frac)
    return p

@jit(nopython=True)
def erlang_cdf(k,lam,t_low,t_high):
    frac_low = lam*t_low
    frac_high = lam*t_high
    const_low = np.exp(-frac_low)
    const_high = np.exp(-frac_high)
    p = 0.
    for n in range(k):
        cur_p_low = 1.
        cur_p_high = 1.
        for i in range(1,n+1):
            cur_p_low *= (frac_low/i)
            cur_p_high *= (frac_high/i)
        p += const_low*cur_p_low - const_high*cur_p_high
    return p

def uniform_pmf(rate,tau,num_arrival):
    beta = int(2*rate*tau)
    if beta < 1. and num_arrival == 0:
        return 1.
    elif num_arrival > beta:
        return 0.
    elif num_arrival < 0:
        return 0.
    else:
        return 1/beta

@jit(nopython=True)
def poisson_pmf(lam,t,k):
    if k < 60:
        res = ( np.power(lam*t,k) * np.exp(-lam*t) )
        for i in range(1,k+1):
            res /= i
    else:
        fac = lam*t
        res = np.exp(-lam*t)
        for i in range(1,k+1):
            res *= (fac/i)
        #arr = np.arange(0,k+1).astype(np.float64)
        #arr[1:] = fac/arr[1:]
        #arr[0] = np.exp(-lam*t)
        #res = np.prod(arr)

    return res

@jit(nopython=True)
def custom_poisson_pmf(lam,t,k,mem):
    fac = lam*t
    if mem.shape[0] < k:
        mem = np.arange(1,k+1).astype(np.float64)
    arr = fac/mem[:k]
    res = np.exp(-fac)*np.prod(arr)

    return res

@jit(nopython=True)
def poisson_ppf(lam,t,percentile):
    p_total = 0.0
    k = 0
    while p_total < percentile:
        p_total += poisson_pmf(lam,t,k)
        k += 1
    return k - 1

@jit(nopython=True)
def helper_prob_Karrival_firstArrivalInInterval(lam,first_interv,second_interv,last_interv,K):
    first_interval_p = poisson_pmf(lam,first_interv,0)
    second_interval_p = 0
    for i in range(1,K+1):
        second_interval_p += poisson_pmf(lam,second_interv,i) * poisson_pmf(lam,last_interv,K-i)
    res = first_interval_p * second_interval_p
    return res

def get_poisson_pmf(lam):
    return scipy.stats.poisson(mu=lam).pmf

def get_geometric_pmf(p):
    return lambda x : ((1-p)**x)*(p)

def get_exponential_pmf(mu):
    return lambda x : mu*np.exp(-mu*x)

def get_exponential_cdf(mu):
    def func(x): 
        if x < 0:
            return 0
        else:
            return 1 - np.exp(-mu*x)
    return func

def discretize_dist(pmf,threshold=0.99999999999999,start_k = 0,size_limit=1000):
    k = start_k
    cdf = 0
    discrete_dist = []
    while True:
        new_p = pmf(k)
        cdf += new_p
        discrete_dist.append(new_p)
        if cdf > threshold or len(discrete_dist) >= size_limit:
            break
        k += 1
    discrete_dist = np.array(discrete_dist)
    return discrete_dist/np.sum(discrete_dist)

def plot_discrete_hist(data):
    d = np.diff(np.unique(data)).min()
    left_of_first_bin = data.min() - float(d)/2
    right_of_last_bin = data.max() + float(d)/2
    plt.hist(data, np.arange(left_of_first_bin, right_of_last_bin + d, d))