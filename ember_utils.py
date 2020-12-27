import sys, os, pickle, argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from scipy import integrate, optimize
from scipy.stats import norm, linregress
from sklearn.linear_model import LinearRegression, LogisticRegression


def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2

def calc_h2(cases, conts, prev):
    num_cases = cases.shape[0]
    num_conts = conts.shape[0]
    num_indivs = num_cases + num_conts
    P = num_cases / num_indivs
    K = prev
    t = norm.ppf(1-K)

    x = np.concatenate((cases, conts), axis=0)
    y = np.array([1]*cases.shape[0] + [0]*conts.shape[0])

    Gijs = np.corrcoef(x, rowvar=True)
    Zijs = np.outer((y-P), (y-P)) / (P*(1-P))

    Gijs_list = Gijs[np.triu_indices(Gijs.shape[0], k=1)]
    Zijs_list = Zijs[np.triu_indices(Zijs.shape[0], k=1)]

    reg = LinearRegression().fit(Gijs_list.reshape(-1,1), Zijs_list) # fit(X,y)
    slope = reg.coef_[0]
    const = P*(1-P) / (K**2 * (1-K)**2) * norm.pdf(t)**2
    h2 = slope / const

    # print("beta alt: ")
    # print(Zijs_list.dot(Gijs_list) / np.sum(np.square(Gijs_list, Gijs_list)) / const)
    return h2

def calc_h2_fast(y, x, prev):
    num_indivs = len(y)
    P = np.sum(y) / num_indivs
    K = prev
    t = norm.ppf(1-K)


    x = np.squeeze(x)
    if x.ndim == 1:
        M = 1
        mean = np.mean(x)
        vrnc = np.var(x)
        numer = 0
        denom = 0
        x = (x - mean)/vrnc
        for i in range(num_indivs):
            Zijs = (y[i] - P) * (y[i+1:]-P) / (P*(1-P))
            Gijs = x[i] * x[i+1:] / M
            numer += np.dot(Zijs, Gijs)
            denom += np.sum(np.square(Gijs))
    else:
        M = x.shape[1]
        nmeans = np.mean(x, axis=1)
        nstds = np.std(x, axis=1)
        numer = 0
        denom = 0
        x = np.divide(x - nmeans[:,np.newaxis], nstds[:,np.newaxis])
        for i in range(num_indivs):
            print(i, num_indivs)
            Zijs = (y[i] - P) * (y[i+1:]-P) / (P*(1-P))
            # Gijs = np.divide( (x[i+1:] - nmeans[i+1:,np.newaxis]).dot(x[i,:] - nmeans[i]) / nstds[i], 
            #                   nstds[i+1:]) / M
            Gijs = np.dot(x[i+1:,:], x[i,:].T) / M
            numer += Zijs.dot(Gijs)
            denom += np.sum(np.square(Gijs))
    const = P*(1-P) / (K**2 * (1-K)**2) * norm.pdf(t)**2
    h2 = numer / denom / const
    return h2

def calc_h2_quantitative(y,x,verbose=False):
    P = np.mean(y)
    corry = np.outer((y-P), (y-P)) / np.var(y)
    corrx = np.corrcoef(x, rowvar=True)
    cy_list = corry[np.triu_indices(corry.shape[0], k=1)]
    cx_list = corrx[np.triu_indices(corrx.shape[0], k=1)]

    reg = LinearRegression().fit(cx_list.reshape(-1,1), cy_list) # fit(X,y)
    slope = reg.coef_[0]
    if verbose:
        plt.scatter(cx_list, cy_list,s=2)
        x = np.linspace(min(cx_list), max(cx_list), 50)
        plt.plot(x, x*slope + reg.intercept_,c='k')
        plt.title("%s, %s" % (slope, reg.intercept_))
        plt.show()
    return slope