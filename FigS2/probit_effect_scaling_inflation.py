import sys, os, pickle, argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from scipy import integrate, optimize
from scipy.stats import norm, linregress
from sklearn.linear_model import LinearRegression, LogisticRegression

font = {'size':16}
mpl.rc('font', **font)


def generate_betas(num_genes, h2):
    # sample betas and scale them so that PRS has variance explained h2
    betas = np.random.normal(loc=0, scale=1, size=num_genes)
    a = np.sqrt(np.sum(np.square(betas))/h2)
    betas /= a
    return betas
    
def generate_case_control(num_indivs, betas, prev, status):
    thresh = norm.ppf(1-prev)
    num_genes = len(betas)
    exprs = np.empty((num_indivs, num_genes))
    batch = 10000
    num_counted = 0
    h2 = np.sum(np.square(betas)) # betas were previously scaled to make this the desired h2

    while num_counted < num_indivs:
        exprs_batch = np.random.normal(loc=0, scale=1, size=(num_indivs, num_genes))
        liab_batch = np.dot(exprs_batch, betas) + np.random.normal(loc=0, scale=np.sqrt(1-h2), size=num_indivs)
        if status=="control":
            keep_idxs = np.where(liab_batch < thresh)[0]
        elif status=="case":
            keep_idxs = np.where(liab_batch >= thresh)[0]
        keep_num = min(len(keep_idxs), num_indivs - num_counted)
        keep_idxs = keep_idxs[:keep_num]
        exprs_batch = exprs_batch[keep_idxs,:]
        exprs[num_counted:num_counted+keep_num,:] = exprs_batch
        num_counted += keep_num
        print("generating %s: %s/%s" % (status, num_counted, num_indivs))

    return exprs


def calc_likelihood(z, x, beta2):
    sigma = np.sqrt(1.0 - np.sum(np.square(beta2)))
    lhd = np.dot(z, np.log(norm.cdf(np.dot(x, beta2)/sigma))) + \
          np.dot(1-z, np.log(1-norm.cdf(np.dot(x, beta2)/sigma)))
    lhd *= -1
    return lhd

def test_slope_scaling(z, x, betas):
    eval_values = np.linspace(0.001, 3.0, 100)
    lhds = []
    for ev in eval_values:
        lhd = calc_likelihood(z, x, betas*ev) * -1 # calc_likelihood is negative for scipy minimize
        lhds.append(lhd)
        print(ev, lhd)
    fig, ax = plt.subplots()
    plt.plot(eval_values, lhds,c='k')
    plt.xlabel("probit effect scaling factor")
    plt.ylabel("log likelihood")
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.tight_layout()
    plt.show()

if __name__=="__main__":

    num_indivs = 10000
    num_genes = 20
    h2 = 0.01
    prev = 0.01
    betas = generate_betas(num_genes, h2)
    cases = generate_case_control(num_indivs, betas, prev, status="case")
    conts = generate_case_control(num_indivs, betas, prev, status="control")

    x = np.concatenate((cases, conts), axis=0)
    z = np.array([1]*cases.shape[0] + [0]*conts.shape[0])

    test_slope_scaling(z,x,betas)
    
