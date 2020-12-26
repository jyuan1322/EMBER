import sys, os, pickle, argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from scipy import integrate, optimize
from scipy.stats import norm, linregress

font = {'size':14}
mpl.rc('font', **font)

def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2

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


def calc_likelihood_em(z, x, beta, sigma2, c):
    lhd = -1/(2*c) * np.inner(beta, beta) + \
          np.dot(z, np.log(norm.cdf(np.dot(x, beta)/sigma2))) + \
          np.dot(1-z, np.log(norm.cdf(-np.dot(x, beta)/sigma2)))
    # lhd *= -1
    return lhd

def probit_em(z, x, sigma2, prev, beta_scale=0.1):
    c = 1 # beta prior
    stddev = np.sqrt(1-sigma2)
    thresh = norm.ppf(1-prev)
    x = x - np.mean(x, axis=0)
    N = x.shape[0]
    M = x.shape[1]
    XTX = np.dot(x.T, x)
    betas = np.random.normal(loc=0, scale=beta_scale, size=M)
    tol = 1e-8
    prev_lhd = float("-inf")
    lhds = []
    while True:
        mu = np.dot(x, betas)
        alpha = (thresh - mu)/stddev
        beta = (thresh - mu)/stddev
        Eqyi = np.multiply(z, (mu + stddev * norm.pdf(alpha) / norm.cdf(-alpha))) + \
               np.multiply((1-z), (mu - stddev * norm.pdf(beta) / norm.cdf(beta)))

        # betas = np.dot(np.linalg.inv(1/c * np.eye(M) + 1/(stddev**2) * np.dot(x.T, x)),
        #                   1/(stddev**2) * np.dot(Eqyi, x))
        betas = np.dot(np.linalg.inv(1/c * np.eye(M) + 1/(stddev**2) * XTX),
                          1/(stddev**2) * np.dot(Eqyi, x))
        lhd = calc_likelihood_em(z, x, betas, stddev, 1)
        lhds.append(lhd)
        # print(lhd)
        if np.abs(lhd - prev_lhd) < tol:
            break
        prev_lhd = lhd
    """
    plt.plot(range(len(lhds)), lhds)
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.tight_layout()
    plt.show()
    """
    return betas, lhds

from pynverse import inversefunc
# def calc_lambd(l, H, Leye):
#     return (H.T).dot(np.linalg.inv(Leye + l * np.eye(Leye.shape[0]))).dot(np.linalg.inv(Leye + l * np.eye(Leye.shape[0]))).dot(H)
def calc_lambd2(l, H, L):
    b = np.divide(H, L+l)
    b = np.sum(np.square(b))
    return b
def probit_em_constrained(z, x, sigma2, prev, beta_scale=0.1):
    c = 1 # beta prior
    stddev = np.sqrt(1-sigma2)
    thresh = norm.ppf(1-prev)
    x = x - np.mean(x, axis=0)
    N = x.shape[0]
    M = x.shape[1]
    XTX = np.dot(x.T, x)
    betas = np.random.normal(loc=0, scale=beta_scale, size=M)
    tol = 1e-8
    prev_lhd = float("-inf")
    lhds = []
    while True:
        mu = np.dot(x, betas)
        alpha = (thresh - mu)/stddev
        beta = (thresh - mu)/stddev
        Eqyi = np.multiply(z, (mu + stddev * norm.pdf(alpha) / norm.cdf(-alpha))) + \
               np.multiply((1-z), (mu - stddev * norm.pdf(beta) / norm.cdf(beta)))

        # betas = np.dot(np.linalg.inv(1/c * np.eye(M) + 1/(stddev**2) * np.dot(x.T, x)),
        #                   1/(stddev**2) * np.dot(Eqyi, x))

        # test inverse trick
        """
        a = np.dot(x.T, x)
        a = a[:5, :5]
        L, Q = np.linalg.eigh(a)
        Leye = L*np.eye(a.shape[0])
        Qinv = np.linalg.inv(Q)
        print(a)
        print(Q.dot(Leye).dot(Qinv))

        print("-"*50)
        lambd = 1.0 * np.eye(a.shape[0])
        print(np.linalg.inv(a + lambd))
        print(Q.dot(np.linalg.inv(Leye + lambd)).dot(Qinv))
        """
        xEqy = 1/(stddev**2) * np.dot(Eqyi, x)
        # L, Q = np.linalg.eigh(1/(stddev**2) * np.dot(x.T, x))
        L, Q = np.linalg.eigh(1/(stddev**2) * XTX)
        Leye = np.diag(L)
        H = (Q.T).dot(xEqy)

        # https://pypi.org/project/pynverse/
        # solve_lambd = inversefunc(calc_lambd, args = (H, Leye), domain=0)
        solve_lambd = inversefunc(calc_lambd2, args=(H, L), domain=0)
        # print("Solution lambda:", solve_lambd(sigma2))
        """
        lambd_range = np.linspace(0,1e6,50)
        yval = []
        for l in lambd_range:
            yval.append( (H.T).dot(np.linalg.inv(Leye + l * np.eye(Leye.shape[0]))).dot(np.linalg.inv(Leye + l * np.eye(Leye.shape[0]))).dot(H) )
        print(H)
        plt.plot(lambd_range, yval)
        plt.plot(lambd_range, [sigma2]*len(lambd_range))
        plt.ylim([0, 0.1])
        plt.show()
        """
        lambd = solve_lambd(sigma2)
        # betas = np.dot(np.linalg.inv(lambd * np.eye(M) + 1/(stddev**2) * np.dot(x.T, x)),
        #                   1/(stddev**2) * np.dot(Eqyi, x))
        betas = np.dot(np.linalg.inv(lambd * np.eye(M) + 1 / (stddev ** 2) * XTX),
                          1/(stddev**2) * np.dot(Eqyi, x))

        # print(betas)
        lhd = calc_likelihood_em(z, x, betas, stddev, 1)
        lhds.append(lhd)
        # print(lhd)
        if np.abs(lhd - prev_lhd) < tol:
            break
        prev_lhd = lhd
    """
    plt.plot(range(len(lhds)), lhds)
    plt.xlabel("iteration")
    plt.ylabel("log likelihood")
    plt.tight_layout()
    plt.show()
    """
    return betas, lhds

from timeit import default_timer as timer
def probit_performance():
    PICKLE_FILE="probit_performance.pickle"
    if os.path.exists(PICKLE_FILE):
        results = pickle.load(open(PICKLE_FILE, "rb"))
        num_indivs_list = results.keys()
    else:
        num_indivs_list = [5000, 10000, 20000, 30000, 40000, 50000]
        num_trials = 10
        num_genes = 200
        h2 = 0.1
        prev = 0.01
        results = {}
        for num_indivs in num_indivs_list:
            results[num_indivs] = {"rescale":[], "rescale_niters":[], "constrain":[], "constrain_niters":[]}
            for nt in range(num_trials):
                betas = generate_betas(num_genes, h2)
                cases = generate_case_control(num_indivs, betas, prev, status="case")
                conts = generate_case_control(num_indivs, betas, prev, status="control")

                x = np.concatenate((cases, conts), axis=0)
                z = np.array([1] * cases.shape[0] + [0] * conts.shape[0])

                # run EM constrained
                sigma2 = h2  # set this to be known, as it's the same for both methods
                start_cstr = timer()
                inferred_betas_em_cstr, lhds_cstr = probit_em_constrained(z, x, sigma2, prev, beta_scale=0.01)
                end_cstr = timer()

                # run EM
                sigma2 = h2  # set this to be known, as it's the same for both methods
                start_rescale = timer()
                inferred_betas_em, lhds_rscl = probit_em(z, x, sigma2, prev, beta_scale=0.01)
                a = np.sqrt(np.sum(np.square(inferred_betas_em)) / sigma2)
                inferred_betas_em_scaled = inferred_betas_em / a
                end_rescale = timer()

                print("rescale:")
                t_rescale = end_rescale - start_rescale
                print(t_rescale)
                print("constrain:")
                t_constrain = end_cstr - start_cstr
                print(t_constrain)
                print(rsquared(betas, inferred_betas_em_scaled),
                      rsquared(betas, inferred_betas_em_cstr))
                results[num_indivs]["rescale"].append(t_rescale)
                results[num_indivs]["constrain"].append(t_constrain)
                results[num_indivs]["rescale_niters"].append(len(lhds_rscl))
                results[num_indivs]["constrain_niters"].append(len(lhds_cstr))
        pickle.dump(results, open(PICKLE_FILE, "wb"))
    rescale_means = []
    rescale_stds = []
    constrain_means = []
    constrain_stds = []
    rescale_niters_means = []
    rescale_niters_stds = []
    constrain_niters_means = []
    constrain_niters_stds = []
    for num_indivs in num_indivs_list:
        rescale_means.append(np.mean(results[num_indivs]["rescale"]))
        rescale_stds.append(np.std(results[num_indivs]["rescale"]))
        constrain_means.append(np.mean(results[num_indivs]["constrain"]))
        constrain_stds.append(np.std(results[num_indivs]["constrain"]))
        rescale_niters_means.append(np.mean(results[num_indivs]["rescale_niters"]))
        rescale_niters_stds.append(np.std(results[num_indivs]["rescale_niters"]))
        constrain_niters_means.append(np.mean(results[num_indivs]["constrain_niters"]))
        constrain_niters_stds.append(np.std(results[num_indivs]["constrain_niters"]))
    fig, ax1 = plt.subplots()
    ax1.errorbar(num_indivs_list, rescale_means, yerr=rescale_stds, capsize=5, c="r", linestyle=":")
    ax1.errorbar(num_indivs_list, constrain_means, yerr=constrain_stds, capsize=5, c="r")
    ax1.set_xlabel("Sample size")
    ax1.set_ylabel("Total exec time (s)", c="r")
    ax2 = ax1.twinx()
    ax2.errorbar(num_indivs_list, rescale_niters_means, yerr=rescale_niters_stds, capsize=5, c="b", linestyle=":")
    ax2.errorbar(num_indivs_list, constrain_niters_means, yerr=constrain_niters_stds, capsize=5, c="b")
    ax2.set_ylabel("Iterations", c="b")
    fig.tight_layout()
    plt.show()

if __name__=="__main__":
    probit_performance()