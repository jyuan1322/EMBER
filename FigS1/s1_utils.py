import sys, os, pickle, argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from pprint import pprint
from scipy import integrate, optimize
from scipy.stats import norm, linregress
from sklearn.linear_model import LinearRegression, LogisticRegression

font = {'size':18}
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

def neg_probit_likelihood_no_sigma(beta2, *args):
    sigma2, z, x = args
    c = 1 # variance of beta prior distribution
    x = x - np.mean(x, axis=0)
    # sigma = 1.0
    # sigma = np.sqrt(1.0 - np.sum(np.square(beta2)))
    """
        NOTE: you need some kind of beta prior so that magnitude of extreme values is not too high
    """
    denom = np.sqrt(1-sigma2)
    lhd = -1.0/(2*c) * np.dot(beta2, beta2) + \
          np.dot(z, np.log(norm.cdf(np.dot(x, beta2)/denom))) + \
          np.dot(1-z, np.log(1-norm.cdf(np.dot(x, beta2)/denom)))
    """
    lhd = np.dot(z, np.log(norm.cdf(np.dot(x, beta2)/sigma))) + \
          np.dot(1-z, np.log(1-norm.cdf(np.dot(x, beta2)/sigma)))
    """
    # minimize the negative likelihood to maximize
    lhd *= -1
    return lhd

def calc_likelihood(z, x, beta2):
    sigma = np.sqrt(1.0 - np.sum(np.square(beta2)))
    lhd = np.dot(z, np.log(norm.cdf(np.dot(x, beta2)/sigma))) + \
          np.dot(1-z, np.log(1-norm.cdf(np.dot(x, beta2)/sigma)))
    lhd *= -1
    return lhd

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
    M = x.shape[1]

    x = np.squeeze(x)
    if x.ndim == 1:
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
        nmeans = np.mean(x, axis=1)
        nstds = np.std(x, axis=1)
        numer = 0
        denom = 0
        x = np.divide(x - nmeans[:,np.newaxis], nstds[:,np.newaxis])
        for i in range(num_indivs):
            Zijs = (y[i] - P) * (y[i+1:]-P) / (P*(1-P))
            # Gijs = np.divide( (x[i+1:] - nmeans[i+1:,np.newaxis]).dot(x[i,:] - nmeans[i]) / nstds[i], 
            #                   nstds[i+1:]) / M
            Gijs = np.dot(x[i+1:,:], x[i,:].T) / M
            numer += Zijs.dot(Gijs)
            denom += np.sum(np.square(Gijs))
    const = P*(1-P) / (K**2 * (1-K)**2) * norm.pdf(t)**2
    h2 = numer / denom / const
    return h2




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
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
def plot_results(betas, logit_betas, inferred_betas_em, inferred_betas_em_scaled, inferred_betas_em_cstr):
    fig, ax = plt.subplots(figsize=(10, 10))
    # ax.scatter(betas, inferred_betas_no_sigma, label="liability no sigma", c='r', alpha=0.3)
    # ax.scatter(betas, inferred_betas_no_sigma_scaled, label="liability sig scaled", c='r')
    ax.scatter(betas, logit_betas, c='k', alpha=0.8, s=30,
               label=r"logistic $r^2$=%0.4f" % (rsquared(betas, logit_betas)))
    ax.scatter(betas, inferred_betas_em, c='r', alpha=0.8, s=30,
               label=r"probit unscaled $r^2$=%0.4f" % (rsquared(betas, inferred_betas_em)))
    # ax.scatter(betas, inferred_betas_em_scaled, label="liability em scaled", c='b')
    ax.scatter(betas, inferred_betas_em_scaled, c='g', alpha=0.8, s=30,
               label=r"probit scaled $r^2$=%0.4f" % (rsquared(betas, inferred_betas_em_scaled)))
    ax.scatter(betas, inferred_betas_em_cstr, c='b', alpha=0.8, s=30,
               label=r"probit constrained $r^2$=%0.4f" % (rsquared(betas, inferred_betas_em_cstr)))
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.xlabel("True effect size")
    plt.ylabel("Inferred effect size")
    box = ax.get_position()
    ax.set_position([box.x0, box.y0 + box.height * 0.1,
                     box.width, box.height * 0.9])
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1),
              frameon=False, shadow=True, ncol=2)
    # plt.legend()

    ax2 = zoomed_inset_axes(ax, 1.5, loc=4)
    ax2.scatter(betas, inferred_betas_em_scaled, c='g', alpha=0.4, s=60)
    ax2.scatter(betas, inferred_betas_em_cstr, c='b', alpha=0.4, s=60)
    lims = [np.min([ax2.get_xlim(), ax2.get_ylim()]),  # min of both axes
            np.max([ax2.get_xlim(), ax2.get_ylim()]),  # max of both axes
            ]
    ax2.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax2.set_aspect('equal')
    ax2.set_xlim(lims)
    ax2.set_ylim(lims)
    ax2.xaxis.tick_top()
    ax2.yaxis.tick_right()
    mark_inset(ax, ax2, loc1=1, loc2=3, fc="none", ec="0.5")
    plt.savefig("probit_constrained_single_run.png", format="png", bbox_inches="tight")
    plt.show()

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
                sigma2 = h2  # for now, skip this step
                start_cstr = timer()
                inferred_betas_em_cstr, lhds_cstr = probit_em_constrained(z, x, sigma2, prev, beta_scale=0.01)
                end_cstr = timer()

                # run EM
                # sigma2 = calc_h2(cases, conts, prev)
                sigma2 = h2  # for now, skip this step
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

    # probit_performance()
    # sys.exit(0)

    PICKLE_FILE = "probit_constrained_single_run.pickle"
    if os.path.exists(PICKLE_FILE):
        (betas, logit_betas, inferred_betas_em, inferred_betas_em_scaled, inferred_betas_em_cstr) = \
            pickle.load(open(PICKLE_FILE, "rb"))
        plot_results(betas, logit_betas, inferred_betas_em, inferred_betas_em_scaled, inferred_betas_em_cstr)
    else:
        num_indivs = 5000
        num_genes = 20
        h2 = 0.1
        prev = 0.01
        betas = generate_betas(num_genes, h2)
        cases = generate_case_control(num_indivs, betas, prev, status="case")
        conts = generate_case_control(num_indivs, betas, prev, status="control")

        x = np.concatenate((cases, conts), axis=0)
        z = np.array([1]*cases.shape[0] + [0]*conts.shape[0])

        """
        res1 = calc_h2(cases, conts, prev)
        print(res1)
        res2 = calc_h2_fast(z, x, prev)
        print(res2)
        sys.exit(0)
        """

        clf = LogisticRegression().fit(x, z)
        logit_betas = clf.coef_[0]

        # run EM constrained
        sigma2 = h2 # for now, skip this step
        start_cstr = timer()
        inferred_betas_em_cstr, _ = probit_em_constrained(z, x, sigma2, prev)
        end_cstr = timer()

        # run EM
        # sigma2 = calc_h2(cases, conts, prev)
        sigma2 = h2 # for now, skip this step
        start_rescale = timer()
        inferred_betas_em, _ = probit_em(z, x, sigma2, prev)
        a = np.sqrt(np.sum(np.square(inferred_betas_em))/sigma2)
        inferred_betas_em_scaled = inferred_betas_em/a
        end_rescale = timer()

        """
        initial_guess = [0.0]*num_genes
        sigma2 = calc_h2(cases, conts, prev)
        result = optimize.minimize(neg_probit_likelihood_no_sigma, initial_guess, method = 'Nelder-Mead',
                                   args=(sigma2, z, x)) # options={'adaptive':True}
        if not result.success:
            print("Failed on combined")
            print(result.message)
        inferred_betas_no_sigma = result.x
        a = np.sqrt(np.sum(np.square(inferred_betas_no_sigma))/sigma2)
        inferred_betas_no_sigma_scaled = inferred_betas_no_sigma/a
        """
        print("true beta likelihood")
        print(calc_likelihood(z,x,betas))
        # print("inferred_beta_likelihood_no_sigma")
        # print(calc_likelihood(z,x,inferred_betas_no_sigma))
        # print("inferred_beta_likelihood_sigma_scaled")
        # print(calc_likelihood(z,x,inferred_betas_no_sigma_scaled))
        # print("inferred_beta_likelihood_em")
        # print(calc_likelihood(z,x,inferred_betas_em)) # this may fail if inferred betas are large
        print("inferred_beta_likelihood_em_scaled")
        print(calc_likelihood(z,x,inferred_betas_em_scaled))
        print("inferred_beta_likelihood_em_constrained")
        print(calc_likelihood(z,x,inferred_betas_em_cstr))

        print("rescale:")
        print(end_rescale - start_rescale)
        print("constrain:")
        print(end_cstr - start_cstr)

        pickle.dump((betas,
                     logit_betas,
                     inferred_betas_em,
                     inferred_betas_em_scaled,
                     inferred_betas_em_cstr), open(PICKLE_FILE, "wb"))
        plot_results(betas, logit_betas, inferred_betas_em, inferred_betas_em_scaled, inferred_betas_em_cstr)