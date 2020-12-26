import sys, os, pickle, argparse
import numpy as np
import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib import colors
from pprint import pprint
from scipy import integrate, optimize
from scipy.stats import norm, linregress
from sklearn.linear_model import LinearRegression, LogisticRegression
from .s1_utils import calc_h2_fast, probit_em_constrained

def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2

def generate_liability(num_indivs, ps, \
                       beta1, beta2, beta3, \
                       beta1_h2, beta2_h2, beta3_h2):
    num_snps = beta1.shape[0]
    num_genes = beta1.shape[1]
    genos = np.empty((num_indivs,num_snps))
    for i in range(num_snps):
        genos[:,i] = np.random.binomial(2, ps[i], size=num_indivs)
        # rescale genotypes to have mean 0 and std 1
        genos[:,i] = (genos[:,i] - 2*ps[i]) / np.sqrt(2*ps[i]*(1-ps[i]))

    exprs = np.empty((num_indivs,num_genes))
    expr_prs = np.dot(genos, beta1)
    for i in range(num_genes):
        exprs[:,i] = expr_prs[:,i] + \
                          np.random.normal(loc=0, scale=np.sqrt(1-beta1_h2), size=num_indivs)
    liability = np.dot(exprs, beta2) + np.dot(genos, beta3) + \
                     np.random.normal(loc=0, scale=np.sqrt(1-beta2_h2-beta3_h2), size=num_indivs)
    return genos, exprs, liability

def generate_case_control(num_indivs, ps,
                          beta1, beta2, beta3,
                          beta1_h2, beta2_h2, beta3_h2,
                          thresh, status="control"):
    assert status=="case" or status=="control"
    num_snps = beta1.shape[0]
    num_genes = beta1.shape[1]
    genos = np.empty((num_indivs, num_snps))
    exprs = np.empty((num_indivs, num_genes))
    batch = 10000
    num_counted = 0

    while num_counted < num_indivs:
        genos_batch, exprs_batch, liab_batch = generate_liability(batch,ps,beta1,beta2,beta3,
                                                                  beta1_h2,beta2_h2,beta3_h2)
        if status=="control":
            keep_idxs = np.where(liab_batch < thresh)[0]
        elif status=="case":
            keep_idxs = np.where(liab_batch >= thresh)[0]
        keep_num = min(len(keep_idxs), num_indivs - num_counted)
        keep_idxs = keep_idxs[:keep_num]
        genos_batch = genos_batch[keep_idxs,:]
        exprs_batch = exprs_batch[keep_idxs,:]
        genos[num_counted:num_counted+keep_num,:] = genos_batch
        exprs[num_counted:num_counted+keep_num,:] = exprs_batch
        num_counted += keep_num
        # print("generating %s: %s/%s" % (status, num_counted, num_indivs))
    return genos, exprs

def simulate(num_snps, num_genes, num_geno_samples, num_expr_samples, \
             beta1_h2, beta2_h2, beta3_h2, prevalence=0.01):
    thresh = norm.ppf(1-0.01)
    ps = np.array([0.5]*num_snps)

    beta1 = np.random.normal(loc=0.0, scale=1.0, size=(num_snps, num_genes))
    beta2 = np.random.normal(loc=0.0, scale=1.0, size=num_genes)
    beta3 = np.random.normal(loc=0.0, scale=1.0, size=num_snps)

    # re-scale effect sizes to set appropriate variance explained
    # NOTE: genotypes are scaled to have mean 0 and std 1, so ps_var should be 1
    # ps_var = 2.0*np.multiply(ps, 1-ps)
    ps_var = np.array([1]*num_snps)
    # beta1
    for i in range(num_genes):
        unscaled_varexp = np.sum(np.multiply(np.square(beta1[:,i]), ps_var))
        scale_factor = np.sqrt(unscaled_varexp / beta1_h2)
        beta1[:,i] /= scale_factor

    # beta2
    unscaled_varexp = np.sum(np.square(beta2))
    scale_factor = np.sqrt(unscaled_varexp / beta2_h2)
    beta2 /= scale_factor

    # beta3
    if beta3_h2 == 0:
        beta3 = np.zeros(num_snps)
    else:
        unscaled_varexp = np.sum(np.multiply(np.square(beta3), ps_var))
        scale_factor = np.sqrt(unscaled_varexp / beta3_h2)
        beta3 /= scale_factor

    # simulate all indivs
    genos_only_cont, exprs_only_cont = generate_case_control(int(num_geno_samples/2), ps,
                                                                 beta1, beta2, beta3,
                                                                 beta1_h2, beta2_h2, beta3_h2,
                                                                 thresh, status="control")
    genos_only_case, exprs_only_case = generate_case_control(int(num_geno_samples/2), ps,
                                                                 beta1, beta2, beta3,
                                                                 beta1_h2, beta2_h2, beta3_h2,
                                                                 thresh, status="case")
    genos_plus_cont, exprs_plus_cont = generate_case_control(int(num_expr_samples/2), ps,
                                                                 beta1, beta2, beta3,
                                                                 beta1_h2, beta2_h2, beta3_h2,
                                                                 thresh, status="control")
    genos_plus_case, exprs_plus_case = generate_case_control(int(num_expr_samples/2), ps,
                                                                 beta1, beta2, beta3,
                                                                 beta1_h2, beta2_h2, beta3_h2,
                                                                 thresh, status="case")
    genos_only = np.concatenate((genos_only_cont, genos_only_case), axis=0)
    phenos_only = np.array([0]*genos_only_cont.shape[0] + [1]*genos_only_case.shape[0])
    genos_plus = np.concatenate((genos_plus_cont, genos_plus_case), axis=0)
    exprs_plus = np.concatenate((exprs_plus_cont, exprs_plus_case), axis=0)
    phenos_plus = np.array([0]*exprs_plus_cont.shape[0] + [1]*exprs_plus_case.shape[0])
    return genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, (beta1, beta2, beta3)


#########################
### EM implementation ###
#########################
def calc_h2_quantitative(y,x):
    P = np.mean(y)
    corry = np.outer((y-P), (y-P)) / np.var(y)
    corrx = np.corrcoef(x, rowvar=True)
    cy_list = corry[np.triu_indices(corry.shape[0], k=1)]
    cx_list = corrx[np.triu_indices(corrx.shape[0], k=1)]
    reg = LinearRegression().fit(cx_list.reshape(-1,1), cy_list) # fit(X,y)
    slope = reg.coef_[0]
    return slope

def calc_h2(y, x, prev):
    num_indivs = len(y)
    P = np.sum(y) / num_indivs
    K = prev
    t = norm.ppf(1-K)

    Gijs = np.corrcoef(x, rowvar=True)
    Zijs = np.outer((y-P), (y-P)) / (P*(1-P))

    Gijs_list = Gijs[np.triu_indices(Gijs.shape[0], k=1)]
    Zijs_list = Zijs[np.triu_indices(Zijs.shape[0], k=1)]

    reg = LinearRegression().fit(Gijs_list.reshape(-1,1), Zijs_list) # fit(X,y)
    slope = reg.coef_[0]
    const = P*(1-P) / (K**2 * (1-K)**2) * norm.pdf(t)**2
    h2 = slope / const
    return h2

def calc_h2_subsample(y, x, prev):
    sig_ests = []
    for i in range(10):
        subsample_size = 5000
        idxs = np.random.choice(range(x.shape[0]), size=subsample_size, replace=False)
        sigma12_sq_est = calc_h2(y[idxs], x[idxs,:], prev)
        sig_ests.append(sigma12_sq_est)
    sigma12_sq = np.mean(sig_ests)
    return sigma12_sq

def calc_combined_em_likelihood(beta1, beta2, sigma1_sq, sigma12_sq, prev, y_obs, x_obs, z_hid, x_hid):
    # SNPs: X, Genes: Y, Trait: Z
    thresh = norm.ppf(1-prev)
    hid_scale = np.sqrt(2)

    # for now, beta2 is effect size for a single gene (so it's scalar)
    # beta1_var = np.sum(np.square(beta1))
    beta1_var = sigma1_sq

    # influence of beta1 on case/control gene liability for unobserved data
    """
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (hid_scale*np.sqrt(1-np.sum(np.square(beta1))
                                        * beta2**2))
    """
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (np.sqrt(2 - sigma12_sq))

    if any(np.isnan(term0)):
       # raise ValueError("NaN detected")
       return np.NINF

    Nobs = x_obs.shape[0]
    linearreg = -Nobs * np.log(1-beta1_var) - \
                1/(2*(1-beta1_var)**2) * np.sum(np.square(y_obs - np.dot(x_obs, beta1)))
    logreg = np.dot(z_hid, np.log(norm.cdf(term0))) + \
             np.dot(1-z_hid, np.log(1-norm.cdf(term0)))
    lhd = linearreg + logreg
    return lhd

def test_beta1_probit_em():
    num_snps = 100
    # num_genes = 50
    num_genes = 1
    num_geno_samples = 50000
    num_expr_samples = 500

    beta1_h2 = 0.1
    beta2_h2 = 0.05
    beta3_h2 = 0

    SIMULATED_GENOS = "probit_simul_genos.pickle"
    if os.path.exists(SIMULATED_GENOS):
        (genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas) = pickle.load(open(SIMULATED_GENOS, "rb"))
    else:
        genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas = \
             simulate(num_snps=num_snps, num_genes=num_genes,
             num_geno_samples=num_geno_samples, num_expr_samples=num_expr_samples,
             beta1_h2=beta1_h2, beta2_h2=beta2_h2, beta3_h2=beta3_h2)
        pickle.dump((genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas), open(SIMULATED_GENOS, "wb"))

    prev = 0.01
    thresh = norm.ppf(1-prev) # the liability distribution is standard normal
    Np = genos_only.shape[0]
    beta2 = true_betas[1]
    beta1 = true_betas[0]
    ps = np.array([0.5]*num_snps)


    # do this first for one gene
    clf = LinearRegression().fit(genos_plus, exprs_plus[:,0])


    fitted_params_em = neg_probit_beta1_em(prev, beta2, exprs_plus[:,0], genos_plus, phenos_only, genos_only)







    # for display
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    fig.suptitle("%s samples w/ observed genes, %s samples w/ SNPs only" % (num_expr_samples, num_geno_samples))
    # ax1.scatter(beta2, fitted_params, label="probit + observed, r-squared %.5f" % (rsquared(beta2,fitted_params)))
    ax1.scatter(beta1[:,0], clf.coef_, label="linear reg + observed, r-squared %.5f" % (rsquared(beta1[:,0],clf.coef_)))
    ax1.set_xlabel("True gene-trait effect")
    ax1.set_ylabel("Inferred gene-trait effect")
    ax1.legend()
    """
    ax2.scatter(beta1[:,0], fitted_params_combined, label="probit + observed + inferred, r-squared %.5f" % (rsquared(beta1[:,0],fitted_params_combined)))
    ax2.set_xlabel("True gene-trait effect")
    ax2.set_ylabel("Inferred gene-trait effect")
    """
    ax2.scatter(beta1[:,0], fitted_params_em, label="EM + observed + inferred, r-squared %.5f" % (rsquared(beta1[:,0],fitted_params_em)))
    ax2.set_xlabel("True gene-trait effect")
    ax2.set_ylabel("Inferred gene-trait effect")
    ax2.legend()

    for ax in [ax1, ax2]:
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
               ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)


    plt.show()
    sys.exit(0)

def test_beta2_accuracy(num_snps=100, num_genes=10, num_geno_samples=50000, num_expr_samples_list=[500], \
             beta1_h2=0.1, beta2_h2=0.05, num_trials=10):
    # fix a particular set of effect sizes
    # beta3 = 0
    prev = 0.01

    r2s_sample_size = []
    for num_expr_samples in num_expr_samples_list:
        r2s = []
        for nt in range(num_trials):
            print(num_expr_samples, nt)
            genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas = \
                simulate(num_snps, num_genes, num_geno_samples, num_expr_samples, \
                         beta1_h2, beta2_h2, beta3_h2=0, prevalence=prev)
            (beta1, beta2, _) = true_betas
            # Estimate beta1 from observed expr samples. Then estimate beta2 given beta1
            clf = LinearRegression().fit(genos_plus, exprs_plus)
            beta1_est = np.transpose(clf.coef_)
            imputed_genes = np.dot(genos_only, beta1_est)
            sigma2 = calc_h2_fast(phenos_only, imputed_genes, prev)
            beta2_est = probit_em_constrained(phenos_only, imputed_genes, sigma2, prev, beta_scale=0.01)
            r2s.append(rsquared(beta2, beta2_est))
        r2s_sample_size.append(r2s)
    return r2s_sample_size

def plot_beta2_accuracy_results(beta2_h2s):
    font = {'size': 16}
    mpl.rc('font', **font)

    fig, ax = plt.subplots()
    cm = colors.LinearSegmentedColormap.from_list("custom", ["orange", "blue"], N=100)
    cmap = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(vmin=min(beta2_h2s),
                                                           vmax=max(beta2_h2s)), cmap=cm)
    for beta2_h2 in beta2_h2s:
        results = pickle.load(open(BETA2_ACCURACY + "_%s.pickle" % (beta2_h2), "rb"))
        num_expr_samples_list = results["num_expr_samples_list"]
        beta2_h2 = results["beta2_h2"]
        trials = results["data"]
        means = [np.mean(x) for x in trials]
        stds = [np.std(x) for x in trials]
        plt.errorbar(num_expr_samples_list, means, yerr=stds, label=beta2_h2, capsize=5, c=cmap.to_rgba(beta2_h2))
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    plt.xlabel("Reference Panel Size")
    plt.ylabel(r"TWAS accuracy ($r^2$)")
    plt.legend(title=r"$\beta_2$ $h^2$")
    plt.tight_layout()
    plt.show()



if __name__=="__main__":

    # test_beta1_probit_em()

    np.seterr(all='raise') # catch runtimewarnings in probit constrained
    # NOTE: in em_test.py, changed initial beta std dev to 0.01 to control magnitude of Eqyi
    beta2_h2s = [0.1, 0.075, 0.05, 0.01]
    BETA2_ACCURACY="beta2_accuracy_results"
    if os.path.exists(BETA2_ACCURACY + "_%s.pickle" % (beta2_h2s[-1])):
        plot_beta2_accuracy_results(beta2_h2s)
    else:
        num_trials = 10
        num_snps = 100
        num_genes = 10
        num_geno_samples = 50000
        num_expr_samples_list = [100, 200, 300, 500, 1000]
        beta1_h2 = 0.1
        for beta2_h2 in beta2_h2s:
            r2s_sample_size = test_beta2_accuracy(num_snps=num_snps,
                                                  num_genes=num_genes,
                                                  num_geno_samples=num_geno_samples,
                                                  num_expr_samples_list=num_expr_samples_list,
                                                  beta1_h2=beta1_h2,
                                                  beta2_h2=beta2_h2,
                                                  num_trials=num_trials)
            pickle.dump({"data":r2s_sample_size,
                         "num_trials":num_trials,
                         "num_snps":num_snps,
                         "num_genes":num_genes,
                         "num_geno_samples":num_geno_samples,
                         "num_expr_samples_list":num_expr_samples_list,
                         "beta1_h2":beta1_h2,
                         "beta2_h2":beta2_h2
                        }, open(BETA2_ACCURACY + "_%s.pickle" % (beta2_h2), "wb"))
