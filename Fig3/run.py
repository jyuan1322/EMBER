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



#####################################################
### Second objective: given beta2, estimate beta1 ###
#####################################################

def neg_probit_likelihood(beta2, *args):
    c, z, x = args
    x = x - np.mean(x, axis=0)
    sigma = 1.0
    lhd = -1.0/(2*c) * np.dot(beta2, beta2) + \
          np.dot(z, np.log(norm.cdf(np.dot(x, beta2)/sigma))) + \
          np.dot(1-z, np.log(1-norm.cdf(np.dot(x, beta2)/sigma)))
    # minimize the negative likelihood to maximize
    lhd *= -1
    return lhd

def neg_probit_beta1_likelihood_combined(beta1, *args):
    # SNPs: X, Genes: Y, Trait: Z
    thresh, beta2, y_obs, x_obs, z_hid, x_hid, true_lhd = args
    hid_scale = np.sqrt(2)

    # for now, beta2 is effect size for a single gene (so it's scalar)
    """
    # basic linear regression w/ regularization
    lhd = -1.0/(2*c) * np.dot(beta1, beta1) - \
          1/(2*c*c) * np.sum(np.square(y_obs - np.dot(x_obs, beta1)))
    """
    beta1_var = np.sum(np.square(beta1))

    # influence of beta1 on case/control gene liability for unobserved data
    term0 = (np.dot(np.dot(x_hid, beta1), beta2) - thresh) / \
            (hid_scale*np.sqrt(1-np.sum(np.dot(np.square(beta1),
                                               np.square(beta2)))))

    if any(np.isnan(term0)):
       # raise ValueError("NaN detected")
       return np.NINF

    Nobs = x_obs.shape[0]
    linearreg = -Nobs * np.log(1-beta1_var) - \
                1/(2*(1-beta1_var)**2) * np.sum(np.square(y_obs - np.dot(x_obs, beta1)))
    logreg = np.dot(z_hid, np.log(norm.cdf(term0))) + \
             np.dot(1-z_hid, np.log(1-norm.cdf(term0)))
    lhd = linearreg + logreg

    """
    if true_lhd is not None:
        print(true_lhd, lhd, linearreg, logreg)
    else:
        print(lhd, linearreg, logreg)
    """
    # minimize the negative likelihood to maximize
    lhd *= -1
    return lhd

def test_beta1_probit():
    num_snps = 100
    # num_genes = 50
    num_genes = 1
    num_geno_samples = 50000
    num_expr_samples = 500

    SIMULATED_GENOS = "probit_simul_genos.pickle"
    if os.path.exists(SIMULATED_GENOS):
        (genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas) = pickle.load(open(SIMULATED_GENOS, "rb"))
    else:
        genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas = \
             simulate(num_snps=num_snps, num_genes=num_genes,
             num_geno_samples=num_geno_samples, num_expr_samples=num_expr_samples,
             beta1_h2=0.1, beta2_h2=0.05, beta3_h2=0)
        pickle.dump((genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas), open(SIMULATED_GENOS, "wb"))

    prev = 0.01
    thresh = norm.ppf(1-prev) # the liability distribution is standard normal
    Np = genos_only.shape[0]
    beta2 = true_betas[1]
    beta1 = true_betas[0]
    ps = np.array([0.5]*num_snps)


    # do this first for one gene
    clf = LinearRegression().fit(genos_plus, exprs_plus[:,0])

    # calculate true likelihood
    true_lhd = -1*neg_probit_beta1_likelihood_combined(beta1[:,0], thresh, beta2[0], exprs_plus[:,0], genos_plus, phenos_only, genos_only, None)

    initial_guess = np.array([0]*num_snps)
    # initial_guess = clf.coef_
    result = optimize.minimize(neg_probit_beta1_likelihood_combined, initial_guess, method = 'Nelder-Mead', options={'maxiter': 10000, 'adaptive':True}, args=(thresh, beta2[0], exprs_plus[:,0], genos_plus, phenos_only, genos_only, true_lhd))
    if not result.success:
        print("Failed on combined")
        print(result.message)
    fitted_params_combined = result.x

    # for display
    font = {'size': 16}
    mpl.rc('font', **font)
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    # fig.suptitle("%s samples w/ observed genes, %s samples w/ SNPs only" % (num_expr_samples, num_geno_samples))
    # ax1.scatter(beta2, fitted_params, label="probit + observed, r-squared %.5f" % (rsquared(beta2,fitted_params)))
    ax1.scatter(beta1[:,0], clf.coef_, label="linear reg + observed, r-squared %.5f" % (rsquared(beta1[:,0],clf.coef_)))
    ax1.set_xlabel(r"True SNP-gene effect ($\beta_1$)")
    ax1.set_ylabel(r"Inferred SNP-gene effect ($\beta_1$)")
    ax1.set_title(r"Linear Regression ($r^2=%0.2f$)" % (rsquared(beta1[:,0],clf.coef_)))
    # ax1.legend()
    ax2.scatter(beta1[:,0], fitted_params_combined, label="probit + observed + inferred, r-squared %.5f" % (rsquared(beta1[:,0],fitted_params_combined)))
    ax2.set_xlabel(r"True SNP-gene effect ($\beta_1$)")
    ax2.set_ylabel(r"Inferred SNP-gene effect ($\beta_1$)")
    ax2.set_title(r"EMBER ($r^2=%0.2f$)" % (rsquared(beta1[:,0],fitted_params_combined)))
    # ax2.legend()
    plt.tight_layout()
    plt.show()
    sys.exit(0)


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
    # hid_scale = np.sqrt(2)

    # for now, beta2 is effect size for a single gene (so it's scalar)
    # beta1_var = np.sum(np.square(beta1))
    beta1_var = sigma1_sq

    # influence of beta1 on case/control gene liability for unobserved data
    """
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (hid_scale*np.sqrt(1-np.sum(np.square(beta1))
                                        * beta2**2))
    """
    """
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (np.sqrt(2 - sigma12_sq))
    """
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (np.sqrt(2*(1 - sigma12_sq)))

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

def neg_probit_beta1_em(prev, beta2, y_obs, x_obs, z_hid, x_hid, verbose=False):
    # SNPs: X, Genes: Y, Trait: Z
    # sigma12_sq is the SNP-gene variance explained
    # sigma1_sq is the gene-trait variance explained of beta2

    # estimate sigma12_sq and sigma1_sq
    sigma1_sq = calc_h2_quantitative(y_obs,x_obs)
    # subsample to get an estimate of sigma12_sq


    ###
    ### Note: testing setting sigmas equal vs calculating custom sigma variance
    ###
    # sigma12_sq = calc_h2_subsample(z_hid, x_hid*beta2, prev)
    sigma12_sq = sigma1_sq

    c = 1 # beta prior
    stddev = np.sqrt(1-sigma12_sq)
    thresh = norm.ppf(1-prev)

    M = x_obs.shape[1]
    beta1 = np.random.normal(loc=0, scale=1, size=M)
    tol = 1e-8
    prev_lhd = float("-inf")
    lhds = []
    while True:
        # E-step over GWAS-only data
        mu_hid = np.dot(x_hid, beta1) * beta2
        alpha = (thresh - mu_hid)/stddev
        Eqyi = np.multiply(z_hid, (mu_hid + stddev * norm.pdf(alpha) / norm.cdf(-alpha))) + \
               np.multiply((1-z_hid), (mu_hid - stddev * norm.pdf(alpha) / norm.cdf(alpha)))

        # additional 2x factor from using x*beta instead of a real observed x
        """
        beta1 = np.dot(np.linalg.inv(1/c * np.eye(M) + \
                                     1/(2-sigma12_sq) * np.dot(x_hid.T, x_hid) * beta2**2 + \
                                     1/(1-sigma1_sq) * np.dot(x_obs.T, x_obs)),
                       1/(2-sigma12_sq) * np.dot(Eqyi, x_hid) * beta2 + \
                       1/(1-sigma1_sq) * np.dot(y_obs, x_obs))
        """
        beta1 = np.dot(np.linalg.inv(1 / c * np.eye(M) + \
                                     1 / (2*(1 - sigma12_sq)) * np.dot(x_hid.T, x_hid) * beta2 ** 2 + \
                                     1 / (1 - sigma1_sq) * np.dot(x_obs.T, x_obs)),
                       1 / (2*(1 - sigma12_sq)) * np.dot(Eqyi, x_hid) * beta2 + \
                       1 / (1 - sigma1_sq) * np.dot(y_obs, x_obs))

        lhd = calc_combined_em_likelihood(beta1, beta2, sigma1_sq, sigma12_sq,
                                          prev, y_obs, x_obs, z_hid, x_hid)
        lhds.append(lhd)
        if verbose:
            print(lhd)
        if np.abs(lhd - prev_lhd) < tol:
            break
        prev_lhd = lhd
    a = np.sqrt(np.sum(np.square(beta1))/sigma1_sq)
    beta1 = beta1/a

    if verbose:
        plt.plot(range(len(lhds)), lhds)
        plt.xlabel("iteration")
        plt.ylabel("log likelihood")
        plt.tight_layout()
        plt.show()
    return beta1

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

    # calculate true likelihood
    true_lhd = -1*neg_probit_beta1_likelihood_combined(beta1[:,0], thresh, beta2[0], exprs_plus[:,0], genos_plus, phenos_only, genos_only, None)


    fitted_params_em = neg_probit_beta1_em(prev, beta2, exprs_plus[:,0], genos_plus, phenos_only, genos_only)





    """
    initial_guess = np.array([0]*num_snps)
    # initial_guess = clf.coef_
    result = optimize.minimize(neg_probit_beta1_likelihood_combined, initial_guess, method = 'Nelder-Mead', options={'maxiter': 10000, 'adaptive':True}, args=(thresh, beta2[0], exprs_plus[:,0], genos_plus, phenos_only, genos_only, true_lhd))
    if not result.success:
        print("Failed on combined")
        print(result.message)
    fitted_params_combined = result.x
    """
    # for display
    fig, (ax1, ax2) = plt.subplots(1,2, figsize=(16,8))
    ax1.scatter(beta1[:,0], clf.coef_, c='k', label="linear reg + observed, r-squared %.5f" % (rsquared(beta1[:,0],clf.coef_)))
    ax1.set_xlabel(r"True SNP-gene effect ($\beta_1$)")
    ax1.set_ylabel(r"Inferred SNP-gene effect ($\beta_1$)")
    ax1.set_title(r"Linear Regression ($r^2=%0.2f$)" % (np.around(rsquared(beta1[:, 0], clf.coef_), decimals=2)))

    ax2.scatter(beta1[:,0], fitted_params_em, c='k', label="EM + observed + inferred, r-squared %.5f" % (rsquared(beta1[:,0],fitted_params_em)))
    ax2.set_xlabel(r"True SNP-gene effect ($\beta_1$)")
    ax2.set_ylabel(r"Inferred SNP-gene effect ($\beta_1$)")
    ax2.set_title(r"EMBER ($r^2=%0.2f$)" % (np.around(rsquared(beta1[:,0],fitted_params_em), decimals=2)))

    for ax in [ax1, ax2]:
        lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
                np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
               ]
        ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
        ax.set_aspect('equal')
        ax.set_xlim(lims)
        ax.set_ylim(lims)

    plt.tight_layout()
    plt.show()

    # for poster
    font = {'size': 16}
    mpl.rc('font', **font)
    fig, ax = plt.subplots()
    ax.scatter(beta1[:, 0], clf.coef_, c='b', s=10, alpha=0.8,
               label="linear reg + observed, r-squared %.5f" % (rsquared(beta1[:, 0], clf.coef_)))
    ax.scatter(beta1[:, 0], fitted_params_em, c='r', s=10, alpha=0.8,
               label="EM + observed + inferred, r-squared %.5f" % (rsquared(beta1[:, 0], fitted_params_em)))
    ax.set_xlabel(r"True SNP-gene effect ($\beta_1$)")
    ax.set_ylabel(r"Inferred SNP-gene effect ($\beta_1$)")
    plt.figtext(0.50, 0.94, r"Linear Reg. ($r^2=%0.2f$)" % \
        (np.around(rsquared(beta1[:, 0], clf.coef_), decimals=2)), color='b', ha ='right')
    plt.figtext(0.56, 0.94, r"EMBER ($r^2=%0.2f$)" % \
                (np.around(rsquared(beta1[:, 0], fitted_params_em), decimals=2)), color='r', ha='left')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
           ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    plt.locator_params(axis="x", nbins=8)
    plt.locator_params(axis="y", nbins=8)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()
    sys.exit(0)









def beta1_est_vs_num_beta2_genos():
    # https://stackoverflow.com/questions/13670333/multiple-variables-in-scipys-optimize-minimize
    """
        Estimate beta1 for 100 SNPs and 1 gene, when beta2 variance expl and cohort estimating beta2 are varied
    """
    num_trials = 10
    # num_trials = 2
    # num_probit_comb_reinits = 5
    # num_probit_comb_reinits = 0
    num_snps = 100
    num_genes = 1
    num_geno_samples = 50000
    num_expr_samples = 500

    # variance explained for a single gene
    # beta2_h2_list = [0.005, 0.01, 0.025, 0.05]
    beta2_h2_list = [0.05]

    # independent cohort to estimate beta2 effect size
    num_beta2_samples_list = [100, 500, 1000, 5000, 10000]

    rsq_results = []
    for num_beta2_samples in num_beta2_samples_list:
        for beta2_h2 in beta2_h2_list:
            real_beta2s = []
            inferred_beta2s = []
            linreg_results = []
            beta2_liability_results = []
            for nt in range(num_trials):

                genos_only, phenos_only, genos_plus, exprs_plus, phenos_plus, true_betas = \
                    simulate(num_snps=num_snps, num_genes=num_genes,
                    num_geno_samples=num_geno_samples, num_expr_samples=num_expr_samples,
                    beta1_h2=0.1, beta2_h2=beta2_h2, beta3_h2=0)

                thresh = norm.ppf(1-0.01) # the liability distribution is standard normal
                Np = genos_only.shape[0]
                beta2 = true_betas[1]
                real_beta2s.append(beta2)
                beta1 = true_betas[0]
                ps = np.array([0.5]*num_snps)

                # get an estimate of beta2 from an independent sample of data
                genos_beta2_cont, exprs_beta2_cont = generate_case_control(int(num_beta2_samples/2), ps,
                                                                           beta1=beta1, beta2=beta2, beta3=np.zeros(num_snps),
                                                                           beta1_h2=0.1, beta2_h2=beta2_h2, beta3_h2=0,
                                                                           thresh=thresh, status="control")
                genos_beta2_case, exprs_beta2_case = generate_case_control(int(num_beta2_samples/2), ps,
                                                                           beta1=beta1, beta2=beta2, beta3=np.zeros(num_snps),
                                                                           beta1_h2=0.1, beta2_h2=beta2_h2, beta3_h2=0,
                                                                           thresh=thresh, status="case")
                z_beta2 = np.array([0]*int(num_beta2_samples/2) + [1]*int(num_beta2_samples/2))
                x_beta2 = np.append(exprs_beta2_cont, exprs_beta2_case)
                x_beta2 = x_beta2.reshape(-1,1)

                initial_guess = [0.0]
                result = optimize.minimize(neg_probit_likelihood, initial_guess, method = 'Nelder-Mead',
                                           args=(thresh, z_beta2, x_beta2)) # options={'adaptive':True}
                if not result.success:
                    print("Failed on combined")
                    print(result.message)
                inferred_beta2 = result.x
                inferred_beta2s.append(inferred_beta2)
                print("real beta2:", beta2, "inferred beta2:", inferred_beta2)

                # plain linear regression on observed SNP-gene data
                clf = LinearRegression().fit(genos_plus, exprs_plus[:,0])
                linreg_results.append(rsquared(beta1[:,0], clf.coef_))


                # beta1 inference given inferred beta2
                reinits = []
                initial_guesses = []
                initial_guesses.append(np.array([0]*num_snps))
                initial_guesses.append(clf.coef_)
                # for ri in range(num_probit_comb_reinits):
                #     initial_guesses.append(np.random.normal(loc=0,scale=0.01,size=num_snps))


                for ig in initial_guesses:
                    print("num beta1 samples: %s, beta2 h2: %s, trial: %s, initial guess: %s" % (num_beta2_samples, beta2_h2, nt, ig))
                    result = optimize.minimize(neg_probit_beta1_likelihood_combined, ig, method = 'Nelder-Mead',
                                               options={'maxiter': 10000, 'adaptive':True},
                                               args=(thresh, inferred_beta2[0], exprs_plus[:,0],
                                                     genos_plus, phenos_only, genos_only, None))
                    if not result.success:
                        print("Failed on combined")
                        print(result.message)
                    fitted_params_combined = result.x
                    reinits.append(rsquared(beta1[:,0], fitted_params_combined))


                beta2_liability_results.append(max(reinits))


            print(real_beta2s)
            print(inferred_beta2s)
            real_beta2s = np.array(real_beta2s).flatten()
            inferred_beta2s = np.array(inferred_beta2s).flatten()
            print(rsquared(real_beta2s, inferred_beta2s))

            beta2_rsqs = rsquared(real_beta2s, inferred_beta2s)
            rsq_results.append((num_beta2_samples, beta2_h2, beta2_rsqs, linreg_results, beta2_liability_results))
    return rsq_results

def plot_existing_results():
    """
    files = ["beta1_est_from_beta2_inferred_b2h2_0p05.pickle",
             "beta1_est_from_beta2_inferred_b2h2_0p025.pickle",
             "beta1_est_from_beta2_inferred_b2h2_0p01.pickle",
             "beta1_est_from_beta2_inferred_b2h2_0p005.pickle"]
    """
    files = ["beta1_est_from_beta2_inferred_b2h2_0p01_low_sample.pickle"]
    fig1, ax1 = plt.subplots(figsize=(8,6))
    fig2, ax2 = plt.subplots()

    rsq_results = []
    for fl in files:
        if os.path.exists(fl):
            rsq_result = pickle.load(open(fl, "rb"))
            rsq_results = rsq_results + rsq_result
    # separate plots by beta2_h2
    beta2_h2s = np.sort(list(set([x[1] for x in rsq_results])))
    colors = mpl.cm.ScalarMappable(norm=mpl.colors.Normalize(
                                        vmin=0.005, # vmin=min(beta2_h2s),
                                        vmax=0.05), # vmax=max(beta2_h2s)),
                                        cmap='summer_r')
    colors.set_array(beta2_h2s)
    for beta2_h2 in beta2_h2s:
        print(beta2_h2)
        print(colors.to_rgba(beta2_h2))
        rsq_select = [x for x in rsq_results if x[1]==beta2_h2]

        num_geno_beta2s = []
        beta2_rsqs_mean = []
        beta2_rsqs_std = []
        linreg_list_mean = []
        linreg_list_std = []
        liability_list_mean = []
        liability_list_std = []

        for tpl in rsq_select:
            num_geno_b2 = tpl[0]
            # beta2_h2 = tpl[1]
            beta2_rsq = tpl[2]
            linreg = tpl[3]
            liability = tpl[4]

            num_geno_beta2s.append(num_geno_b2)
            beta2_rsqs_mean.append(np.mean(beta2_rsq))
            beta2_rsqs_std.append(np.std(beta2_rsq))
            linreg_list_mean.append(np.mean(linreg))
            linreg_list_std.append(np.std(linreg))
            liability_list_mean.append(np.mean(liability))
            liability_list_std.append(np.std(liability))

        ax1.errorbar(num_geno_beta2s, linreg_list_mean, yerr=linreg_list_std, label="linreg h2=%s" % (beta2_h2), capsize=5, linestyle=':', c=colors.to_rgba(beta2_h2))
        ax1.errorbar(num_geno_beta2s, liability_list_mean, yerr=liability_list_std, label="liability h2=%s" % (beta2_h2), capsize=5, linestyle='-', c=colors.to_rgba(beta2_h2))



        ax2.errorbar(num_geno_beta2s, beta2_rsqs_mean, yerr=beta2_rsqs_std, label="h2=%s" % (beta2_h2), capsize=5, c=colors.to_rgba(beta2_h2))
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    ax1.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15),
               fancybox=False, shadow=False, ncol=len(beta2_h2s))
    ax1.set_xlabel("beta2 estimate sample size")
    ax1.set_ylabel("beta1 r-squared")
    ax2.legend()
    plt.show()


if __name__=="__main__":

    # test_beta1_probit()
    test_beta1_probit_em()

    # plot_existing_results()

    # usually, adding inferred genes here produces better r-squared for beta2
    # test_probit()

    # run test_probit() over a range of sample sizes for genos-only cohort
    BETA1_RES = "beta1_est_from_beta2_inferred.pickle"
    if os.path.exists(BETA1_RES):
        rsq_results = pickle.load(open(BETA1_RES, "rb"))
        num_samples_list = []
        probit_list_mean = []
        probit_list_std = []
        combined_list_mean = []
        combined_list_std = []
        for tpl in rsq_results:
            num_samples = tpl[0]
            probit = tpl[1]
            combined = tpl[2]

            num_samples_list.append(num_samples)
            probit_list_mean.append(np.mean(probit))
            probit_list_std.append(np.std(probit))
            combined_list_mean.append(np.mean(combined))
            combined_list_std.append(np.std(combined))
        plt.errorbar(num_samples_list, probit_list_mean, yerr=probit_list_std, label="probit",capsize=5)
        plt.errorbar(num_samples_list, combined_list_mean, yerr=combined_list_std, label="combined",capsize=5)
        plt.legend()
        plt.show()
    else:
        results = beta1_est_vs_num_beta2_genos()
        pickle.dump(results, open(BETA1_RES, "wb"))
    # test performance at varying num_genos_only

