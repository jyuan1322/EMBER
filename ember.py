import sys, pickle, os, argparse, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint
from scipy.stats import norm, linregress, binom
from sklearn.linear_model import LinearRegression, Ridge, ElasticNet
from sklearn.decomposition import PCA
from ember_utils import calc_h2_fast, calc_h2_quantitative
from process_gene_info import get_all_beta1_cmc

font = {'size': 16}
mpl.rc('font', **font)

# set seed in numpy
np.random.seed(0)

def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2 * np.sign(slope)

def calc_combined_em_likelihood(beta1, beta2, sigma1_sq, sigma12_sq, prev, y_obs, x_obs, z_hid, x_hid):
    # SNPs: X, Genes: Y, Trait: Z
    thresh = norm.ppf(1-prev)
    beta1_var = sigma1_sq

    # influence of beta1 on case/control gene liability for unobserved data
    term0 = (np.dot(x_hid, beta1) * beta2 - thresh) / \
            (np.sqrt(2*(1 - sigma12_sq)))

    if any(np.isnan(term0)):
       # raise ValueError("NaN detected")
       return np.NINF

    Nobs = x_obs.shape[0]
    linearreg = -Nobs * np.log(1-beta1_var) - \
                1/(2*(1-beta1_var)**2) * np.sum(np.square(y_obs - np.dot(x_obs, beta1)))
    logreg = np.dot(z_hid, np.log(norm.cdf(term0))) + \
             np.dot(1-z_hid, np.log(norm.cdf(-term0)))
    lhd = linearreg + logreg
    print("likelihoods: %s, %s" % (linearreg, logreg))
    return lhd

def neg_probit_beta1_em(prev, beta2, y_obs, x_obs, z_hid, x_hid, verbose=False):
    # SNPs: X, Genes: Y, Trait: Z
    sigma1_sq = calc_h2_quantitative(y_obs,x_obs)
    sigma12_sq = sigma1_sq

    c = 1e-4 # beta prior
    stddev = np.sqrt(1-sigma12_sq)
    thresh = norm.ppf(1-prev)

    M = x_obs.shape[1]
    beta1 = np.random.normal(loc=0, scale=0.1, size=M)
    tol = 1e-8
    prev_lhd = float("-inf")
    lhds = []
    while True:
        # E-step over GWAS-only data
        mu_hid = np.dot(x_hid, beta1) * beta2
        alpha = (thresh - mu_hid)/stddev
        print("-"*50)
        print(alpha)
        Eqyi = np.multiply(z_hid, (mu_hid + stddev * norm.pdf(alpha) / norm.cdf(-alpha))) + \
               np.multiply((1-z_hid), (mu_hid - stddev * norm.pdf(alpha) / norm.cdf(alpha)))

        # additional 2x factor from using x*beta instead of a real observed x
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
    print(beta1)
    print(sigma1_sq)
    a = np.sqrt(np.sum(np.square(beta1))/sigma1_sq)
    beta1 = beta1/a

    if verbose:
        plt.plot(range(len(lhds)), lhds)
        plt.xlabel("iteration")
        plt.ylabel("log likelihood")
        plt.tight_layout()
        plt.show()
    return beta1

def blup_calc_lhd(X, Z, y, betas, eu_y):
    scores = np.dot(X, betas) + np.dot(Z, eu_y)
    # indiv_probs = norm.pdf(y, loc=scores, scale=1)
    # return np.sum(np.log(indiv_probs))
    return np.sum(np.square(y-scores))
def linreg_blup(X,Z,y,c):
    num_fixed = X.shape[1]
    num_rand = Z.shape[1]
    betas = np.zeros(num_fixed)
    eu_y = np.zeros(num_rand)
    lhd = blup_calc_lhd(X, Z, y, betas, eu_y)
    XTX = np.dot(X.T, X)
    XTZ = np.dot(X.T, Z)
    ZZpG = np.dot(Z.T, Z) + 1/c * np.eye(Z.shape[1])
    blockmat_inv = np.linalg.inv(np.block([[XTX, XTZ], [XTZ.T, ZZpG]]))
    Mx = X.shape[1]
    XTy = np.dot(y, X)
    ZTy = np.dot(y, Z)
    bu = blockmat_inv.dot(np.concatenate((XTy, ZTy), axis=0))
    betas = bu[:Mx]
    eu_y = bu[Mx:]
    return eu_y
def linreg_blup_slower(X,Z,y, c=0):
    # c is the prior variance of u (diagonal)

    num_fixed = X.shape[1]
    num_rand = Z.shape[1]
    betas = np.zeros(num_fixed)
    eu_y = np.zeros(num_rand)
    Vinv = np.linalg.inv(c * np.dot(Z,Z.T) + np.eye(Z.shape[0]))
    invXTVinvX = np.linalg.inv((X.T).dot(Vinv).dot(X))
    XTVinv = (X.T).dot(Vinv)
    UTVinv = (Z.T).dot(Vinv)
    ZZpG = np.dot(Z.T, Z) + np.eye(Z.shape[1])
    betas = invXTVinvX.dot(XTVinv).dot(y)
    eu_y = c * UTVinv.dot(y-np.dot(X,betas))
    return eu_y

def bpflip(snp):
    pair = {"A":"T", "T":"A", "C":"G", "G":"C"}
    return pair[snp]

def get_phenos(directory):
    indivs_dict = {}
    for fname in os.listdir(directory):
        if fname.endswith(".sample"):
            with open(os.path.join(directory, fname), "r") as f:
                header = next(f).strip().split()
                # ['ID_1', 'ID_2', 'missing', 'father', 'mother', 'sex', 'plink_pheno']
                for line in f:
                    line = line.strip().split()
                    name = line[0] + '--' + line[1]
                    sex = line[5]
                    pheno = line[6]
                    indivs_dict[name] = {"sex":sex, "pheno":pheno}
    return indivs_dict

def get_ancestry(fname, indivs_dict):
    with open(fname, "r") as f:
        header = next(f).strip().split()
        # ['#FID', 'IID', 'PC1', 'PC2', 'PC3', 'PC4', 'PC5', 'PC6', 'PC7', 'PC8', 'PC9', 'PC10']
        for line in f:
            line = line.strip().split()
            name = line[0] + '--' + line[1]
            pcs = [float(x) for x in line[2:]]
            indivs_dict[name]["pcs"] = pcs

def extract_snps(genfile, samplefile, snps):
    sample_names = []
    with open(samplefile, "r") as sf:
        next(sf)
        next(sf)
        count = 0
        for line in sf:
            line = line.strip().split()
            sample_names.append(line[0] + '--' + line[1])

    # print("num samples: ", len(sample_names))
    genos = {}
    with open(genfile, "r") as hf:
        for line in hf:
            line = line.split()
            snp_name = line[1].split(":")
            if len(snp_name[0]) == 1:
                snp_name = snp_name[0] + ":" + snp_name[1]
            else:
                snp_name = snp_name[0]

            if snp_name in snps:
                # A1 A1, A1 A2, A2 A2
                allele1 = line[3]
                allele2 = line[4]

                valid_als = {'A','T','C','G'}
                if allele1 in valid_als and allele2 in valid_als:
                    data = line[5:]
                    num_inds = int(len(data) / 3)
                    geno = []

                    for i in range(num_inds):
                        probs = [float(x) for x in data[i * 3:i * 3 + 3]]
                        alcnt = np.argmax(probs)  # this counts the number of allele2

                        # account for opposite strand
                        if allele1 == snps[snp_name]["eff_al"] and allele2 == snps[snp_name]["alt_al"]:
                            alcnt = 2 - alcnt
                        elif allele1 == snps[snp_name]["alt_al"] and allele2 == snps[snp_name]["eff_al"]:
                            pass
                        elif bpflip(allele1) == snps[snp_name]["eff_al"] and bpflip(allele2) == snps[snp_name][
                            "alt_al"]:
                            alcnt = 2 - alcnt
                        elif bpflip(allele1) == snps[snp_name]["alt_al"] and bpflip(allele2) == snps[snp_name][
                            "eff_al"]:
                            pass
                        else:
                            print(snp_name)
                            print(allele1, snps[snp_name]["eff_al"])
                            print(allele2, snps[snp_name]["alt_al"])
                            sys.exit(0)
                        geno.append(alcnt)

                genos[snp_name] = geno
    return genos, sample_names

def extract_all_PGC_cohorts(snps_dict, geno_data):
    genos_all = None
    samples_all = None
    for genfile, samplefile in geno_data:
        genos, sample_names = extract_snps(genfile, samplefile, snps_dict)
        if len(genos) == 0:
            print("skipped: ", genfile, " no genos imputed")
            continue
        max_temp = max([len(genos[x]) for x in genos])
        if max_temp != len(sample_names): # check if haps and sample match, otherwise skip
            print("skipped: ", genfile, max_temp, len(sample_names))
            continue
        if genos_all is None:
            genos_all = genos
            samples_all = sample_names
        else:
            samples_all = samples_all + sample_names
            for gene in genos:
                if gene in genos_all:
                    genos_all[gene] = genos_all[gene] + genos[gene]
    # remove SNPs from dictionary which are missing from some cohorts
    max_count = max([len(genos_all[x]) for x in genos_all])
    pop_snps = []
    for snp in genos_all:
        # print(max_count, len(genos_all[snp]))
        if len(genos_all[snp]) < max_count:
            pop_snps.append(snp)
    for snp in pop_snps: # do this in a separate loop so dict doesn't change size during iteration
        genos_all.pop(snp)

    # for each gene, create genotype matrix and list of odds ratios
    num_indivs = len(samples_all)
    snp_names = sorted(genos_all.keys())
    num_snps = len(snp_names)
    snp_matrix = np.empty((num_snps, num_indivs))
    for i,snpn in enumerate(snp_names):
        snp_matrix[i,:] = genos_all[snpn]
    snp_matrix = snp_matrix.T
    return snp_matrix, snp_names, samples_all

def extract_exprs(filename):
    exprs_dict = {}
    with open(filename, 'r') as f:
        line = next(f)
        sample_names = line.strip().split('\t')[1:]
        for i in range(len(sample_names)):
            splt = sample_names[i].split('_')
            sample_names[i] = "0--%s_%s" % (splt[0],splt[-1])
        for line in f:
            line = line.strip().split('\t')
            gene_name = line[0]
            exprs = [float(x) for x in line[1:]]
            exprs_dict[gene_name] = dict(zip(sample_names, exprs))
    return exprs_dict

def create_gene_matrix(gene, sg_dict, gt_dict, cmc_exprs_dict, ref_genodata, gwas_genodata):
    if gene not in cmc_exprs_dict:
        print("%s not in CMC gene list" % (gene))
        return
    gene_pickle_file = "all_data_%s.pickle" % (gene)
    if not os.path.exists(gene_pickle_file):
        # try:
        # get list of SNPs to extract
        snps_dict = gt_dict[gene]
        print(gene)

        pgc_genos, pgc_snp_names, pgc_samples = extract_all_PGC_cohorts(snps_dict, gwas_genodata)
        cmc_genos, cmc_snp_names, cmc_samples = extract_all_PGC_cohorts(snps_dict, ref_genodata)

        pgclen = len(pgc_snp_names)
        cmclen = len(cmc_snp_names)
        common_snps = set(pgc_snp_names)
        if pgclen != cmclen:
            if pgclen > cmclen:
                unique_snps = list(set(pgc_snp_names) - set(cmc_snp_names))
                for usnp in unique_snps:
                    idx = pgc_snp_names.index(usnp)
                    pgc_snp_names.remove(usnp)
                    pgc_genos = np.delete(pgc_genos, idx, axis=1)
            else:
                unique_snps = list(set(cmc_snp_names) - set(pgc_snp_names))
                for usnp in unique_snps:
                    idx = cmc_snp_names.index(usnp)
                    cmc_snp_names.remove(usnp)
                    cmc_genos = np.delete(cmc_genos, idx, axis=1)
        # gene expression matrix
        cmc_genos_filtered = []
        cmc_exprs = []
        cmc_samples_filtered = []
        for i in range(len(cmc_samples)):
            indiv = cmc_samples[i]
            if indiv in cmc_exprs_dict[gene]:  # must match otherwise removed from genos
                cmc_genos_filtered.append(cmc_genos[i, :])
                cmc_exprs.append(cmc_exprs_dict[gene][indiv])
                cmc_samples_filtered.append(indiv)
        cmc_genos = np.array(cmc_genos_filtered).astype(np.float)
        cmc_exprs = np.array(cmc_exprs).astype(np.float)

        # vector of phenotypes
        pgc_phenos = np.array([indivs_dict[x]['pheno'] for x in pgc_samples]).astype(np.float)
        cmc_phenos = np.array([indivs_dict[x]['pheno'] for x in cmc_samples_filtered]).astype(np.float)
        # matrix of covariates
        sex_vals = [indivs_dict[x]['sex'] for x in pgc_samples]
        pgc_sex = np.array([1.5 if x == 0 else x for x in sex_vals]).astype(np.float)
        sex_vals = [indivs_dict[x]['sex'] for x in cmc_samples_filtered]
        cmc_sex = np.array([1.5 if x == 0 else x for x in sex_vals]).astype(np.float)
        pgc_pcs = np.array([indivs_dict[x]['pcs'] for x in pgc_samples]).astype(np.float)
        cmc_pcs = np.array([indivs_dict[x]['pcs'] for x in cmc_samples_filtered]).astype(np.float)
        pgc_covariates = np.concatenate((pgc_pcs, pgc_sex[:, None]), axis=1)
        cmc_covariates = np.concatenate((cmc_pcs, cmc_sex[:, None]), axis=1)

        # convert all data matrices to standard normal columns

        pgc_genos = (pgc_genos - np.mean(pgc_genos, axis=0)) / np.std(pgc_genos, axis=0)
        cmc_genos = (cmc_genos - np.mean(cmc_genos, axis=0)) / np.std(cmc_genos, axis=0)
        pgc_covariates = (pgc_covariates - np.mean(pgc_covariates, axis=0)) / np.std(pgc_covariates, axis=0)
        cmc_covariates = (cmc_covariates - np.mean(cmc_covariates, axis=0)) / np.std(cmc_covariates, axis=0)
        cmc_exprs = (cmc_exprs - np.mean(cmc_exprs)) / np.std(cmc_exprs)
        # pgc_phenos = (pgc_phenos - np.mean(pgc_phenos)) / np.std(pgc_phenos)
        # cmc_phenos = (cmc_phenos - np.mean(cmc_phenos)) / np.std(cmc_phenos)
        pgc_phenos = pgc_phenos - 1
        cmc_phenos = cmc_phenos - 1  # from fam files, 1=control, 2=case

        # estimate variance explained by all covariates and snps

        # run beta1 re-estimation
        prev = 0.01
        beta2s = sg_dict[gene]['alpha']
        beta2 = np.mean(beta2s)  # for now, just take the mean of all tissues
        genos_plus = np.concatenate((cmc_genos, cmc_covariates), axis=1)
        genos_only = np.concatenate((pgc_genos, pgc_covariates), axis=1)

        # save to pickle first
        all_data = (beta2, cmc_exprs, genos_plus, genos_only, cmc_phenos, pgc_phenos, pgc_snp_names)
        pickle.dump(all_data, open(gene_pickle_file, "wb"))
        # except Exception as e:
        #     print(e)

def plot_mediated_varexp(gene_varexp, gene_list):
    # plot variance explained
    snp_h2s = []
    med_h2s = []
    gene_varexp_labels = []
    # for gene in gene_varexp:
    for gene in gene_list:
        snp_h2s.append(gene_varexp[gene]["total_snp_h2"])
        # med_h2s.append(gene_varexp[gene]["mediated_h2"])
        med_h2s.append(gene_varexp[gene]["inf_mediated_h2"])
        gene_varexp_labels.append("*" + gene.strip("ENSG").lstrip("0"))
    fig, ax = plt.subplots()
    ax.scatter(snp_h2s, med_h2s)
    ax.set_xlabel("total SNP varexp")
    ax.set_ylabel("total gene-mediated varexp")
    for i, txt in enumerate(gene_varexp_labels):
        ax.annotate(txt, (snp_h2s[i], med_h2s[i]))
    plt.tight_layout()
    plt.savefig("snp_mediated_varexp.png")
    plt.close(fig)

def table_performance_by_varexp(gene_varexp, gene_list, sg_dict, xaxis="inf_mediated_h2"):
    # sort table by decreasing fraction of mediated variance explained
    table_list = []
    for gene in gene_list:
        frac_med_varexp = gene_varexp[gene][xaxis] / \
                                gene_varexp[gene]["total_snp_h2"]
        linreg_r2 = gene_varexp[gene]["linreg_r2"]
        inferred_r2 = gene_varexp[gene]["inferred_r2"]
        delta_r2 = inferred_r2 - linreg_r2
        gene_name = sg_dict[gene]["name"]
        table_list.append((gene_name, inferred_r2, linreg_r2, delta_r2, frac_med_varexp))
    table_list = sorted(table_list, key=lambda x: x[-2], reverse=True)

    print(r"\begin{tabular}{|l|c|c|c|c|}")
    print("\hline")
    print(r"Gene & EMBER/GTEx $r^2$ & LR/GTEx $r^2$ & $\Delta r^2$ & frac mediated $h^2$\\")
    print("\hline")
    for table_entry in table_list:
        print("%s & %0.3f & %0.3f & %0.3f & %0.3f \\\\" % table_entry)
    print("\hline")
    print("\end{tabular}")

def plot_performance_by_varexp(gene_varexp, gene_list, xaxis="inf_mediated_h2"):
    linreg_r2s = []
    inferred_r2s = []
    frac_med_varexps = []
    gene_labels = []
    # for gene in gene_varexp:
    for gene in gene_list:
        frac_med_varexp = gene_varexp[gene][xaxis] / \
                          gene_varexp[gene]["total_snp_h2"]
        # if frac_med_varexp > 0.1:
        if frac_med_varexp > -10:
            frac_med_varexps.append(frac_med_varexp)
            linreg_r2s.append(gene_varexp[gene]["linreg_r2"])
            inferred_r2s.append(gene_varexp[gene]["inferred_r2"])
            gene_labels.append("*" + gene.strip("ENSG").lstrip("0"))
    fig, ax = plt.subplots()
    cm = plt.cm.get_cmap('cool')
    sc = plt.scatter(linreg_r2s, inferred_r2s, c=frac_med_varexps,
                     vmin=min(0,min(frac_med_varexps)), vmax=max(frac_med_varexps), cmap=cm)
    plt.colorbar(sc)

    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel(r"linear reg. vs GTEx $r^2$")
    plt.ylabel(r"EMBER vs GTEx $r^2$")
    plt.tight_layout()
    plt.savefig("performance_by_varexp.png")
    plt.close(fig)

    # plot improvement by EMBER vs fraction mediated variance explained
    impvmt = np.array(inferred_r2s) - np.array(linreg_r2s)
    slope, intercept, r_value, p_value, std_err = linregress(frac_med_varexps, impvmt)
    rsq = rsquared(frac_med_varexps, impvmt)
    fig, ax = plt.subplots()
    plt.scatter(frac_med_varexps, impvmt, label="p-val=%0.4f, r2=%0.4f" % (p_value, rsq))
    plt.plot(frac_med_varexps, np.array(frac_med_varexps)*slope + intercept, c='k')
    plt.legend()
    plt.xlabel("fraction gene-mediated variance explained")
    plt.ylabel("EMBER r2 - linreg r2")
    for i, txt in enumerate(gene_labels):
        ax.annotate(txt, (frac_med_varexps[i], impvmt[i]))
    plt.tight_layout()
    plt.savefig("performance_by_varexp_improvement.png")
    plt.close(fig)

    # boxplot
    fig, ax = plt.subplots()
    plt.hist(impvmt)
    ax.set_xlabel(r"Improvement in $r^2$")
    ax.set_ylabel("Frequency")
    ax.get_xaxis().tick_bottom()
    ax.get_yaxis().tick_left()
    plt.tight_layout()
    plt.savefig("performance_histogram.png")
    plt.close(fig)

def plot_performance_blup(gene_varexp, gene_list):
    linreg_r2s = []
    inferred_r2s = []
    blup_r2s = []
    comp_blup_r2s = []
    frac_med_varexps = []

    for gene in gene_list:
        frac_med_varexp = gene_varexp[gene]["inf_mediated_h2"] / \
                          gene_varexp[gene]["total_snp_h2"]

        frac_med_varexps.append(frac_med_varexp)
        linreg_r2s.append(gene_varexp[gene]["linreg_r2"])
        inferred_r2s.append(gene_varexp[gene]["inferred_r2"])
        blup_r2s.append(gene_varexp[gene]["blup_r2"])
        if "comp_blup_r2" in gene_varexp[gene]:
            comp_blup_r2s.append(gene_varexp[gene]["comp_blup_r2"])
    fig, ax = plt.subplots()
    idxs = np.argsort(frac_med_varexps)
    ax.plot(np.array(frac_med_varexps)[idxs], np.array(blup_r2s)[idxs], label="BLUP")
    ax.plot(np.array(frac_med_varexps)[idxs], np.array(linreg_r2s)[idxs], label="LinReg + VR")
    ax.plot(np.array(frac_med_varexps)[idxs], np.array(inferred_r2s)[idxs], label="EMBER + VR")

    plt.xlabel(r"Frac. Mediated Var. Expl.")
    plt.ylabel(r"concordance vs TWAS/GTEx $r^2$")
    plt.legend()
    plt.tight_layout()
    plt.savefig("performance_by_varexp_blup.png")
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.scatter(blup_r2s, linreg_r2s, c='k')
    ax.scatter(blup_r2s, inferred_r2s, c='limegreen')
    # ax.scatter(comp_blup_r2s, linreg_r2s, c='k')
    # ax.scatter(comp_blup_r2s, inferred_r2s, c='limegreen')
    lims = [np.min([ax.get_xlim(), ax.get_ylim()]),  # min of both axes
            np.max([ax.get_xlim(), ax.get_ylim()]),  # max of both axes
            ]
    ax.plot(lims, lims, 'k-', alpha=0.75, zorder=0)
    ax.set_aspect('equal')
    ax.set_xlim(lims)
    ax.set_ylim(lims)

    plt.xlabel(r"BLUP $r^2$")
    # plt.ylabel(r"LinReg + VR $r^2$")
    plt.text(-0.15, 0.31, 'EMBER',
             verticalalignment='top', horizontalalignment='center',
             transform=ax.transAxes, rotation=90,
             color='limegreen', fontsize=18)
    plt.text(-0.15, 0.32, r'& LinReg + VR $r^2$',
             verticalalignment='bottom', horizontalalignment='center',
             transform=ax.transAxes, rotation=90,
             color='k', fontsize=18)
    plt.tight_layout()
    plt.savefig("blup_vs_linreg_baseline.png")
    plt.close(fig)

    print("mean/std performance of LinReg+VR: %s +/- %s" % (np.mean(linreg_r2s), np.std(linreg_r2s)))
    print("mean/std performance of EMBER+VR: %s +/- %s" % (np.mean(inferred_r2s), np.std(inferred_r2s)))
    print("mean/std performance of BLUP: %s +/- %s" % (np.mean(blup_r2s), np.std(blup_r2s)))
    print("mean/std performance of BLUP (TWAS/Fusion): %s +\- %s" % (np.mean(comp_blup_r2s), np.std(comp_blup_r2s)))
    # sign test
    num_tot = len(inferred_r2s)
    num_pos = np.sum(np.array(inferred_r2s) > np.array(linreg_r2s))
    print("k=%s, n=%s, sign test: %s" % (num_pos, num_tot, binom.sf(k=num_pos, n=num_tot, p=0.5))) # survival function, 1-cdf


def plot_effects_by_pos(gene_name, snp_pos, true_blups_gtex, linreg_effs, ember_effs, blup_effs, snp_pos_comp=None,
                        true_blups_cmc=None):
    twascmc_color = "k"
    blup_color = "dodgerblue"
    ember_color = "limegreen"
    # compare locations to effect sizes
    fig, (ax1, ax2, ax4) = plt.subplots(3, sharex=True, figsize=(16, 12))
    ax1.text(0.025, 1.025, gene_name,
             verticalalignment='bottom', horizontalalignment='left',
             transform=ax1.transAxes,
             color='k', fontsize=20)
    ax1.scatter(snp_pos, true_blups_gtex, label="true BLUPs", c='k')
    ax1.set_title("TWAS/GTEx reported effect", fontsize=18)
    ax1.tick_params(axis='y', labelcolor='k', labelsize=16)
    ax1.set_ylabel(r"Effect size $(\beta_1)$", fontsize=16)

    ax2.scatter(snp_pos, blup_effs, label="BLUP", c=blup_color)
    ax2.tick_params(axis='y', labelcolor='k', labelsize=16)
    ax2.set_ylabel(r"Effect size $(\beta_1)$", fontsize=16, c=blup_color)
    if snp_pos_comp is not None and true_blups_cmc is not None:
        ax2.text(0.39, 1.025, 'BLUP',
                 verticalalignment='bottom', horizontalalignment='right',
                 transform=ax2.transAxes,
                 color=blup_color, fontsize=18)
        ax2.text(0.4, 1.025, '&',
                 verticalalignment='bottom', horizontalalignment='center',
                 transform=ax2.transAxes,
                 color='k', fontsize=18)
        ax2.text(0.41, 1.025, 'TWAS/CMC reported effect',
                 verticalalignment='bottom', horizontalalignment='left',
                 transform=ax2.transAxes,
                 color=twascmc_color, fontsize=18)
        ax3 = ax2.twinx()
        ax3.scatter(snp_pos_comp, true_blups_cmc, label="LR no PCA", c=twascmc_color, zorder=1)
        ax3.set_ylabel(r"Effect size $(\beta_1)$", c=twascmc_color, fontsize=16)
        # rescale the axes so the points overlap
        min2, max2 = np.amin(blup_effs), np.amax(blup_effs)
        min3, max3 = np.amin(true_blups_cmc), np.amax(true_blups_cmc)
        mean2, std2 = np.mean(blup_effs), np.std(blup_effs)
        mean3, std3 = np.mean(true_blups_cmc), np.std(true_blups_cmc)
        stddiff = np.amax(
            np.abs([(min2 - mean2) / std2, (max2 - mean2) / std2, (min3 - mean3) / std3, (max3 - mean3) / std3]))
        ax2.set_ylim([mean2 - 1.05 * stddiff * std2, mean2 + 1.05 * stddiff * std2])
        ax3.set_ylim([mean3 - 1.05 * stddiff * std3, mean3 + 1.05 * stddiff * std3])
        # rearrange z-order across axes
        ax2.set_zorder(2)
        ax3.set_zorder(1)
        ax2.patch.set_visible(False)
    else:
        ax2.text(0.5, 1.025, 'BLUP',
                 verticalalignment='bottom', horizontalalignment='center',
                 transform=ax2.transAxes,
                 color=blup_color, fontsize=18)

    ax4.scatter(snp_pos, ember_effs, label="EMBER & Variable Selection", c=ember_color)
    ax4.tick_params(axis='y', labelcolor='k', labelsize=16)
    ax4.tick_params(axis='x', labelcolor='k', labelsize=16)
    ax4.set_ylabel(r"Effect size $(\beta_1)$", c=ember_color, fontsize=16)
    ax4.set_xlabel("SNP Position (BP)", fontsize=16)
    ax4.text(0.39, 1.025, 'EMBER',
             verticalalignment='bottom', horizontalalignment='right',
             transform=ax4.transAxes,
             color=ember_color, fontsize=18)
    ax4.text(0.4, 1.025, '&',
             verticalalignment='bottom', horizontalalignment='center',
             transform=ax4.transAxes,
             color='k', fontsize=18)
    ax4.text(0.41, 1.025, 'LinReg with Variable Reduction',
             verticalalignment='bottom', horizontalalignment='left',
             transform=ax4.transAxes,
             color='k', fontsize=18)
    ax5 = ax4.twinx()
    ax5.scatter(snp_pos, linreg_effs, label="LinReg & Variable Selection", c='k')
    ax5.set_ylabel(r"Effect size $(\beta_1)$", c='k', fontsize=16)

    # rescale the axes so the points overlap
    min5, max5 = np.amin(linreg_effs), np.amax(linreg_effs)
    min4, max4 = np.amin(ember_effs), np.amax(ember_effs)
    mean5, std5 = np.mean(linreg_effs), np.std(linreg_effs)
    mean4, std4 = np.mean(ember_effs), np.std(ember_effs)
    stddiff = np.amax(
        np.abs([(min4 - mean4) / std4, (max4 - mean4) / std4, (min5 - mean5) / std5, (max5 - mean5) / std5]))

    ax4.set_ylim([mean4 - 1.05 * stddiff * std4, mean4 + 1.05 * stddiff * std4])
    ax5.set_ylim([mean5 - 1.05 * stddiff * std5, mean5 + 1.05 * stddiff * std5])

    # rearrange z-order across axes
    ax4.set_zorder(2)
    ax5.set_zorder(1)
    ax4.patch.set_visible(False)

    plt.tight_layout()
    plt.savefig("blup_vs_pcalinreg_locus_%s.png" % (gene_name))
    plt.close(fig)


def plot_effects(gene_name, true_blups_gtex, linreg_effs, ember_effs, blup_effs, true_blups_gtex_comp, true_blups_cmc):
    # enet_color = "dodgerblue"
    twascmc_color = "k"
    blup_color = "dodgerblue"
    ember_color = "limegreen"
    # plot BLUP scatterplot for single gene
    # fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(24, 8))
    fig, (ax1, ax3) = plt.subplots(nrows=1, ncols=2, figsize=(16, 8))
    line1 = ax1.scatter(true_blups_gtex, linreg_effs, c='k',
                        label=r'LinReg & Variable Reduction, $r^2$=%.5f' % \
                              (rsquared(true_blups_gtex, linreg_effs)))
    m, b = np.polyfit(true_blups_gtex, linreg_effs, 1)
    ax1.plot(true_blups_gtex, m * true_blups_gtex + b, c='k')
    ax1.set_xlabel(r"TWAS/GTEx reported effect $(\beta_1)$")
    ax1.set_ylabel(r"Inferred effect size $(\beta_1)$")

    ax2 = ax1.twinx()
    line2 = ax2.scatter(true_blups_gtex, ember_effs, c=ember_color,
                        label=r'EMBER & Variable Selection, $r^2$=%.5f' % \
                              (rsquared(true_blups_gtex, ember_effs)))
    m, b = np.polyfit(true_blups_gtex, ember_effs, 1)
    ax2.plot(true_blups_gtex, m * true_blups_gtex + b, c=ember_color)
    ax2.tick_params(axis='y', labelcolor=ember_color, labelsize=16)
    # ax2.set_ylabel(r"Inferred effect size $(\beta_1)$", c=ember_color, fontsize=16)
    box = ax1.get_position()
    ax1.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    ax1.legend(handles=[line1, line2], labels=[line1.get_label(), line2.get_label()],
               loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, frameon=False)

    # if gene_name in expr_dict_cmc:
    if true_blups_gtex_comp is not None and true_blups_cmc is not None:
        line4 = ax3.scatter(true_blups_gtex_comp, true_blups_cmc, c=twascmc_color,
                            label=r'TWAS/CMC reported effect, $r^2$=%0.5f' % (
                                rsquared(true_blups_gtex_comp, true_blups_cmc)))
        m, b = np.polyfit(true_blups_gtex_comp, true_blups_cmc, 1)
        ax3.plot(true_blups_gtex_comp, m * np.array(true_blups_gtex_comp) + b, c=twascmc_color)
    line3 = ax3.scatter(true_blups_gtex, blup_effs, c=blup_color,
                        label=r'BLUP (MME), $r^2$=%0.5f' % (rsquared(true_blups_gtex, blup_effs)))
    m, b = np.polyfit(true_blups_gtex, blup_effs, 1)
    ax3.plot(true_blups_gtex, m * true_blups_gtex + b, c=blup_color)
    ax3.set_xlabel(r"TWAS/GTEx reported effect $(\beta_1)$")
    box = ax3.get_position()
    ax3.set_position([box.x0, box.y0 + box.height * 0.1,
                      box.width, box.height * 0.9])
    # if gene_name in expr_dict_cmc:
    if true_blups_gtex_comp is not None and true_blups_cmc is not None:
        ax3.legend(handles=[line3, line4], labels=[line3.get_label(), line4.get_label()],
                   loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, frameon=False)
    else:
        ax3.legend(handles=[line3], labels=[line3.get_label()],
                   loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, frameon=False)
    # lines = [line1, line2]
    # labels = [x.get_label() for x in lines]
    fig.suptitle(gene_name, fontsize=20)
    fig.tight_layout()
    fig.subplots_adjust(top=0.8, bottom=0.1)  # make room for header
    plt.savefig("scatterplot_rsquared_%s.png" % (gene_name))
    plt.close(fig)

if __name__=="__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-dir", dest="sample_dir", required=True,
                        help="Directory of .sample files containing phenotype labels")
    parser.add_argument("--ancestry-dir", dest="ancestry_dir", required=True,
                        help=".eigenvec file storing ancestry PCs")
    parser.add_argument("--gene-list", dest="gene_list", required=True,
                        help="list of genes, one gene name per line")
    parser.add_argument("--expr-data", dest="expr_data", required=True,
                        help="tsv file containing gene expression measurements")
    parser.add_argument("--ref-genos", dest="ref_genos_files", required=True,
                        help="tsv file listing reference genotype files. Each line contains two entries: "
                             "an Oxford .gen filepath and a .sample filepath")
    parser.add_argument("--gwas-genos", dest="gwas_genos_files", required=True,
                        help="tsv file listing GWAS genotype files. Each line contains two entries: "
                             "an Oxford .gen filepath and a .sample filepath")
    args = parser.parse_args()

    SAMPLEDIR = args.sample_dir
    ANCESTRYDIR = args.ancestry_dir
    indivs_dict = get_phenos(SAMPLEDIR)
    get_ancestry(ANCESTRYDIR, indivs_dict)

    sg_dict, gt_dict = pickle.load(open("ember_all_betas.pickle","rb"))

    # get list of genes
    GENEFILE=args.gene_list
    genes = []
    with open(GENEFILE, "r") as f:
        for line in f:
            genes.append(line.strip())

    # get gene expression matrix
    EXPRFILE=args.expr_data
    cmc_exprs_dict = extract_exprs(EXPRFILE)
    genes_filtered = []
    for gene in genes:
        if gene in cmc_exprs_dict:
            genes_filtered.append(gene)
    genes = genes_filtered


    # for each gene, extract gene expression from CMC and SNPs from CMC+PGC
    ref_genos = []
    gwas_genos = []
    with open(args.ref_genos_files, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            ref_genos.append((line[0], line[1]))
    with open(args.gwas_genos_files, "r") as f:
        for line in f:
            line = line.strip().split("\t")
            gwas_genos.append((line[0], line[1]))
    for gene in genes:
        print("processing %s" % (gene))
        create_gene_matrix(gene, sg_dict, gt_dict, cmc_exprs_dict, ref_genos, gwas_genos)

    gene_list = []
    for pfile in os.listdir("."):
        if pfile.startswith("all_data_") and pfile.endswith(".pickle"):
            print(pfile.strip("all_data_").strip(".pickle"))
            gene_list.append(pfile.strip("all_data_").strip(".pickle"))
    """
    gene_list = [
        "ENSG00000016864",
        "ENSG00000049167",
        "ENSG00000092020",
        "ENSG00000102882",
        "ENSG00000111850", # not in TWAS/CMC reference
        "ENSG00000112763",
        "ENSG00000120451",
        "ENSG00000124508",
        "ENSG00000124613",
        # "ENSG00000126464", # sigma1_sq is negative
        "ENSG00000132600",
        # "ENSG00000137331", # sigma1_sq is negative
        "ENSG00000146830",
        "ENSG00000152219", # not in TWAS/CMC reference
        "ENSG00000159231",
        "ENSG00000163938",
        "ENSG00000164088",
        "ENSG00000164241",
        "ENSG00000167264", # not in TWAS/CMC reference
        "ENSG00000170802",
        "ENSG00000172687",
        "ENSG00000172922",
        "ENSG00000173295", # not in TWAS/CMC reference
        "ENSG00000175662",
        "ENSG00000177082", # not in TWAS/CMC reference
        "ENSG00000186470",
        "ENSG00000187987",
        "ENSG00000189298",
        "ENSG00000196236",
        "ENSG00000198951",
        "ENSG00000203875",
        "ENSG00000213523",
        "ENSG00000214435",
        "ENSG00000226314", # not in TWAS/CMC reference
        "ENSG00000234745", # not in TWAS/CMC reference
        "ENSG00000235109", # not in TWAS/CMC reference
        "ENSG00000238083", # not in TWAS/CMC reference
        # "ENSG00000241106", # Not in CMC
        # "ENSG00000256053", # Not in CMC
        "ENSG00000262246" # removed column of NANs in genos_plus
    ]
    """

    # load variance explained values - this takes a long time to calculate
    frac_varexp_file = "fraction_variance_explained.txt"
    gene_varexp = {}
    if os.path.exists(frac_varexp_file):
        with open(frac_varexp_file, "r") as f:
            for line in f:
                line = line.strip().split()
                gene_varexp[line[0]] = {
                    "total_snp_h2":float(line[1]),
                    "snp_expr_h2":float(line[2]),
                    "mean_alpha":float(line[3]),
                    "mediated_h2":float(line[4])
                }

    # load TWAS/CMC gene properties
    expr_dict_cmc = get_all_beta1_cmc()

    for gene in gene_list:
        (beta2, cmc_exprs, genos_plus, genos_only, cmc_phenos, pgc_phenos, pgc_snp_names) = \
            pickle.load(open("all_data_%s.pickle" % (gene), "rb"))
        prev = 0.01

        # filter out nan columns
        nan_xs, nan_ys_plus = np.where(np.isnan(genos_plus))
        nan_xs, nan_ys_only = np.where(np.isnan(genos_only))
        unique_nan_ys = list(set(nan_ys_plus).union(set(nan_ys_only)))
        genos_plus = np.delete(genos_plus, unique_nan_ys, axis=1)
        genos_only = np.delete(genos_only, unique_nan_ys, axis=1)
        for idx in unique_nan_ys:
            if idx < len(pgc_snp_names): # make sure we're deleting a SNP and not a covariate
                pgc_snp_names.pop(idx)
        print("removed nan: %s" % (unique_nan_ys))

        # test for variance explained
        # This is saved in the file "fraction_variance_explained.txt"
        if not os.path.exists(frac_varexp_file):
            num_snps = len(pgc_snp_names)
            try:
                total_snp_h2 = calc_h2_fast(y=pgc_phenos, x=genos_only[:,:num_snps], prev=prev)
                snp_expr_h2 = calc_h2_quantitative(y=cmc_exprs, x=genos_plus[:,:num_snps])
                mean_alpha = sg_dict[gene]['mean_alpha']
                print(gene, total_snp_h2, snp_expr_h2, mean_alpha, mean_alpha**2 * snp_expr_h2)
                gene_varexp[line[0]] = {
                    "total_snp_h2": total_snp_h2,
                    "snp_expr_h2": snp_expr_h2,
                    "mean_alpha": mean_alpha,
                    "mediated_h2": mean_alpha**2 * snp_expr_h2
                }
            except Exception as e:
                print(gene, e)

        # to resolve LD among cis-SNPs, perform PCA over the input parameters
        num_comps = 10

        # linear regression
        # to resolve multicollinearity, apply PCA first, then regress on PCs
        num_snps = len(pgc_snp_names)
        pca = PCA(n_components=num_comps)
        genos_pca = pca.fit_transform(genos_plus)
        beta1s_pcalr = LinearRegression().fit(genos_pca, cmc_exprs).coef_
        beta1s_linreg = np.dot(pca.components_.T, beta1s_pcalr)
        beta1s_linreg = beta1s_linreg[:len(pgc_snp_names)]

        # calculate TWAS alpha
        exprs_inferred = genos_only[:, :num_snps].dot(beta1s_linreg)
        alpha_inf1 = LinearRegression().fit(exprs_inferred.reshape(-1,1), pgc_phenos).coef_[0]
        # alpha_inf2 = np.cov(np.vstack((exprs_inferred, pgc_phenos)))[0,1] / np.var(exprs_inferred)

        # update gene_varexp
        gene_varexp[gene]["inferred_alpha"] = alpha_inf1
        gene_varexp[gene]["inf_mediated_h2"] = gene_varexp[gene]["snp_expr_h2"] * alpha_inf1**2

        # EMBER method
        num_snps = len(pgc_snp_names)
        pca = PCA(n_components=num_comps)
        genos_plus_comb = pca.fit_transform(genos_plus)
        genos_only_comb = pca.transform(genos_only)

        beta1s_inferred = neg_probit_beta1_em(prev, alpha_inf1, cmc_exprs, genos_plus_comb,
                                              pgc_phenos, genos_only_comb, verbose=False)
        beta1s_inferred = np.dot(pca.components_.T, beta1s_inferred)
        beta1s_inferred_test = beta1s_inferred # for further adjustments below
        beta1s_inferred = beta1s_inferred[:len(pgc_snp_names)]

        # linreg_only, no PCA
        beta1s_linreg_nopca = LinearRegression().fit(genos_plus, cmc_exprs).coef_
        beta1s_linreg_nopca = beta1s_linreg_nopca[:num_snps]
        beta1s_enet_nopca = ElasticNet(alpha=1e-1, max_iter=10000, selection='random').fit(genos_plus, cmc_exprs).coef_
        beta1s_enet_nopca = beta1s_enet_nopca[:num_snps]

        # linreg BLUP
        x_obs = genos_plus[:, :num_snps]
        x_obs_covs = genos_plus[:, num_snps:]
        beta1s_linreg_blup = linreg_blup(X=x_obs_covs, Z=x_obs, y=cmc_exprs, c=0.0001)
        beta1s_linreg_blup_slower = linreg_blup_slower(X=x_obs_covs, Z=x_obs, y=cmc_exprs, c=0.0001)


        # read TWAS/FUSION BLUP results from file
        if sg_dict[gene]["name"] in expr_dict_cmc:
            snps_dict_temp = expr_dict_cmc[sg_dict[gene]["name"]]
            true_blups_gtex_comp_tmp = np.array([gt_dict[gene][x]['blup_wgt'] for x in pgc_snp_names]).astype(np.float)
            true_snp_pos_comp_tmp = [gt_dict[gene][snp_name]['bppos'] for snp_name in pgc_snp_names]
            true_blups_cmc_comp = []
            true_blups_gtex_comp = []
            true_snp_pos_comp = []
            for i,snp_name in enumerate(pgc_snp_names):
                if snp_name in snps_dict_temp:
                    eff_al_1 = gt_dict[gene][snp_name]["eff_al"]
                    alt_al_1 = gt_dict[gene][snp_name]["alt_al"]
                    eff_al_2 = snps_dict_temp[snp_name]["eff_al"]
                    alt_al_2 = snps_dict_temp[snp_name]["alt_al"]
                    true_blups_gtex_comp.append(true_blups_gtex_comp_tmp[i])
                    true_snp_pos_comp.append(true_snp_pos_comp_tmp[i])
                    if eff_al_1==eff_al_2 and alt_al_1==alt_al_2:
                        true_blups_cmc_comp.append(float(snps_dict_temp[snp_name]["blup_wgt"]))
                    elif eff_al_1==alt_al_2 and alt_al_1==eff_al_2:
                        true_blups_cmc_comp.append(-1*float(snps_dict_temp[snp_name]["blup_wgt"]))
                    else:
                        print(gene, snp_name, "error in matching alleles between CMC and GTEx")
                else:
                    print(gene, "missing SNP", snp_name)
            comp_blup_r2 = rsquared(true_blups_gtex_comp, true_blups_cmc_comp)
            gene_varexp[gene]["comp_blup_r2"] = comp_blup_r2


        true_blups = np.array([gt_dict[gene][x]['blup_wgt'] for x in pgc_snp_names]).astype(np.float)


        if sg_dict[gene]["name"] in expr_dict_cmc:
            plot_effects(gene_name=sg_dict[gene]["name"],
                         true_blups_gtex=true_blups,
                         linreg_effs=beta1s_linreg[:len(pgc_snp_names)],
                         ember_effs=beta1s_inferred,
                         blup_effs=beta1s_linreg_blup_slower,
                         true_blups_gtex_comp=true_blups_gtex_comp,
                         true_blups_cmc=true_blups_cmc_comp)
            plot_effects_by_pos(gene_name=sg_dict[gene]["name"],
                                snp_pos=[gt_dict[gene][snp_name]['bppos'] for snp_name in pgc_snp_names],
                                true_blups_gtex=true_blups,
                                linreg_effs=beta1s_linreg[:len(pgc_snp_names)],
                                ember_effs=beta1s_inferred,
                                blup_effs=beta1s_linreg_blup_slower,
                                snp_pos_comp=true_snp_pos_comp,
                                true_blups_cmc=true_blups_cmc_comp)
            snp_pos_comp_pickle = true_snp_pos_comp
            true_blups_gtex_comp_pickle = true_blups_gtex_comp
            true_blups_cmc_pickle = true_blups_cmc_comp
        else:
            plot_effects(gene_name=sg_dict[gene]["name"],
                         true_blups_gtex=true_blups,
                         linreg_effs=beta1s_linreg[:len(pgc_snp_names)],
                         ember_effs=beta1s_inferred,
                         blup_effs=beta1s_linreg_blup_slower,
                         true_blups_gtex_comp=None,
                         true_blups_cmc=None)
            plot_effects_by_pos(gene_name=sg_dict[gene]["name"],
                                snp_pos=[gt_dict[gene][snp_name]['bppos'] for snp_name in pgc_snp_names],
                                true_blups_gtex=true_blups,
                                linreg_effs=beta1s_linreg[:len(pgc_snp_names)],
                                ember_effs=beta1s_inferred,
                                blup_effs=beta1s_linreg_blup_slower,
                                snp_pos_comp=None,
                                true_blups_cmc=None)
            snp_pos_comp_pickle = None
            true_blups_gtex_comp_pickle = None
            true_blups_cmc_pickle = None
        pickle_file = "ember_effects_%s.pickle" % (sg_dict[gene]["name"])
        if not os.path.exists(pickle_file):
            pickle.dump({"gene_name":sg_dict[gene]["name"],
                         "true_blups_gtex":true_blups,
                         "linreg_effs":beta1s_linreg[:len(pgc_snp_names)],
                         "ember_effs":beta1s_inferred,
                         "blup_effs":beta1s_linreg_blup_slower,
                         "true_blups_gtex_comp":true_blups_gtex_comp_pickle,
                         "snp_pos":[gt_dict[gene][snp_name]['bppos'] for snp_name in pgc_snp_names],
                         "snp_pos_comp":snp_pos_comp_pickle,
                         "true_blups_cmc":true_blups_cmc_pickle}, open(pickle_file, "wb"))


        # linear regression single variable
        beta1s_lrsngl = []
        covars = genos_plus[:, num_snps:]
        for i in range(num_snps):  # ignore the covariates
            genos_single = np.concatenate((genos_plus[:, i][:, None], covars), axis=1)
            beta1_lr = LinearRegression().fit(genos_single, cmc_exprs).coef_[0]
            beta1s_lrsngl.append(beta1_lr)
        beta1s_lrsngl = np.array(beta1s_lrsngl)

        # plot reported BLUP vs e-net for cis-SNPs to gene
        true_blups = np.array([gt_dict[gene][x]['blup_wgt'] for x in pgc_snp_names]).astype(np.float)
        true_enets = np.array([gt_dict[gene][x]['enet_wgt'] for x in pgc_snp_names]).astype(np.float)

        beta1s_lrsngl_r2 = rsquared(true_blups, beta1s_lrsngl)
        beta1s_linreg_r2 = rsquared(true_blups, beta1s_linreg[:len(pgc_snp_names)])
        beta1s_inferred_r2 = rsquared(true_blups, beta1s_inferred)
        beta1s_blup_r2 = rsquared(true_blups, beta1s_linreg_blup_slower)
        gene_varexp[gene]["linreg_r2"] = beta1s_linreg_r2
        gene_varexp[gene]["inferred_r2"] = beta1s_inferred_r2
        gene_varexp[gene]["blup_r2"] = beta1s_blup_r2

    pickle_file = "ember_global_performance.pickle"
    if not os.path.exists(pickle_file):
        pickle.dump((gene_varexp, gene_list), open(pickle_file, "wb"))
    plot_mediated_varexp(gene_varexp, gene_list)
    plot_performance_by_varexp(gene_varexp, gene_list, "inf_mediated_h2")
    plot_performance_blup(gene_varexp, gene_list)
    # plot_performance_by_varexp(gene_varexp, gene_list, "inf_expr_varexp")
    table_performance_by_varexp(gene_varexp, gene_list, sg_dict, xaxis="inf_mediated_h2")
