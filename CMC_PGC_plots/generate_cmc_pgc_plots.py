import sys, pickle, os, re
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from pprint import pprint
from scipy.stats import norm, linregress, binom

font = {'size': 16}
mpl.rc('font', **font)

# set seed in numpy
np.random.seed(0)

def rsquared(x, y):
    slope, intercept, r_value, p_value, std_err = linregress(x, y)
    return r_value**2 * np.sign(slope)

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

    for pfile in os.listdir("."):
        if pfile.startswith("ember_effects_") and pfile.endswith(".pickle"):
            effects = pickle.load(open(pfile, "rb"))
            plot_effects(gene_name=effects["gene_name"],
                         true_blups_gtex=effects["true_blups_gtex"],
                         linreg_effs=effects["linreg_effs"],
                         ember_effs=effects["ember_effs"],
                         blup_effs=effects["blup_effs"],
                         true_blups_gtex_comp=effects["true_blups_gtex_comp"],
                         true_blups_cmc=effects["true_blups_cmc"])
            plot_effects_by_pos(gene_name=effects["gene_name"],
                                snp_pos=effects["snp_pos"],
                                true_blups_gtex=effects["true_blups_gtex"],
                                linreg_effs=effects["linreg_effs"],
                                ember_effs=effects["ember_effs"],
                                blup_effs=effects["blup_effs"],
                                snp_pos_comp=effects["snp_pos_comp"],
                                true_blups_cmc=effects["true_blups_cmc"])


    pickle_file = "ember_global_performance.pickle"
    gene_varexp, gene_list = pickle.load(open(pickle_file, "rb"))
    plot_mediated_varexp(gene_varexp, gene_list)
    plot_performance_by_varexp(gene_varexp, gene_list, "inf_mediated_h2")
    plot_performance_blup(gene_varexp, gene_list)