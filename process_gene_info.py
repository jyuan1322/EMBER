import sys, os, pickle
import numpy as np
from pprint import pprint

# alpha p-value cutoff set to 1e-5.
def gen_gene_props(CMC_only = False, p_cutoff = None):
    gene_conv = {}
    #Gene stable ID,Gene name,Gene start (bp),Gene end (bp)
    with open("gene_ensembl_conversion.txt","rb") as f:
        next(f)
        for line in f:
            line = line.decode().strip().split(',')
            esbid = line[0]
            gname = line[1]
            if gname not in gene_conv:
                gene_conv[gname] = esbid

    gene_props = {}
    #TRAIT GENE CHR TxS TxE STUDY TWAS.Z TWAS.P BEST.GWAS.RSID BEST.GWAS.P GWAS.HSQ.LOCAL ALPHA
    with open("scz.txt","rb") as f:
        next(f)
        for line in f:
            line = line.decode().strip().split(' ')
            gname = line[1]
            study = line[5]
            twas_p = float(line[7])
            if not CMC_only or study=="CMC":
                if p_cutoff is None or twas_p < p_cutoff:
                    if gname in gene_conv:
                        esbid = gene_conv[gname]
                        if esbid not in gene_props:
                            gene_props[esbid] = {"esbid":esbid,
                                                "name":gname,
                                                "chrom":int(line[2][3:]),
                                                "bpstart":int(line[3]),
                                                "bpend":int(line[4]),
                                                "study":[study],
                                                "alpha":[float(line[11])]}
                        else:
                            gene_props[esbid]["study"].append(study)
                            gene_props[esbid]["alpha"].append(float(line[11]))
    for g in gene_props:
        gene_props[g]["mean_alpha"] = np.mean(gene_props[g]["alpha"])
    return gene_props

def get_all_beta1():
    GENOEXPRDIR = "Brain_Frontal_Cortex_BA9_exportcsv"
    # snpdict = {}
    # exprlist = []
    expr_dict = {}
    count = 0
    for filename in os.listdir(GENOEXPRDIR):
        gene_name = filename.split(".")[1]
        print(count, gene_name)
        count += 1
        snps = {}
        with open(os.path.join(GENOEXPRDIR,filename), "r") as f:
            next(f) # skip header line
            for line in f:
                # Grab the lasso or enet weights, and only if they are nonzero
                # TODO: verify effect/alt allele columns
                line = line.strip().split(",")
                snp = line[0].strip('"')
                chrm = int(line[1])
                bppos = int(line[3])
                eff_al = line[4].strip('"')
                alt_al = line[5].strip('"')
                try:
                    blup_wgt = float(line[6])
                    enet_wgt = float(line[9])
                except:
                    print("Invalid effect:", gene_name, snp, line[6], line[9])
                    continue


                snps[snp] = {"chrm":chrm,
                             "bppos":bppos,
                             "eff_al":eff_al,
                             "alt_al":alt_al,
                             "blup_wgt":blup_wgt,
                             "enet_wgt":enet_wgt}
        expr_dict[gene_name] = snps
    return expr_dict

def get_all_beta1_cmc():
    pickle_file = "expr_dict_cmc.pickle"
    if os.path.exists(pickle_file):
        expr_dict = pickle.load(open(pickle_file, "rb"))
    else:
        GENOEXPRDIR = "CMC.BRAIN.RNASEQ_exportcsv"
        expr_dict = {}
        count = 0
        for filename in os.listdir(GENOEXPRDIR):
            gene_name = filename.split(".")[1]
            print(count, gene_name)
            count += 1
            snps = {}
            with open(os.path.join(GENOEXPRDIR,filename), "r") as f:
                next(f) # skip header line
                for line in f:
                    # Grab the lasso or enet weights, and only if they are nonzero
                    # TODO: verify effect/alt allele columns
                    line = line.strip().split(",")
                    snp = line[0].strip('"')
                    chrm = int(line[1])
                    bppos = int(line[3])
                    eff_al = line[4].strip('"')
                    alt_al = line[5].strip('"')
                    try:
                        blup_wgt = float(line[6])
                        bslmm_wgt = float(line[7])
                    except:
                        print("Invalid effect:", gene_name, snp, line[6], line[7])
                        continue


                    snps[snp] = {"chrm":chrm,
                                 "bppos":bppos,
                                 "eff_al":eff_al,
                                 "alt_al":alt_al,
                                 "blup_wgt":blup_wgt,
                                 "bslmm_wgt":bslmm_wgt}
            expr_dict[gene_name] = snps
        pickle.dump(expr_dict, open(pickle_file, "wb"))
    return expr_dict

if __name__=="__main__":
    gene_props = gen_gene_props(CMC_only=False, p_cutoff = 1e-5)
    expr_dict = get_all_beta1()
    pickle.dump((gene_props, expr_dict), open("ember_all_betas.pickle", "wb"))

    # write SNP region file (for each gene) into file for imputation
    gene_props, expr_dict = pickle.load(open("ember_all_betas.pickle","rb"))
    snp_names = []
    enet_only = False
    with open("ember_snp_regions_impute.txt", "w") as f:
        for gene in expr_dict:
            if gene in gene_props:
                # print("*"*50)
                print("Filling snp regions file:", gene)
                snps = expr_dict[gene]
                regions = {}
                for snp in snps:
                    chrm = snps[snp]["chrm"]
                    bppos = snps[snp]["bppos"]
                    enet_wgt = float(snps[snp]["enet_wgt"])
                    if (not enet_only) or enet_wgt > 0:
                        snp_names.append(",".join(map(str, [snp, chrm, bppos])))
                        if chrm not in regions:
                            regions[chrm] = [bppos, bppos]
                        if bppos < regions[chrm][0]:
                            regions[chrm][0] = bppos
                        if bppos > regions[chrm][1]:
                            regions[chrm][1] = bppos
                for chrm in regions:
                    vals = [chrm, regions[chrm][0], regions[chrm][1]]
                    # print(vals)
                    f.write(",".join(map(str, vals)) + "\n")

    with open("ember_snp_names.txt", "w") as f:
        for snp in snp_names:
            f.write(snp + "\n")
