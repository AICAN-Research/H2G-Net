import numpy as np
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from prettytable import PrettyTable
from scipy import stats
import os
import pandas as pd
from utils import df_t_statistic, from_triu, create_upper_matrix, array2pretty, tukey_hsd


def main():
    # increase length of string in pandas
    pd.options.display.max_colwidth = 100

    model_results = "/mnt/EncryptedPathology/pathology/output/results/"

    grades = [1, 2, 3]
    skip_tissue_flag = True

    curr_name = ""

    f1s_models = []
    names_models = []

    # only get results from latest updated GT on single, merged HUNT dataset
    dataset = []
    for x in os.listdir(model_results):
        if x.endswith("_HUNT.csv"):
            if skip_tissue_flag and ("tissue" in x):
                continue
            dataset.append(x)

    for model in dataset:
        path = model_results + model

        data = np.genfromtxt(path, delimiter=",", dtype=str)

        names = data[0]
        data = data[1:].astype(np.float32)

        if "arch_" in model:
            curr_name = model.split("arch_")[-1].split("_")[0]
        else:
            if "deep_kmeans_" in model:
                curr_name = model.split("_model_")[-1].split("_")[0]
            else:
                if "tumor_unet_full" in model:
                    curr_name = "lowres_unet"
                else:
                    curr_name = "tissue"

        names_models.append(curr_name)

        print()
        print("#" * 100)
        print(model)
        print(curr_name)

        grade_vals = data[:, -1]

        for i, name in enumerate(names[:4]):
            if name == "Recall":
                i += 1
            elif name == "Precision":
                i -= 1
            print(name, ":", np.round(np.mean(data[:, i]), 3), np.round(np.std(data[:, i]), 3))
            if name == "F1":
                f1s_models.append(data[:, i])
                for grade in grades:
                    tmps = data[:, i]
                    print("Grade:", grade-1, "|", np.round(np.mean(tmps[grade_vals == grade-1]), 4), np.round(np.std(tmps[grade_vals == grade-1]), 4))

        print("#" * 100)

    new_order = np.array(["tissue", "lowres_unet", "inceptionv3", "mobile", "clustering", "unet", "agunet", "dagunet", "doubleunet"])
    if skip_tissue_flag:
        new_order = new_order[1:]
    names_models = np.array(names_models)
    tmp_f1s = np.array(f1s_models)
    tmp = []
    for n in new_order:
        tmp.append(tmp_f1s[names_models == n][0])
    f1s_models = tmp.copy()
    names_models = new_order.copy()

    # check significance in F1 (compare all groups)
    print("\n", "#" * 50, "\n")
    names_models = np.array(names_models)
    f1s_models_list = f1s_models.copy()
    f1s_models = np.array(f1s_models).transpose()

    # perform Tukey HSD, pairwise comparisons for all contrasts
    iters = list("123456789")
    if skip_tissue_flag:
        iters = iters[1:]
    m_comp = tukey_hsd(f1s_models_list, iters, f1s_models.shape[0], alpha=0.05)

    all_pvalues = -1 * np.ones((len(names_models), len(names_models)), dtype=np.float32)

    ps = m_comp.pvalues
    cnt = 0
    for i in range(len(names_models)):
        for j in range(i + 1, len(names_models)):
            all_pvalues[i, j] = ps[cnt]
            cnt += 1
    all_pvalues = np.round(all_pvalues, 4)  # .astype(np.str)
    all_pvalues = all_pvalues[:-1, 1:]

    new_names = ["Otsu", "UNet-LR", "Inc-PW", "Mob-PW", "Mob-KM-PW", "Mob-KM-PW-UNet", "Mob-KM-PW-AGUNet", "Mob-KM-PW-DAGUNet", "Mob-KM-PW-DoubleUNet"]
    if skip_tissue_flag:
        new_names = new_names[1:]

    col_new_names = ["\textbf{\rot{\multicolumn{1}{r}{" + n + "}}}" for n in new_names]

    out_pd = pd.DataFrame(data=all_pvalues, index=new_names[:-1], columns=col_new_names[1:])

    stack = out_pd.stack()
    stack[(0 < stack) & (stack <= 0.001)] = '\cellcolor{green!25}$<$0.001'

    for i in range(stack.shape[0]):
        try:
            curr = stack[i]
            #print(curr)
            if (float(curr) > 0.0011) & (float(curr) < 0.05):
                stack[i] = '\cellcolor{green!50}' + str(np.round(stack[i], 3))
            elif (float(curr) >= 0.05) & (float(curr) < 0.1):
                stack[i] = '\cellcolor{red!50}' + str(np.round(stack[i], 3))
            elif (float(curr) >= 0.1):
                stack[i] = '\cellcolor{red!25}' + str(np.round(stack[i], 3))
        except Exception:
            continue

    out_pd = stack.unstack()
    out_pd = out_pd.replace(0.0, " ")
    out_pd = out_pd.replace(-1.0, "-")

    curr_dataset = "HUNT0"

    with open("./tukey_pvalues_result_" + curr_dataset + ".txt", "w") as pfile:
        pfile.write("{}".format(out_pd.to_latex(escape=False, column_format="r" + "c"*all_pvalues.shape[1], bold_rows=True)))

    print(out_pd)


if __name__ == "__main__":
    main()
