import numpy as np
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from prettytable import PrettyTable


def df_t_statistic(mu1, mu2, std1, std2, n):
    diff = mu1 - mu2
    se = np.sqrt((std1 ** 2 + std2 ** 2) / N)
    return diff / se


def from_triu(A):
    out = A.T + A
    idx = np.arange(A.shape[0])
    out[idx, idx] = A[idx, idx]
    return out


def create_upper_matrix(values, size, val=3):
    upper = np.zeros((size, size))
    upper[np.triu_indices(size, val)] = values
    return upper


def array2pretty(res, fields, prec=".2"):
    x = PrettyTable()
    x.float_format = prec
    x.field_names = fields
    for row in range(res.shape[0]):
        x.add_row(res[row])
    return x


# https://www.tutorialfor.com/questions-62131.htm
def tukey_hsd (lst, ind, n, alpha=0.05):
    data_arr = np.hstack(lst)
    ind_arr = np.repeat(ind, n)
    return pairwise_tukeyhsd(data_arr, ind_arr, alpha=alpha)
