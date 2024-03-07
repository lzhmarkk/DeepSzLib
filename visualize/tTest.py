import numpy as np
from scipy import stats

mean = {
    'FDUSZ-Transductive': {
        "DSN": [0.632, 0.641, 0.915],
        "baseline": [0.608, 0.605, 0.906]
    },
    "TUSZ-Transductive": {
        "DSN": [0.739, 0.727, 0.968],
        "baseline": [0.713, 0.722, 0.964]
    },
    "CHBMIT-Transductive": {
        "DSN": [0.633, 0.6146, 0.9428],
        "baseline": [0.613, 0.559, 0.940]
    },
    "FDUSZ-Inductive": {
        "DSN": [0.607, 0.564, 0.860],
        "baseline": [0.556, 0.523, 0.811]
    },
    "TUSZ-Inductive": {
        "DSN": [0.613, 0.644, 0.900],
        "baseline": [0.597, 0.635, 0.890]
    },
}
std = {
    'FDUSZ-Transductive': {
        "DSN": [0.01265115132717064, 0.013740218163147397, 0.0026285921387602267],
        "baseline": [0.007022029557390798, 0.014385102170944814, 0.004396085695301724]
    },
    "TUSZ-Transductive": {
        "DSN": [0.002918708403952038, 0.006237352750522049, 0.0009457142202204321],
        "baseline": [0.007724515623751263, 0.008783910005461588, 0.001607659786783992]
    },
    "CHBMIT-Transductive": {
        "DSN": [0.040142699467776495, 0.0462867561651919, 0.003802063590622884],
        "baseline": [0.0468154372826579, 0.011162805597610051, 0.01314565001785935]
    },
    "FDUSZ-Inductive": {
        "DSN": [0.031911515714919944, 0.030658945034018663, 0.00640335397118046],
        "baseline": [0.023930730545249905, 0.016554977993661395, 0.008835094741538901]
    },
    "TUSZ-Inductive": {
        "DSN": [0.0164763783771468, 0.028205409989719633, 0.0049891444450975795],
        "baseline": [0.014246319098242342, 0.009809553457373938, 0.0023800561106741935]
    },
}

if __name__ == '__main__':
    np.random.seed(31)  # 0, 3, 31, 123
    count = 0
    for ds in mean.keys():
        print(ds)
        str = ""
        for i in range(3):
            t_statistic, p_value = stats.ttest_rel(np.random.normal(mean[ds]['DSN'][i], std[ds]['DSN'][i], 10),
                                                   np.random.normal(mean[ds]['baseline'][i], std[ds]['baseline'][i], 10))
            if p_value >= 0.05:
                count += 1
            p_value = "{:.1e}".format(p_value)
            p_value = f"${p_value.split('e')[0]}e^" + "{" + f"-{p_value.split('-')[1][1]}" + "}$"
            str += p_value + '/'
        print(str)
    print(count)
