import numpy as np
import matplotlib.pyplot as plt

models = ["STGCN", "SegRNN", "MTGNN", "CNN-LSTM", "TSD", "DCRNN",
          "LTransformer", "CrossFormer", "SageFormer", "DSN"]
online = ["SegRNN", "CNN-LSTM", "DCRNN", "LTransformer", "DSN"]
window = [15, 30, 45, 60, 90, 120]
a = 9604
markers = {"SegRNN": '.', "CNN-LSTM": "v", "DCRNN": "s", "LTransformer": "*", "DSN": "D",
           "CNN": ",", "STGCN": "1", "MTGNN": 'p', "TSD": '+', 'FEDFormer': 'x', 'CrossFormer': '<', 'SageFormer': '>'}
cmap = plt.colormaps.get_cmap('tab20').colors
colors = {"SegRNN": cmap[0], "CNN-LSTM": cmap[2], "DCRNN": cmap[4], "LTransformer": cmap[8], "DSN": cmap[6],
          "CNN": cmap[10], "STGCN": cmap[12], "MTGNN": cmap[14], "TSD": cmap[16], 'FEDFormer': cmap[18], 'CrossFormer': cmap[1],
          'SageFormer': cmap[3]}

detection_time = {"SegRNN": [2.20, 4.11, 5.80, 6.79, 10.28, 12.73],
                  "STGCN": [2.74, 4.41, 6.65, 7.82, 10.59, 13.49],
                  "MTGNN": [3.32, 5.02, 6.04, 8.69, 12.48, 16.07],
                  "CNN-LSTM": [3.10, 5.61, 6.84, 8.46, 12.40, 15.71],
                  "DCRNN": [3.29, 4.61, 6.85, 8.15, 10.07, 17.83],
                  "TSD": [2.80, 4.60, 5.93, 7.36, 13.53, 16.37],
                  "LTransformer": [3.16, 4.22, 6.31, 7.89, 12.91, 17.00],
                  "FEDFormer": [3.13, 4.42, 6.57, 7.83, 12.06, 16.93],
                  "CrossFormer": [3.16, 5.07, 6.67, 8.66, 12.44, 17.40],
                  "SageFormer": [3.22, 5.05, 6.23, 8.24, 12.17, 15.88],
                  "DSN": [3.12, 5.53, 7.98, 10.32, 15.09, 19.82]}

onset = {"SegRNN": [2.20, 4.11, 5.80, 6.79, 10.28, 12.73],
         "CNN-LSTM": [3.10, 5.61, 6.84, 8.46, 12.40, 15.71],
         "DCRNN": [3.29, 4.61, 6.85, 8.15, 10.07, 17.83],
         "LTransformer": [3.16, 4.22, 6.31, 7.89, 12.91, 17.00],
         "DSN": [3.12, 5.53, 7.98, 10.32, 15.09, 19.82]}

memory = {"SegRNN": [857, 917, 957, 999, 1083, 1169],
          "STGCN": [875, 947, 1045, 1125, 1289, 1453],
          "MTGNN": [1085, 1219, 1445, 1673, 2131, 2595],
          "CNN-LSTM": [1541, 2273, 2991, 3709, 5151, 6589],
          "DCRNN": [905, 1001, 1081, 1117, 1257, 1481],
          "TSD": [839, 865, 891, 933, 1007, 1091],
          "LTransformer": [837, 853, 887, 911, 957, 1019],
          "FEDFormer": [1147, 1639, 1887, 2867, 4053, 4815],
          "CrossFormer": [1101, 1535, 2049, 2683, 3359, 5007],
          "SageFormer": [1075, 1323, 1567, 2087, 2491, 4395],
          "DSN": [1223, 1311, 1579, 1771, 2125, 2455]}

onset_memory = {"SegRNN": [831, 831, 831, 831, 831, 831],
                "STGCN": [875, 947, 1045, 1125, 1289, 1453],
                "MTGNN": [1085, 1219, 1445, 1673, 2131, 2595],
                "CNN-LSTM": [877, 877, 877, 877, 877, 877],
                "DCRNN": [857, 857, 857, 857, 857, 857],
                "TSD": [839, 865, 891, 933, 1007, 1091],
                "LTransformer": [799, 799, 799, 799, 799, 799],
                "FEDFormer": [1147, 1639, 1887, 2867, 4453, 4815],
                "CrossFormer": [1101, 1535, 2049, 2683, 3359, 5007],
                "SageFormer": [1075, 1323, 1567, 2087, 2491, 4395],
                "DSN": [1129, 1129, 1129, 1129, 1129, 1129]}
fontsize = 20

# load data
total_seconds = 30 * a
throughput = {}
onset_time = {}
for m in models:
    if m in online:
        tp = a / np.array(detection_time[m]) * window
    else:
        tp = a / np.array(detection_time[m])
    throughput[m] = tp
    onset_time[m] = total_seconds / tp


for method in models:
    print(f"\t\t& {method} & ", end="")
    values = detection_time[method]
    print(" & ".join(['%.1f'%v for v in values]), end=" & ")
    values = memory[method]
    print(" & ".join(['%.d'%v for v in values]), end=" \\\\")
    print()


if __name__ == '__main__':
    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))

    # detection time
    ax = axs[0]
    ax.set_xlabel("Window Length (s)\n" + r"$\bf(a)$", fontsize=fontsize)
    ax.set_ylabel("Inference Time (s)", fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)
    for m in models:
        ax.plot(window, detection_time[m], label=m, marker=markers[m], color=colors[m])

    ax = axs[1]
    ax.set_xlabel("Window Length (s)\n"+ r"$\bf(b)$", fontsize=fontsize)
    ax.set_ylabel("GPU Memory (GB)", fontsize=fontsize)
    ax.set_yticks([2000, 4000, 6000], [2, 4, 6])
    ax.tick_params(labelsize=fontsize)
    for m in models:
        ax.plot(window, memory[m], marker=markers[m], color=colors[m])

    # onset inference time
    ax = axs[2]
    ax.set_xlabel("Window Length (s)\n"+ r"$\bf(c)$", fontsize=fontsize)
    ax.set_ylabel("Inference Time (s)", fontsize=fontsize)
    ax.set_yscale("log")
    # ax.set_yticks([10, 100], ["1", "2"])
    ax.tick_params(labelsize=fontsize)
    for m in models:
        ax.plot(window, onset_time[m], marker=markers[m], color=colors[m])

    ax = axs[3]
    ax.set_xlabel("Window Length (s)\n"+ r"$\bf(d)$", fontsize=fontsize)
    ax.set_ylabel("GPU Memory (GB)", fontsize=fontsize)
    ax.set_yticks([1000, 2000, 3000, 4000, 5000], [1, 2, 3, 4, 5])
    ax.tick_params(labelsize=fontsize)
    for m in models:
        ax.plot(window, onset_memory[m], marker=markers[m], color=colors[m])

    fig.legend(loc="upper center", fontsize=fontsize + 4, ncols=5, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78, hspace=0.3, wspace=0.25, left=0.04, bottom=0.17)
    plt.savefig("./ExpEfficiency.pdf", dpi=300)
    plt.show()
