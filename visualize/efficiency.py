import numpy as np
import matplotlib.pyplot as plt

models = ["STGCN", "SegRNN", "MTGNN", "CNN-LSTM", "TSD", "DCRNN-dist",
          "FEDFormer", "LinearTransformer", "CrossFormer", "DSN", "SageFormer", ]
online = {"SegRNN": 1, "STGCN": 0, "MTGNN": 0, "CNN-LSTM": 1, "DCRNN-dist": 1, "TSD": 0, "LinearTransformer": 1,
          "FEDFormer": 0, "CrossFormer": 0, "SageFormer": 0, "DSN": 1}
window = [15, 30, 45, 60, 90, 120]
a = 9604
detection_time = {"SegRNN": [2.20, 4.11, 5.80, 6.79, 10.28, 12.73],
                  "STGCN": [2.74, 4.41, 6.65, 7.82, 10.59, 13.49],
                  "MTGNN": [3.32, 5.02, 6.04, 8.69, 12.48, 16.07],
                  "CNN-LSTM": [3.10, 5.61, 6.84, 8.46, 12.40, 15.71],
                  "DCRNN-dist": [3.29, 4.61, 6.85, 8.15, 10.07, 17.83],
                  "TSD": [2.80, 4.60, 5.93, 7.36, 13.53, 16.37],
                  "LinearTransformer": [3.16, 4.22, 6.31, 7.89, 12.91, 17.00],
                  "FEDFormer": [3.13, 4.42, 6.57, 7.83, 12.06, 16.93],
                  "CrossFormer": [3.16, 5.07, 6.67, 8.66, 12.44, 17.40],
                  "SageFormer": [3.22, 5.05, 6.23, 8.24, 12.17, 15.88],
                  "DSN": [3.12, 5.53, 7.98, 10.32, 15.09, 19.82]}

onset = {"SegRNN": [2.20, 4.11, 5.80, 6.79, 10.28, 12.73],
                  "CNN-LSTM": [3.10, 5.61, 6.84, 8.46, 12.40, 15.71],
                  "DCRNN-dist": [3.29, 4.61, 6.85, 8.15, 10.07, 17.83],
                  "LinearTransformer": [3.16, 4.22, 6.31, 7.89, 12.91, 17.00],
                  "DSN": [3.12, 5.53, 7.98, 10.32, 15.09, 19.82]}

memory = {"SegRNN": [857, 917, 957, 999, 1083, 1169],
          "STGCN": [875, 947, 1045, 1125, 1289, 1453],
          "MTGNN": [1085, 1219, 1445, 1673, 2131, 2595],
          "CNN-LSTM": [1541, 2273, 2991, 3709, 5151, 6589],
          "DCRNN-dist": [905, 1001, 1081, 1117, 1257, 1481],
          "TSD": [839, 865, 891, 933, 1007, 1091],
          "LinearTransformer": [837, 853, 887, 911, 957, 1019],
          "FEDFormer": [1147, 1639, 1887, 2867, 4453, 4815],
          "CrossFormer": [1101, 1535, 2049, 2683, 3359, 5007],
          "SageFormer": [1075, 1323, 1567, 2087, 2491, 4395],
          "DSN": [1223, 1311, 1579, 1771, 2125, 2455]}

onset_memory = {"SegRNN": [831, 831, 831, 831, 831, 831],
                "STGCN": [875, 947, 1045, 1125, 1289, 1453],
                "MTGNN": [1085, 1219, 1445, 1673, 2131, 2595],
                "CNN-LSTM": [877, 877, 877, 877, 877, 877],
                "DCRNN-dist": [857, 857, 857, 857, 857, 857],
                "TSD": [839, 865, 891, 933, 1007, 1091],
                "LinearTransformer": [799, 799, 799, 799, 799, 799],
                "FEDFormer": [1147, 1639, 1887, 2867, 4453, 4815],
                "CrossFormer": [1101, 1535, 2049, 2683, 3359, 5007],
                "SageFormer": [1075, 1323, 1567, 2087, 2491, 4395],
                "DSN": [1129, 1129, 1129, 1129, 1129, 1129]}
markers = {"SegRNN": '.', "CNN-LSTM": "v", "DCRNN-dist": "s", "LinearTransformer": "*", "DSN": "D",
           "STGCN": "1", "MTGNN": 'p', "TSD": '+', 'FEDFormer': 'x', 'CrossFormer': '<', 'SageFormer': '>'}

fontsize = 15

if __name__ == '__main__':
    # load data
    total_seconds = 30 * a
    throughput = {}
    onset_time = {}
    for m in models:
        if online[m]:
            tp = a / np.array(detection_time[m]) * window
        else:
            tp = a / np.array(detection_time[m])
        throughput[m] = tp
        onset_time[m] = total_seconds / tp

    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 5))

    # detection time
    axs[0].set_title("Detection Inference Time")
    axs[0].set_yscale("log")
    for m in models:
        axs[0].plot(window, detection_time[m], label=m, marker=markers[m])

    axs[1].set_title("Detection GPU Memory (bs=256)")
    axs[1].set_yscale("log")
    for m in models:
        axs[1].plot(window, memory[m], marker=markers[m])

    # onset inference time
    axs[2].set_title("Onset Inference Time")
    axs[2].set_yscale("log")
    for m in models:
        axs[2].plot(window, onset_time[m], marker=markers[m])

    axs[3].set_title("Onset GPU Memory (bs=256)")
    axs[3].set_yscale("log")
    for m in models:
        axs[3].plot(window, onset_memory[m], marker=markers[m])

    fig.legend(loc="upper center", fontsize=fontsize, ncols=6, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(top=0.78)
    plt.savefig("./ExpEfficiency.png", dpi=500)
    plt.show()
