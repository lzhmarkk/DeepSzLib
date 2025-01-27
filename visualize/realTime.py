import matplotlib.pyplot as plt

datasets = ["FDUSZ-Transductive", "TUSZ-Transductive"]
models = ["SegRNN", "STGCN", "CNN-LSTM", "DCRNN-dist", "TSD", "LTransformer", "CrossFormer", "SageFormer", "DeepSOZ", "DSN"]
markers = {"SegRNN": '.', "CNN-LSTM": "v", "DCRNN-dist": "s", "LTransformer": "*", "DSN": "D",
           "CNN": ",", "STGCN": "1", "MTGNN": 'p', "TSD": '+', 'DeepSOZ': 'x', 'CrossFormer': '<', 'SageFormer': '>'}
cmap = plt.colormaps.get_cmap('tab20').colors
colors = {"SegRNN": cmap[0], "CNN-LSTM": cmap[2], "DCRNN-dist": cmap[4], "LTransformer": cmap[8], "DSN": cmap[6],
          "CNN": cmap[10], "STGCN": cmap[12], "MTGNN": cmap[14], "TSD": cmap[16], 'DeepSOZ': cmap[18], 'CrossFormer': cmap[1],
          'SageFormer': cmap[3]}
fontsize = 24
k_range = [1, 15]

dr_all = {
    "FDUSZ-Transductive": {
        'SegRNN': [0.141, 0.25, 0.33, 0.382, 0.427, 0.463, 0.492, 0.516, 0.537, 0.557, 0.575, 0.591, 0.605, 0.614, 0.625],
        'STGCN': [0.021, 0.025, 0.026, 0.047, 0.157, 0.22, 0.309, 0.362, 0.43, 0.522, 0.604, 0.624, 0.643, 0.656, 0.667],
        'CNN-LSTM': [0.225, 0.353, 0.428, 0.478, 0.511, 0.542, 0.563, 0.582, 0.599, 0.612, 0.624, 0.635, 0.643, 0.652, 0.66],
        'DCRNN-dist': [0.116, 0.271, 0.369, 0.424, 0.465, 0.498, 0.53, 0.555, 0.575, 0.595, 0.61, 0.625, 0.641, 0.656, 0.667],
        'TSD': [0.256, 0.38, 0.438, 0.468, 0.493, 0.522, 0.539, 0.556, 0.568, 0.58, 0.591, 0.606, 0.618, 0.629, 0.637],
        'LTransformer': [0.448, 0.529, 0.563, 0.588, 0.606, 0.62, 0.636, 0.647, 0.657, 0.666, 0.675, 0.684, 0.691, 0.7, 0.706],
        'FEDFormer': [0.179, 0.271, 0.343, 0.405, 0.449, 0.487, 0.514, 0.538, 0.559, 0.573, 0.592, 0.607, 0.619, 0.632, 0.64],
        'CrossFormer': [0.29, 0.344, 0.446, 0.46, 0.498, 0.509, 0.552, 0.557, 0.579, 0.588, 0.61, 0.616, 0.634, 0.642, 0.656],
        'SageFormer': [0.186, 0.346, 0.423, 0.468, 0.498, 0.512, 0.527, 0.54, 0.554, 0.568, 0.578, 0.587, 0.596, 0.605, 0.612],
        'DeepSOZ': [0.215, 0.337, 0.409, 0.457, 0.489, 0.518, 0.538, 0.556, 0.573, 0.585, 0.597, 0.607, 0.615, 0.623, 0.631],
        'DSN': [0.402, 0.506, 0.559, 0.605, 0.64, 0.667, 0.689, 0.706, 0.727, 0.744, 0.759, 0.769, 0.777, 0.784, 0.793]
    },
    "TUSZ-Transductive": {
        'SegRNN': [0.12, 0.277, 0.379, 0.446, 0.503, 0.545, 0.581, 0.609, 0.634, 0.655, 0.669, 0.683, 0.694, 0.703, 0.713],
        'STGCN': [0.019, 0.086, 0.289, 0.393, 0.488, 0.548, 0.599, 0.634, 0.655, 0.673, 0.691, 0.704, 0.711, 0.718, 0.724],
        'CNN-LSTM': [0.13, 0.347, 0.461, 0.536, 0.588, 0.63, 0.659, 0.682, 0.702, 0.719, 0.733, 0.745, 0.754, 0.76, 0.77],
        'DCRNN-dist': [0.082, 0.276, 0.391, 0.455, 0.503, 0.541, 0.569, 0.592, 0.611, 0.626, 0.635, 0.645, 0.656, 0.662, 0.668],
        'TSD': [0.362, 0.446, 0.5, 0.536, 0.563, 0.584, 0.603, 0.623, 0.642, 0.653, 0.663, 0.67, 0.68, 0.684, 0.692],
        'LTransformer': [0.449, 0.541, 0.583, 0.614, 0.639, 0.66, 0.679, 0.694, 0.709, 0.722, 0.729, 0.736, 0.741, 0.747, 0.752],
        'FEDFormer': [0.05, 0.314, 0.43, 0.506, 0.57, 0.612, 0.642, 0.659, 0.673, 0.687, 0.696, 0.707, 0.72, 0.727, 0.735],
        'CrossFormer': [0.316, 0.447, 0.554, 0.59, 0.643, 0.657, 0.713, 0.722, 0.753, 0.76, 0.782, 0.785, 0.802, 0.806, 0.821],
        'SageFormer': [0.361, 0.485, 0.562, 0.615, 0.654, 0.679, 0.703, 0.718, 0.733, 0.745, 0.755, 0.765, 0.774, 0.781, 0.786],
        'DeepSOZ': [0.123, 0.328, 0.435, 0.506, 0.555, 0.595, 0.623, 0.644, 0.663, 0.679, 0.692, 0.704, 0.712, 0.718, 0.727],
        'DSN': [0.436, 0.537, 0.607, 0.655, 0.69, 0.728, 0.754, 0.775, 0.787, 0.798, 0.807, 0.816, 0.826, 0.831, 0.838]
    }
}

wr_all = {
    "FDUSZ-Transductive": {
        'SegRNN': [0.521, 0.52, 0.519, 0.517, 0.516, 0.515, 0.513, 0.512, 0.511, 0.51, 0.509, 0.508, 0.507, 0.507, 0.505],
        'STGCN': [0.511, 0.51, 0.51, 0.51, 0.509, 0.508, 0.508, 0.508, 0.508, 0.508, 0.509, 0.51, 0.511, 0.512, 0.513],
        'CNN-LSTM': [0.455, 0.455, 0.454, 0.454, 0.454, 0.454, 0.455, 0.455, 0.455, 0.456, 0.456, 0.457, 0.458, 0.459, 0.46],
        'DCRNN-dist': [0.449, 0.45, 0.451, 0.451, 0.452, 0.454, 0.455, 0.457, 0.458, 0.46, 0.462, 0.464, 0.466, 0.468, 0.47],
        'TSD': [0.51, 0.51, 0.511, 0.511, 0.511, 0.512, 0.513, 0.513, 0.515, 0.515, 0.516, 0.517, 0.519, 0.52, 0.521],
        'LTransformer': [0.559, 0.561, 0.563, 0.565, 0.567, 0.57, 0.572, 0.575, 0.578, 0.582, 0.585, 0.589, 0.594, 0.598, 0.603],
        'FEDFormer': [0.448, 0.449, 0.45, 0.451, 0.453, 0.455, 0.458, 0.46, 0.463, 0.466, 0.469, 0.473, 0.477, 0.483, 0.488],
        'CrossFormer': [0.481, 0.481, 0.481, 0.481, 0.481, 0.481, 0.481, 0.481, 0.482, 0.483, 0.484, 0.486, 0.487, 0.489, 0.492],
        'SageFormer': [0.746, 0.746, 0.747, 0.747, 0.748, 0.748, 0.749, 0.75, 0.752, 0.753, 0.755, 0.757, 0.759, 0.761, 0.764],
        'DeepSOZ': [0.476, 0.476, 0.475, 0.475, 0.475, 0.475, 0.476, 0.476, 0.476, 0.477, 0.477, 0.478, 0.479, 0.48, 0.481],
        'DSN': [0.439, 0.439, 0.44, 0.441, 0.442, 0.443, 0.445, 0.447, 0.449, 0.452, 0.454, 0.457, 0.461, 0.465, 0.469]
    },
    "TUSZ-Transductive": {
        'SegRNN': [0.418, 0.416, 0.414, 0.412, 0.41, 0.408, 0.406, 0.404, 0.402, 0.4, 0.398, 0.396, 0.394, 0.392, 0.39],
        'STGCN': [0.475, 0.474, 0.473, 0.473, 0.472, 0.472, 0.472, 0.471, 0.471, 0.47, 0.47, 0.469, 0.469, 0.468, 0.468],
        'CNN-LSTM': [0.349, 0.347, 0.345, 0.343, 0.342, 0.34, 0.339, 0.338, 0.336, 0.335, 0.334, 0.332, 0.331, 0.329, 0.328],
        'DCRNN-dist': [0.53, 0.53, 0.529, 0.529, 0.529, 0.529, 0.529, 0.53, 0.53, 0.531, 0.531, 0.532, 0.533, 0.534, 0.535],
        'TSD': [0.546, 0.546, 0.545, 0.545, 0.545, 0.545, 0.545, 0.545, 0.546, 0.546, 0.547, 0.547, 0.548, 0.55, 0.552],
        'LTransformer': [0.491, 0.492, 0.493, 0.495, 0.496, 0.498, 0.5, 0.503, 0.505, 0.508, 0.511, 0.515, 0.519, 0.523, 0.528],
        'FEDFormer': [0.741, 0.741, 0.742, 0.742, 0.744, 0.745, 0.746, 0.748, 0.75, 0.752, 0.755, 0.758, 0.761, 0.765, 0.769],
        'CrossFormer': [0.407, 0.405, 0.403, 0.401, 0.4, 0.399, 0.398, 0.398, 0.397, 0.397, 0.398, 0.398, 0.4, 0.402, 0.405],
        'SageFormer': [0.316, 0.316, 0.315, 0.315, 0.316, 0.317, 0.318, 0.32, 0.322, 0.324, 0.328, 0.331, 0.335, 0.34, 0.346],
        'DeepSOZ': [0.369, 0.367, 0.365, 0.363, 0.362, 0.36, 0.359, 0.358, 0.356, 0.355, 0.354, 0.351, 0.35, 0.348, 0.347],
        'DSN': [0.305, 0.305, 0.304, 0.304, 0.304, 0.304, 0.304, 0.305, 0.306, 0.307, 0.309, 0.311, 0.312, 0.314, 0.317]
    }
}

import numpy as np
results = []
for method in models:
    results.append(wr_all["FDUSZ-Transductive"][method])
results = np.array(results)

best_idx, second_best_idx = [], []
for i in range(15):
    idx = results[:, i].argsort()
    idx = list(reversed(idx))
    best_idx.append(idx[-1])
    second_best_idx.append(idx[-2])

best_values = []
second_best_values = []
for i, method in enumerate(models):
    print(f"\t\t& {method} & ", end="")
    values = results[i]
    for j, value in enumerate(values):
        if i == best_idx[j]:
            print("\\textbf{"+ "%.3f" % value +"}", end="")
        elif i == second_best_idx[j]:
            print("\\underline{"+ "%.3f" % value +"}", end="")
        else:
            print('%.3f'%value, end="")
        if j != len(values)- 1:
            print(" & ", end="")
    print(" \\\\")

if __name__ == '__main__':
    # plot
    fig, axs = plt.subplots(1, 4, figsize=(20, 6))
    for j, dataset in enumerate(datasets):
        dr = dr_all[dataset]
        wr = wr_all[dataset]

        # correct rate
        ax = axs[2 * j]
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel(f"Horizon\n"
                      f"({['a', 'b', 'c', 'd'][2 * j]})", fontsize=fontsize)
        ax.set_ylabel("Diagnosis Rate", fontsize=fontsize)
        ax.set_yticks([0.0, 0.3, 0.6, 0.7])
        ax.tick_params(labelsize=fontsize)
        for model in models:
            p = False
            score = dr[model]
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s for (i, s) in score]
            for i, s in score:
                if not p and s > 0.7:
                    print(model, i)
                    p = True
            if not p:
                print(model, 'inf')
            ax.plot(x, y, marker=markers[model], color=colors[model])
            if j == 0:
                ax.axhline(y=0.6, color='gray', linestyle='--')
                ax.axhline(y=0.7, color='gray', linestyle='--')

        # wrong rate
        ax = axs[2 * j + 1]
        ax.set_xticks(range(0, 16, 3))
        ax.set_xlabel(f"Horizon\n"
                      f"({['a', 'b', 'c', 'd'][2 * j + 1]})", fontsize=fontsize)
        ax.set_ylabel("Wrong Rate", fontsize=fontsize)
        ax.tick_params(labelsize=fontsize)
        for model in models:
            score = wr[model]
            score = [(i, s) for i, s in enumerate(score) if isinstance(s, float)]
            x = [i for (i, s) in score]
            y = [s - [0.06, 0.098][j] / 20 * i for (i, s) in score]  # amend the affect of cut off issues when calculating wrong rate
            if j == 0:
                ax.plot(x, y, label=model, marker=markers[model], color=colors[model])
            else:
                ax.plot(x, y, marker=markers[model], color=colors[model])

        print("-" * 30)

    fig.legend(bbox_to_anchor=(0.9, 1.02), fontsize=fontsize, ncols=5, columnspacing=1)
    fig.tight_layout()
    plt.subplots_adjust(wspace=0.3, top=0.8, bottom=0.195, right=0.997)
    plt.savefig("./ExpRealTime.pdf", dpi=300)
    plt.show()
