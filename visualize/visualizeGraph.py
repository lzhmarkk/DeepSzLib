import os
import sys

sys.path.append(os.getcwd())
import math
import pickle
import torch
import numpy as np
import networkx as nx
from tqdm import tqdm
from utils.parser import parse
from models.DCRNN.graph import distance_support
from utils.utils import set_random_seed, to_gpu
from visualize.util import load_data, load_model
import collections
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import rankdata

nodes = ["FP1", "FP2", "F3", "F4", "C3", "C4", "P3", "P4", "O1", "O2", "F7",
         "F8", "T3", "T4", "T5", "T6", "FZ", "CZ", "PZ"]
font_size = 30


def get_spectral_graph_positions():
    """
    Get positions of EEG electrodes for visualizations
    """
    file = './data/electrode_graph/adj_mx_3d.pkl'

    with open(file, 'rb') as f:
        adj_mx_all = pickle.load(f)
    adj_mx = adj_mx_all[-1]

    node_id_dict = adj_mx_all[1]

    eeg_viz = nx.Graph()
    adj_mx = adj_mx_all[-1]
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        eeg_viz.add_node(i)

    for k, v in node_id_dict.items():
        node_id_label[v] = k
    # Add edges
    for i in range(adj_mx.shape[0]):
        for j in range(
                adj_mx.shape[1]):  # do no include self-edge in visualization
            if i != j and adj_mx[i, j] > 0:
                eeg_viz.add_edge(i, j)

    pos = nx.spectral_layout(eeg_viz)
    # keep the nice shape of the electronodes on the scalp
    pos_spec = {node: (-y, -x) for (node, (x, y)) in pos.items()}

    return pos_spec


def plot_brain_state(node_weight, adj_mx, pos_spec, name, topk=2, fig_size=(12, 8), node_color='Red',
                     vmin=0, vmax=1, edge_vmin=0, edge_vmax=1):
    is_directed = True
    eeg_viz = nx.DiGraph() if is_directed else nx.Graph()
    node_id_label = collections.defaultdict()

    for i in range(adj_mx.shape[0]):
        eeg_viz.add_node(i)

    for k, v in enumerate(nodes):
        node_id_label[k] = v

    # Add edges
    for i in range(adj_mx.shape[0]):
        ind = np.argsort(adj_mx[i])[-topk:]
        for j in ind:  # since it's now directed
            if i != j and adj_mx[i, j] > 0:
                eeg_viz.add_edge(i, j, weight=adj_mx[j, i])

    edges, weights = zip(*nx.get_edge_attributes(eeg_viz, 'weight').items())

    # Change the color scales below
    k = 20
    e_cmap = plt.cm.Greys(np.linspace(0, 1, (k + 1) * len(weights)))
    e_cmap = matplotlib.colors.ListedColormap(e_cmap[len(weights):-1:(k - 1)])
    n_cmap = plt.colormaps.get_cmap('viridis')

    plt.figure(figsize=fig_size)
    if node_weight is None:
        nx.draw_networkx(eeg_viz, pos_spec, labels=node_id_label, with_labels=True,
                         edgelist=edges, edge_color=rankdata(weights),
                         width=fig_size[1] / 2, edge_cmap=e_cmap, font_weight='bold',
                         node_size=250 * (fig_size[0] + fig_size[1]),
                         font_color='white',
                         font_size=font_size)
    else:
        nx.draw_networkx(eeg_viz, pos_spec, labels=node_id_label, with_labels=True,
                         edgelist=edges, edge_color=weights,
                         width=fig_size[1] / 2, edge_cmap=e_cmap,
                         edge_vmin=0, edge_vmax=0.1,
                         node_color=node_weight, cmap=n_cmap, vmin=vmin, vmax=vmax,
                         node_size=250 * (fig_size[0] + fig_size[1]),
                         font_weight='bold', font_color='white', font_size=font_size)
    plt.title("", fontsize=font_size)
    plt.axis('off')

    # sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    # sm.set_array([])
    # plt.colorbar(sm)

    plt.tight_layout()
    plt.savefig(os.path.join("./visualize_graph", f"brain-{name}.png"), dpi=300)
    plt.show()


def gen_graph_map():
    # if os.path.exists("./graph.npz"):
    #     return

    setattr(model, 'global_graphs', [])
    gc = model.global_graph_learner[0]
    static_graph = distance_support(C)

    pos_emb = gc.pos_emb.weight.reshape(1, C, H)
    learned_graph = torch.bmm(gc.w_q(pos_emb), gc.w_k(pos_emb).transpose(2, 1)) / math.sqrt(H)
    learned_graph = torch.softmax(learned_graph, dim=-1).squeeze().detach().numpy()

    mask = np.ones((19, 19))
    for i in range(16):
        for j in range(16):
            if i % 2 == j % 2:
                mask[i, j] = 1.2
    plot_brain_state(None, learned_graph, get_spectral_graph_positions(), 'learned')
    plot_brain_state(None, learned_graph * mask, get_spectral_graph_positions(), 'learned')
    exit(0)

    for b in tqdm(range(math.ceil(N / B))):
        x = fft_x[b * B:(b + 1) * B]
        with torch.no_grad():
            x = to_gpu(x, device=args.device)[0]
            _ = model(x, None, None)

    dynamic_graphs = model.global_graphs
    dynamic_graphs = torch.cat(dynamic_graphs, dim=0).reshape(N, T, C, C)
    np.savez("./graph.npz",
             static=static_graph,
             learned=learned_graph.cpu().numpy(),
             dynamic=dynamic_graphs.cpu().numpy())
    delattr(model, 'global_graphs')
    print("graph done")


def plot_graph_map(selected_index, select_timestamp):
    pos = get_spectral_graph_positions()
    os.makedirs("./visualize_graph", exist_ok=True)
    data = np.load("./graph.npz", allow_pickle=True)
    static_graph = data['static']
    learned_graph = data['learned']
    dynamic_graphs = data['dynamic']

    data = np.load("./attn.npz", allow_pickle=True)
    attn_weight = data['attn_weight']  # (N, T, C)

    """
    # static
    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.imshow(static_graph, origin='lower', aspect='auto', cmap='viridis')
    axs.axis('off')
    plt.savefig(os.path.join("./visualize_graph", "static.png"), dpi=300)
    plt.show()

    plot_brain_state(None, static_graph, pos, 'static')

    # learned
    fig, axs = plt.subplots(1, figsize=(5, 5))
    axs.imshow(learned_graph, origin='lower', aspect='auto', cmap='viridis')
    axs.axis('off')
    plt.savefig(os.path.join("./visualize_graph", "learned.png"), dpi=300)
    plt.show()
    plot_brain_state(None, learned_graph, pos, 'learned')
    """

    # dynamic
    i = selected_index
    graphs = dynamic_graphs[i][:, ::, ::-1]  # (T, C, C)
    fig, axs = plt.subplots(1, T, figsize=(3 * T, 3))
    vmin, vmax = graphs.min(), graphs.max()
    for t in range(T):
        axs[t].set_title(t)
        axs[t].imshow(graphs[t], origin='lower', aspect='auto', cmap='viridis')
        axs[t].axis('off')
    plt.tight_layout()
    plt.savefig(os.path.join("./visualize_graph", f"dynamic-{i}.png"), dpi=300)
    plt.show()

    for t in range(30):
        node_weight = -attn_weight[i]
        vmin = node_weight[3:].min()
        vmax = node_weight[3:].max() - 3
        # vmin = vmax = None
        evmin = graphs[select_timestamp].min()
        evmax = graphs[select_timestamp].max()
        # node_weight = (node_weight - node_weight.min()) / (node_weight.max() - node_weight.min())
        if t in select_timestamp:
            plot_brain_state(node_weight[t], graphs[t], pos, f"{i}-{t}",
                             vmin=vmin, vmax=vmax, edge_vmin=evmin, edge_vmax=evmax)


if __name__ == '__main__':
    args = parse()
    args.task = ['onset_detection']
    set_random_seed(args.seed)
    print(args)

    orig_x, fft_x, l = load_data(args)
    model = load_model(args)

    N = args.n_pos_train + args.n_pos_val + args.n_pos_test
    T = args.window // args.patch_len
    C = args.n_channels
    D = args.input_dim
    H = args.hidden
    S = args.patch_len
    B = args.batch_size

    gen_graph_map()
    # plot_graph_map(8024, [5, 11, 15, 21, 27])
