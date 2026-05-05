import pandas as pd
import numpy as np
import networkx as nx
import os
from bct.utils.other import threshold_proportional, normalize
from small_world_propensity import SWP, characteristic_path_length, clustering_coefficient_bct
import dill

spi_dir = '/home/gbz6qn/Documents/research/data/coupling/working/included_subs/spis/'
sc_dir = '/home/gbz6qn/Documents/research/data/hcp_shen_sc/'
# List of subs
with open('/home/gbz6qn/Documents/research/code/coupling/compute_gca/subs.txt', 'r') as f:
    lines = f.read()
    subs = lines.split('\n')[:-1]
subs = sorted(subs)

# Load in all subjects into stacked array
all_subs = []
for sub in subs:
    fc_path = os.path.join(spi_dir, sub, 'symmetrized', 'covariance_symmetrized.npy')
    fc_raw = np.load(fc_path)
    fc = np.abs(fc_raw)
    all_subs.append(fc)
stacked = np.stack(all_subs)

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7]
results = {t: {'clustering': [], 'path_length': [], 'modularity': [],
               'swp': [], 'delta_C': [], 'delta_L': []} for t in thresholds}
results_gcc = {t: {'clustering': [], 'path_length': [], 'modularity': [],
                   'swp': [], 'delta_C': [], 'delta_L': []} for t in thresholds}
# Loop over each threshold
for threshold in thresholds:
    # Loop over each subject
    for i in range(stacked.shape[0]):
        # Get current subject
        current = stacked[i]
        # Threshold at current level
        adj = threshold_proportional(current, threshold)
        # Clustering
        c = clustering_coefficient_bct(adj)
        # Path length
        l = characteristic_path_length(adj)
        # Modularity
        G = nx.Graph(adj)
        communities = nx.community.greedy_modularity_communities(G, weight='weight')
        m = nx.community.modularity(G, communities, weight='weight')
        # SWP
        swp, delta_C, delta_L = SWP(adj)
        # Save
        results[threshold]['clustering'].append(c)
        results[threshold]['path_length'].append(l)
        results[threshold]['modularity'].append(m)
        results[threshold]['swp'].append(swp)
        results[threshold]['delta_C'].append(delta_C)
        results[threshold]['delta_L'].append(delta_L)

        # Repeat but use only GCC for everything
        G = nx.Graph(adj)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G_gcc = G.subgraph(largest_cc)
            adj_gcc = nx.adjacency_matrix(G_gcc).toarray()
        else:
            adj_gcc = adj.copy()
        
        # Clustering
        c_gcc = clustering_coefficient_bct(adj_gcc)
        # Path length
        l_gcc = characteristic_path_length(adj_gcc)
        # Modularity
        G_gcc = nx.Graph(adj_gcc)
        communities_gcc = nx.community.greedy_modularity_communities(G_gcc, weight='weight')
        m_gcc = nx.community.modularity(G_gcc, communities_gcc, weight='weight')
        # SWP
        swp_gcc, delta_C_gcc, delta_L_gcc = SWP(adj_gcc)
        # Save
        results_gcc[threshold]['clustering'].append(c_gcc)
        results_gcc[threshold]['path_length'].append(l_gcc)
        results_gcc[threshold]['modularity'].append(m_gcc)
        results_gcc[threshold]['swp'].append(swp_gcc)
        results_gcc[threshold]['delta_C'].append(delta_C_gcc)
        results_gcc[threshold]['delta_L'].append(delta_L_gcc)
        
results_all = {'all_nodes': results, 'gcc': results_gcc}
with open('threshold_testing_results_v2.dill', 'wb') as f:
    dill.dump(results_all, f)