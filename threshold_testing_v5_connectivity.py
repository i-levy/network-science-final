import pandas as pd
import numpy as np
import networkx as nx
import igraph as ig
import random
import os
from bct.utils.other import threshold_proportional, normalize
from small_world_propensity import SWP, characteristic_path_length, clustering_coefficient_bct
from multiscale_backbone import disparity_filter, disparity_filter_alpha_cut
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

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
results = {t: {'connectivity': []} for t in thresholds}
results_gcc = {t: {'connectivity': []} for t in thresholds}
results_disp = {t: {'connectivity': []} for t in thresholds}
# Loop over each threshold
for threshold in thresholds:
    # Loop over each subject
    for i in range(stacked.shape[0]):
        # Get current subject
        current = stacked[i]
        # Threshold at current level
        adj = threshold_proportional(current, threshold)
        upper_tri = adj[np.triu_indices_from(adj, k=1)]
        connectivity = upper_tri[upper_tri > 0].mean()
        results[threshold]['connectivity'].append(connectivity)

        # -----------------------------------------------------------
        # Repeat but use only GCC for everything
        G = nx.Graph(adj)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G_gcc = G.subgraph(largest_cc)
            adj_gcc = nx.adjacency_matrix(G_gcc).toarray()
        else:
            adj_gcc = adj.copy()
        upper_tri_gcc = adj_gcc[np.triu_indices_from(adj_gcc, k=1)]
        connectivity_gcc = upper_tri_gcc[upper_tri_gcc > 0].mean()
        results_gcc[threshold]['connectivity'].append(connectivity_gcc)

        # -----------------------------------------------------------
        # Repeat but do disparity filter and use all remaining nodes
        try:
            G2 = nx.Graph(current.copy())
            G2 = disparity_filter(G2)
            G_disp = disparity_filter_alpha_cut(G2, alpha_t=threshold)
            adj_disp = nx.to_numpy_array(G_disp)
            upper_tri_disp = adj_disp[np.triu_indices_from(adj_disp, k=1)]
            connectivity_disp = upper_tri_disp[upper_tri_disp > 0].mean()
            results_disp[threshold]['connectivity'].append(connectivity_disp)
        except:
            try:
                print(f'i={i}, threshold={threshold}, edges={edges_disp}, nodes={nodes_disp}')
            except:
                print('something really weird failed.')
        
results_all = {'all_nodes': results, 'gcc': results_gcc, 'disparity': results_disp}
with open('threshold_testing_results_v5_connectivity.dill', 'wb') as f:
    dill.dump(results_all, f)