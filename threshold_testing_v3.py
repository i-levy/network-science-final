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
results = {t: {'clustering': [], 'path_length': [],
               'greedy_modularity': [], 'greedy_num_communities': [],
               'louvain_modularity': [], 'louvain_num_communities': [],
               'leiden_modularity': [], 'leiden_num_communities': [],
               'swp': [], 'delta_C': [], 'delta_L': [],
               'density': [], 'nodes': [], 'edges': []} for t in thresholds}
results_gcc = {t: {'clustering': [], 'path_length': [],
                   'greedy_modularity': [], 'greedy_num_communities': [],
                   'louvain_modularity': [], 'louvain_num_communities': [],
                   'leiden_modularity': [], 'leiden_num_communities': [],
                   'swp': [], 'delta_C': [], 'delta_L': [],
                   'density': [], 'nodes': [], 'edges': []} for t in thresholds}
results_disp = {t: {'clustering': [], 'path_length': [],
                    'greedy_modularity': [], 'greedy_num_communities': [],
                    'louvain_modularity': [], 'louvain_num_communities': [],
                    'leiden_modularity': [], 'leiden_num_communities': [],
                    'swp': [], 'delta_C': [], 'delta_L': [],
                    'density': [], 'nodes': [], 'edges': []} for t in thresholds}
# Loop over each threshold
for threshold in thresholds:
    # Loop over each subject
    for i in range(stacked.shape[0]):
        # Get current subject
        current = stacked[i]
        # Threshold at current level
        adj = threshold_proportional(current, threshold)
        G = nx.Graph(adj)
        nodes = G.number_of_nodes()
        density = nx.density(G)
        edges = G.number_of_edges()
        # Clustering
        c = clustering_coefficient_bct(adj)
        # Path length
        l = characteristic_path_length(adj)
        # Greedy modularity
        greedy_communities = nx.community.greedy_modularity_communities(G, weight='weight')
        greedy_q = nx.community.modularity(G, greedy_communities, weight='weight')
        greedy_num = len(greedy_communities)
        # Louvain modularity
        louvain_communities = nx.community.louvain_communities(G, weight='weight', seed=59)
        louvain_q = nx.community.modularity(G, louvain_communities, weight='weight')
        louvain_num = len(louvain_communities)
        # Leiden modularity
        g = ig.Graph.Weighted_Adjacency(adj.tolist(), mode="undirected")
        ig.set_random_number_generator(random.Random(59))
        leiden_partition = g.community_leiden(
            objective_function="modularity",
            weights="weight",
            n_iterations=10
        )
        leiden_q = leiden_partition.modularity
        membership = leiden_partition.membership
        leiden_communities = [set(c) for c in leiden_partition]
        leiden_num = len(leiden_communities)
        # SWP
        swp, delta_C, delta_L = SWP(adj)
        # Save
        results[threshold]['clustering'].append(c)
        results[threshold]['path_length'].append(l)
        results[threshold]['greedy_modularity'].append(greedy_q)
        results[threshold]['greedy_num_communities'].append(greedy_num)
        results[threshold]['louvain_modularity'].append(louvain_q)
        results[threshold]['louvain_num_communities'].append(louvain_num)
        results[threshold]['leiden_modularity'].append(leiden_q)
        results[threshold]['leiden_num_communities'].append(leiden_num)
        results[threshold]['swp'].append(swp)
        results[threshold]['delta_C'].append(delta_C)
        results[threshold]['delta_L'].append(delta_L)
        results[threshold]['density'].append(density)
        results[threshold]['nodes'].append(nodes)
        results[threshold]['edges'].append(edges)

        # -----------------------------------------------------------
        # Repeat but use only GCC for everything
        G = nx.Graph(adj)
        if not nx.is_connected(G):
            largest_cc = max(nx.connected_components(G), key=len)
            G_gcc = G.subgraph(largest_cc)
            adj_gcc = nx.adjacency_matrix(G_gcc).toarray()
        else:
            adj_gcc = adj.copy()
        G_gcc = nx.Graph(adj_gcc)
        nodes_gcc = G_gcc.number_of_nodes()
        density_gcc = nx.density(G_gcc)
        edges_gcc = G_gcc.number_of_edges()
        # Clustering
        c_gcc = clustering_coefficient_bct(adj_gcc)
        # Path length
        l_gcc = characteristic_path_length(adj_gcc)
        # Greedy modularity
        G_gcc = nx.Graph(adj_gcc)
        greedy_communities_gcc = nx.community.greedy_modularity_communities(G_gcc, weight='weight')
        greedy_q_gcc = nx.community.modularity(G_gcc, greedy_communities_gcc, weight='weight')
        greedy_num_gcc = len(greedy_communities_gcc)
        # Louvain modularity
        louvain_communities_gcc = nx.community.louvain_communities(G_gcc, weight='weight', seed=59)
        louvain_q_gcc = nx.community.modularity(G_gcc, louvain_communities_gcc, weight='weight')
        louvain_num_gcc = len(louvain_communities_gcc)
        # Leiden modularity
        g_gcc = ig.Graph.Weighted_Adjacency(adj_gcc.tolist(), mode="undirected")
        ig.set_random_number_generator(random.Random(59))
        leiden_partition_gcc = g_gcc.community_leiden(
            objective_function="modularity",
            weights="weight",
            n_iterations=10
        )
        leiden_q_gcc = leiden_partition_gcc.modularity
        membership_gcc = leiden_partition_gcc.membership
        leiden_communities_gcc = [set(c) for c in leiden_partition_gcc]
        leiden_num_gcc = len(leiden_communities_gcc)
        # SWP
        swp_gcc, delta_C_gcc, delta_L_gcc = SWP(adj_gcc)
        # Save
        results_gcc[threshold]['clustering'].append(c_gcc)
        results_gcc[threshold]['path_length'].append(l_gcc)
        results_gcc[threshold]['greedy_modularity'].append(greedy_q_gcc)
        results_gcc[threshold]['greedy_num_communities'].append(greedy_num_gcc)
        results_gcc[threshold]['louvain_modularity'].append(louvain_q_gcc)
        results_gcc[threshold]['louvain_num_communities'].append(louvain_num_gcc)
        results_gcc[threshold]['leiden_modularity'].append(leiden_q_gcc)
        results_gcc[threshold]['leiden_num_communities'].append(leiden_num_gcc)
        results_gcc[threshold]['swp'].append(swp_gcc)
        results_gcc[threshold]['delta_C'].append(delta_C_gcc)
        results_gcc[threshold]['delta_L'].append(delta_L_gcc)
        results_gcc[threshold]['density'].append(density_gcc)
        results_gcc[threshold]['nodes'].append(nodes_gcc)
        results_gcc[threshold]['edges'].append(edges_gcc)

        # -----------------------------------------------------------
        # Repeat but do disparity filter and use all remaining nodes
        try:
            G2 = nx.Graph(current.copy())
            G2 = disparity_filter(G2)
            G_disp = disparity_filter_alpha_cut(G2, alpha_t=threshold)
            density_disp = nx.density(G_disp)
            edges_disp = G_disp.number_of_edges()
            nodes_disp = G_disp.number_of_nodes()
            print(f'i={i}, threshold={threshold}, edges={edges_disp}, nodes={nodes_disp}')
            adj_disp = nx.to_numpy_array(G_disp)
            # Clustering
            c_disp = clustering_coefficient_bct(adj_disp)
            # Path length
            l_disp = characteristic_path_length(adj_disp)
            # Greedy modularity
            greedy_communities_disp = nx.community.greedy_modularity_communities(G_disp, weight='weight')
            greedy_q_disp = nx.community.modularity(G_disp, greedy_communities_disp, weight='weight')
            greedy_num_disp = len(greedy_communities_disp)
            # Louvain modularity
            louvain_communities_disp = nx.community.louvain_communities(G_disp, weight='weight', seed=59)
            louvain_q_disp = nx.community.modularity(G_disp, louvain_communities_disp, weight='weight')
            louvain_num_disp = len(louvain_communities_disp)
            # Leiden modularity
            g_disp = ig.Graph.Weighted_Adjacency(adj_disp.tolist(), mode="undirected")
            ig.set_random_number_generator(random.Random(59))
            leiden_partition_disp = g_disp.community_leiden(
                objective_function="modularity",
                weights="weight",
                n_iterations=10
            )
            leiden_q_disp = leiden_partition_disp.modularity
            membership_disp = leiden_partition_disp.membership
            leiden_communities_disp = [set(c) for c in leiden_partition_disp]
            leiden_num_disp = len(leiden_communities_disp)
            # SWP
            swp_disp, delta_C_disp, delta_L_disp = SWP(adj_disp)
            # Save
            results_disp[threshold]['clustering'].append(c_disp)
            results_disp[threshold]['path_length'].append(l_disp)
            results_disp[threshold]['greedy_modularity'].append(greedy_q_disp)
            results_disp[threshold]['greedy_num_communities'].append(greedy_num_disp)
            results_disp[threshold]['louvain_modularity'].append(louvain_q_disp)
            results_disp[threshold]['louvain_num_communities'].append(louvain_num_disp)
            results_disp[threshold]['leiden_modularity'].append(leiden_q_disp)
            results_disp[threshold]['leiden_num_communities'].append(leiden_num_disp)
            results_disp[threshold]['swp'].append(swp_disp)
            results_disp[threshold]['delta_C'].append(delta_C_disp)
            results_disp[threshold]['delta_L'].append(delta_L_disp)
            results_disp[threshold]['density'].append(density_disp)
            results_disp[threshold]['nodes'].append(nodes_disp)
            results_disp[threshold]['edges'].append(edges_disp)
        except:
            try:
                print(f'i={i}, threshold={threshold}, edges={edges_disp}, nodes={nodes_disp}')
            except:
                print('something really weird failed.')
        
results_all = {'all_nodes': results, 'gcc': results_gcc, 'disparity': results_disp}
with open('threshold_testing_results_v3.dill', 'wb') as f:
    dill.dump(results_all, f)