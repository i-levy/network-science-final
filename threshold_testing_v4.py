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
all_subs_fc = []
all_subs_sparser = []
for sub in subs:
    fc_path = os.path.join(spi_dir, sub, 'symmetrized', 'covariance_symmetrized.npy')
    sc_path = os.path.join(sc_dir, f'sub-{sub}_parc-shen268_tract-prob_sc.npy')
    
    fc_raw = np.load(fc_path)
    fc = np.abs(fc_raw)
    all_subs_fc.append(fc)

    sc = np.load(sc_path)

    # Remove edges in FC that don't exist in SC
    sparser = fc*(sc!=0)
    all_subs_sparser.append(sparser)
stacked_fc = np.stack(all_subs_fc)
stacked_sparser = np.stack(all_subs_sparser)

results_fc = {'clustering': [], 'path_length': [],
              'greedy_modularity': [], 'greedy_num_communities': [],
              'louvain_modularity': [], 'louvain_num_communities': [],
              'leiden_modularity': [], 'leiden_num_communities': [],
              'swp': [], 'delta_C': [], 'delta_L': [],
              'density': [], 'nodes': [], 'edges': []}

results_sparser = {'clustering': [], 'path_length': [],
                   'greedy_modularity': [], 'greedy_num_communities': [],
                   'louvain_modularity': [], 'louvain_num_communities': [],
                   'leiden_modularity': [], 'leiden_num_communities': [],
                   'swp': [], 'delta_C': [], 'delta_L': [],
                   'density': [], 'nodes': [], 'edges': []}

for i in range(stacked_fc.shape[0]):
    # FULLY CONNECTED
    adj = stacked_fc[i]
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
    results_fc['clustering'].append(c)
    results_fc['path_length'].append(l)
    results_fc['greedy_modularity'].append(greedy_q)
    results_fc['greedy_num_communities'].append(greedy_num)
    results_fc['louvain_modularity'].append(louvain_q)
    results_fc['louvain_num_communities'].append(louvain_num)
    results_fc['leiden_modularity'].append(leiden_q)
    results_fc['leiden_num_communities'].append(leiden_num)
    results_fc['swp'].append(swp)
    results_fc['delta_C'].append(delta_C)
    results_fc['delta_L'].append(delta_L)
    results_fc['density'].append(density)
    results_fc['nodes'].append(nodes)
    results_fc['edges'].append(edges)

    # SPARSER WITH SC
    adj_sparser = stacked_sparser[i]
    G_sparser = nx.Graph(adj_sparser)
    nodes_sparser = G_sparser.number_of_nodes()
    density_sparser = nx.density(G_sparser)
    edges_sparser = G_sparser.number_of_edges()
    # Clustering
    c_sparser = clustering_coefficient_bct(adj_sparser)
    # Path length
    l_sparser = characteristic_path_length(adj_sparser)
    # Greedy modularity
    greedy_communities_sparser = nx.community.greedy_modularity_communities(G_sparser, weight='weight')
    greedy_q_sparser = nx.community.modularity(G_sparser, greedy_communities_sparser, weight='weight')
    greedy_num_sparser = len(greedy_communities_sparser)
    # Louvain modularity
    louvain_communities_sparser = nx.community.louvain_communities(G_sparser, weight='weight', seed=59)
    louvain_q_sparser = nx.community.modularity(G_sparser, louvain_communities_sparser, weight='weight')
    louvain_num_sparser = len(louvain_communities_sparser)
    # Leiden modularity
    g_sparser = ig.Graph.Weighted_Adjacency(adj_sparser.tolist(), mode="undirected")
    ig.set_random_number_generator(random.Random(59))
    leiden_partition_sparser = g_sparser.community_leiden(
        objective_function="modularity",
        weights="weight",
        n_iterations=10
    )
    leiden_q_sparser = leiden_partition_sparser.modularity
    membership_sparser = leiden_partition_sparser.membership
    leiden_communities_sparser = [set(c) for c in leiden_partition_sparser]
    leiden_num_sparser = len(leiden_communities_sparser)
    # SWP
    swp_sparser, delta_C_sparser, delta_L_sparser = SWP(adj_sparser)
    # Save
    results_sparser['clustering'].append(c_sparser)
    results_sparser['path_length'].append(l_sparser)
    results_sparser['greedy_modularity'].append(greedy_q_sparser)
    results_sparser['greedy_num_communities'].append(greedy_num_sparser)
    results_sparser['louvain_modularity'].append(louvain_q_sparser)
    results_sparser['louvain_num_communities'].append(louvain_num_sparser)
    results_sparser['leiden_modularity'].append(leiden_q_sparser)
    results_sparser['leiden_num_communities'].append(leiden_num_sparser)
    results_sparser['swp'].append(swp_sparser)
    results_sparser['delta_C'].append(delta_C_sparser)
    results_sparser['delta_L'].append(delta_L_sparser)
    results_sparser['density'].append(density_sparser)
    results_sparser['nodes'].append(nodes_sparser)
    results_sparser['edges'].append(edges_sparser)
        
results_all = {'fully_connected': results_fc, 'sparser': results_sparser}
with open('threshold_testing_results_v4.dill', 'wb') as f:
    dill.dump(results_all, f)