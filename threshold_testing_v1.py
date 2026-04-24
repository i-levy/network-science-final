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

thresholds = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5]
results = {t: {'clustering': [], 'path_length': [], 'swp': []} for t in thresholds}
# Loop over each threshold
for threshold in thresholds:
    # Loop over each subject
    for i in range(stacked.shape[0]):
        # Threshold at current level
        current = stacked[i]
        adj = threshold_proportional(current, threshold)
        c = clustering_coefficient_bct(adj)
        l = characteristic_path_length(adj)
        swp, _, _ = SWP(adj)
        results[threshold]['clustering'].append(c)
        results[threshold]['path_length'].append(l)
        results[threshold]['swp'].append(swp)

with open('threshold_testing_results_v1.dill', 'wb') as f:
    dill.dump(results, f)