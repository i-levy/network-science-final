import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from bct.algorithms.reference import latmio_und, randmio_und
from itertools import combinations
from scipy.sparse.csgraph import shortest_path


def clustering_coefficient(adj):
    """
    Calculates the local clustering coefficient for each node using the method from Onnela et al. and returns the average clustering coefficient across all nodes.

    Inputs
    ---
    adj  :  numpy.ndarray
            Weighted, undirected graph adjacency matrix.

    Returns
    ---
    C  :  float
          Global clustering coefficient for inputted network.
    
    """
    # Ensure floats
    adj = adj.astype(float)

    # Ensure symmetric matrix
    adj = (adj + adj.T) / 2
    
    # Get max weight, excluding diagonal
    w_max = np.max(adj[~np.eye(adj.shape[0], dtype=bool)])
    if w_max == 0:
        return 0.0

    # Normalize weights
    w_hat = adj / w_max
    # Initialize empty array to store all clustering coefficients
    N = adj.shape[0]
    local_coeffs = np.zeros(N)

    # Loop through each node
    for i in range(N):
        # Find neighbors (nodes with positive weight)
        neighbors = np.where(adj[i] > 0)[0]
        k_i = len(neighbors)

        if k_i < 2:
            local_coeffs[i] = 0
            continue

        # Calculate normalizing constant
        norm = 2.0 / (k_i * (k_i - 1))

        # Sum over all neighbors
        inner_sum = 0
        for idx_j in range(k_i):
            for idx_k in range(idx_j + 1, k_i):
                j = neighbors[idx_j]
                k = neighbors[idx_k]

                # Get normalized weights
                w_ij = w_hat[i, j]
                w_ik = w_hat[i, k]
                w_jk = w_hat[j, k]

                # Calculate inner product
                product = (w_ij * w_ik * w_jk) ** (1/3)
                inner_sum += product
        local_coeffs[i] = norm * inner_sum
        
    # Average all local coefficients
    C = np.mean(local_coeffs)

    # Return average clustering coefficient
    return C

def characteristic_path_length(adj):
    """
    Calculates the characteristic path length, L. Since this is for weighted networks, the distance between two nodes is defined as the inverse of the weight of the edge connecting the nodes, hence, d_ij = 1/w_ij.
    This uses scipy.sparse.csgraph.shortest_path to find the shortest paths.

    Inputs
    ---
    adj    Weighted, undirected graph adjacency matrix.

    Returns
    ---
    L      Characteristic path length for inputted network.
    
    """
    adj = adj.astype(float)
    # Number of nodes
    N = adj.shape[0]
    # Create distance matrix using 1/w_ij
    distance_matrix = np.zeros_like(adj)
    for i in range(N):
        for j in range(N):
            if i == j:
                distance_matrix[i, j] = 0
            elif adj[i, j] > 0:
                distance_matrix[i, j] = 1 / adj[i, j]
            else:
                distance_matrix[i, j] = np.inf
    # Calculate shortest path
    distance_matrix = shortest_path(distance_matrix, method='auto', directed=False)

    total_sum = 0
    valid_pairs = 0

    for i in range(N):
        for j in range(N):
            if i != j and not np.isinf(distance_matrix[i, j]):
                total_sum += distance_matrix[i, j]
                valid_pairs += 1
    
    # Calculate normalizing constant
    norm = 1 / (N * (N - 1))
    # Calculate characteristic path length
    L = total_sum * norm
    
    return L

def SWP(adj):
    """
    Calculates the Small-World Propensity of a weighted network. Calculates characteristic path length, L, and clustering coefficient, C, of inputted network. Generates latticized and randomized versions of the inputted network and calculates L and C for the generated networks.

    Inputs
    ---
    adj    Weighted, undirected graph adjacency matrix.

    Returns
    ---
    SWP      Small-World Propensity, SWP in [0, 1]
    """

    # Calculate path length and clustering of observed network
    L_obs = characteristic_path_length(adj)
    print(f'L_obs: {L_obs}')
    C_obs = clustering_coefficient(adj)
    print(f'C_obs: {C_obs}')

    # Randomize network
    rand, _ = randmio_und(R=adj, itr=10)
    # Path length of randomized network
    L_rand = characteristic_path_length(rand)
    print(f'L_rand: {L_rand}')
    # Clustering of randomized network
    C_rand = clustering_coefficient(rand)
    print(f'C_rand: {C_rand}')
    
    # Latticize network
    latt, _, _, _ = latmio_und(R=adj, itr=10)
    # Path length of latticized network
    L_latt = characteristic_path_length(latt)
    print(f'L_latt: {L_latt}')
    # Clustering of latticized network
    C_latt = clustering_coefficient(latt)
    print(f'C_latt: {C_latt}')

    # Calculate delta_C, comparing observed clustering to randomized and latticized versions
    delta_C = (C_latt - C_obs) / (C_latt - C_rand)
    # Keep delta_C between [0, 1]
    if delta_C > 1:
        delta_C = 1
    elif delta_C < 0:
        delta_C = 0
    # Calculate delta_L, comparing observed path length to randomized and latticized versions
    delta_L = (L_obs - L_rand) / (L_latt - L_rand)
    # Keep delta_L between [0, 1]
    if delta_L > 1:
        delta_L = 1
    elif delta_L < 0:
        delta_L = 0

    SWP = 1 - np.sqrt(((delta_C ** 2) + (delta_L ** 2)) / 2)
    # Keep SWP between [0, 1]
    if SWP > 1:
        SWP = 1
    elif SWP < 0:
        SWP = 0

    return SWP, delta_C, delta_L