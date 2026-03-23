import numpy as np
import pandas as pd
from tqdm import tqdm
import os
import networkx as nx
from scipy import sparse
from sklearn.preprocessing import MinMaxScaler
from bct.algorithms.reference import latmio_und, randmio_und
from itertools import combinations

def clustering_coefficient(adj):
    """
    Calculates the local clustering coefficient for each node using the method from Onnela et al. and returns the average clustering coefficient across all nodes.

    Inputs
    ---
    adj    Weighted, undirected graph adjacency matrix.

    Returns
    ---
    C      Global clustering coefficient for inputted network.
    
    """

    # Initialize empty list to store all clustering coefficients
    coeffs = []
    # Get max weight
    w_max = np.max(adj)

    # Loop through all nodes
    for i in range(adj.shape[0]):
        # Calculate normalizing constant
        k_i = np.count_nonzero(adj[i])
        # If i is only connected to one other node, avoid divide by zero error and set the normalizing constant to 0
        if k_i == 1:
            norm = 0
        elif k_i == 0:
            norm = 0
        else:
            norm = 1/(k_i * (k_i - 1))

        # Get list of indices that are not i
        idx_all = np.array(list(range(adj.shape[0])))
        idx = idx_all[idx_all != i]
        # Get all pairwise combinations of indices
        combos = list(combinations(idx, 2))

        # Initialize inner sum
        inner_sum = 0
        # Loop through all cominations
        for j, k in combos:
            # Get weights for ij, ik, and jk
            w_ij = adj[i, j]
            w_jk = adj[j, k]
            w_ik = adj[i, k]
    
            # Calculate normalized weights
            w_hat_ij = w_ij / w_max
            w_hat_jk = w_jk / w_max
            w_hat_ik = w_ik / w_max

            # Calculate product of normalized edge weights
            product = (w_hat_ij * w_hat_jk * w_hat_ik)**(1/3)

            # Add 
            inner_sum += product

        # Calculate local clustering coefficient
        coeff = norm * inner_sum
        # Append local clustering coefficient
        coeffs.append(coeff)
        
    # Average all local coefficients
    C = np.mean(coeffs)

    # Return average clustering coefficient
    return C

def characteristic_path_length(adj):
    """
    Calculates the characteristic path length, L. Since this is for weighted networks, the distance between two nodes is defined as the inverse of the weight of the edge connecting the nodes, hence, d_ij = 1/w_ij.

    Inputs
    ---
    adj    Weighted, undirected graph adjacency matrix.

    Returns
    ---
    L      Characteristic path length for inputted network.
    
    """
    
    # Number of nodes
    N = adj.shape[0]
    # Calculate normalizing constant
    norm = 1 / (N * (N - 1))
    # Initialize sum
    running_sum = 0

    # Get all pairs of nodes
    pairs = list(combinations(list(range(adj.shape[0])), 2))
    # Loop through all nodes
    for i, j in pairs:
        # Get edge weight
        w_ij = adj[i, j]
        # Get distance, avoid divide by 0
        if w_ij == 0:
            d_ij = 0
        else:
            d_ij = 1 / w_ij

        # Add distance to running sum
        running_sum += d_ij

    # Normalize running sum by total number of edges
    L = norm * running_sum
    print(f'Num pairs: {len(pairs)}')
    print(f'N: {N}')
    print(f'norm: {norm}')
    print(f'sum: {running_sum}')
        
    # Return characteristic path length
    return L