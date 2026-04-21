import numpy as np
import pandas as pd
import os
import networkx as nx
from scipy import sparse
from bct.algorithms.clustering import clustering_coef_wu
from itertools import combinations
from scipy.sparse.csgraph import shortest_path
from scipy.spatial.distance import cdist


def make_lattice_null(adj):
    """
    Generates a regular matrix with weights from the original network. 
    The radius r is automatically calculated from the network density.
    
    Parameters:
    -----------
    adj : Weighted, undirected adjacency matrix.
        
    Returns:
    --------
    lattice_null : Lattice null graph adjacency matrix.
    
    """
    # Ensure symmetry
    adj = (adj + adj.T) / 2
    n = len(adj)
    
    # Calculate r from network density
    # Count edges in upper triangle (excluding diagonal)
    upper_tri = np.triu(adj, k=1)
    E = np.count_nonzero(upper_tri)
    r = int(np.ceil(E / n))
    
    # Get all weights from upper triangle (including zeros)
    full_upper = np.triu(adj)
    all_weights = full_upper.flatten()
    
    # Sort in descending order
    sorted_weights = np.sort(all_weights)[::-1]
    
    # Calculate number of columns needed
    num_els = int(np.ceil(n / 2))
    
    # Pad or trim to reach required size: n * num_els
    required_size = n * num_els
    if len(sorted_weights) < required_size:
        sorted_weights = np.pad(sorted_weights, (0, required_size - len(sorted_weights)), 
                               constant_values=0)
    else:
        sorted_weights = sorted_weights[:required_size]
    
    # Reshape to matrix (n rows, num_els cols) using column-major order (MATLAB style)
    weight_matrix = sorted_weights.reshape(n, num_els, order='F')
    
    # Initialize output matrix
    lattice_null = np.zeros((n, n))
    
    # Assign weights to create regular network
    # For each node (origin)
    for node in range(n):
        # For each distance ring from 1 to r
        for dist in range(1, r + 1):            
            # Convert to 0-indexing
            col_idx = dist - 1  
            
            # Find available (non-zero) weights in this column
            available = np.where(weight_matrix[:, col_idx] != 0)[0]
            
            if len(available) > 0:
                # Randomly select a weight from available ones
                chosen_row = np.random.choice(available)
                weight = weight_matrix[chosen_row, col_idx]
                
                # Calculate target node at distance 'dist' around the ring
                target = (node + dist) % n
                
                # Assign symmetrically
                lattice_null[node, target] = weight
                lattice_null[target, node] = weight
                
                # Mark this weight as used (remove from pool)
                weight_matrix[chosen_row, col_idx] = 0
    
    return lattice_null


    
def make_random_null(adj):
    """
    Constructs a random null model by randomly redistributing observed edge 
    weights among nodes.

    Inputs
    ---
    adj  :  Weighted, undirected adjacency matrix.
           

    Returns
    ---
    random_null : Random null model with same number of edges and weight distribution
                  as adj, but randomly redistributed.
                  
    """
    adj = adj.astype(float)
    adj = (adj + adj.T) / 2
    N = adj.shape[0]

    # Extract upper triangle edges and their weights
    rows, cols = np.triu_indices(N, k=1)
    weights = adj[rows, cols]  

    # Shuffle the weights randomly
    shuffled_weights = np.random.permutation(weights)

    # Reconstruct symmetric matrix
    random_null = np.zeros((N, N))
    random_null[rows, cols] = shuffled_weights
    random_null[cols, rows] = shuffled_weights

    return random_null

def characteristic_path_length(adj):
    """
    Calculates the characteristic path length, L. Since this is for weighted networks, 
    the distance between two nodes is defined as the inverse of the weight of the edge 
    connecting the nodes, hence, d_ij = 1/w_ij.
    This uses scipy.sparse.csgraph.shortest_path to find the shortest paths.

    Inputs
    ---
    adj : Weighted, undirected graph adjacency matrix.

    Returns
    ---
    L   : Characteristic path length for inputted network.
    
    """
    # All floats
    adj = adj.astype(float)
    # Ensure symmetric
    adj = (adj + adj.T) / 2
    G = nx.Graph(adj)
    # Use only largest connected component (maybe not correct, but ideally all graphs will be connected)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G0 = G.subgraph(largest_cc)
        adj = nx.adjacency_matrix(G0).toarray()
    # Number of nodes
    N = adj.shape[0]
    # Create distance matrix by replacing edge weights with d_ij = 1/w_ij
    with np.errstate(divide='ignore'):
        dist = np.where(adj > 0, 1.0 / adj, np.inf)
    np.fill_diagonal(dist, 0.0)
    dist = shortest_path(dist, method='auto', directed=False)
    
    L = dist.sum() / (N * (N - 1))

    return L

def clustering_coefficient_bct(adj):
    """
    Calculates the local clustering coefficient for each node using the method 
    from Onnela et al. and returns the average clustering coefficient across all nodes. 
    This uses clustering_coef_wu() from bctpy because it is faster than my implementation, 
    but it does not normalize by the max weight, so this is done manually. The bctpy 
    implementation also is not averaged, but returns all local clustering coefficients, so
    this averages the returned local coefficients.

    Inputs
    ---
    adj : Weighted, undirected adjacency matrix of the observed network.

    Returns
    ---
    C   : Global clustering coefficient.
    """
    adj = adj.astype(float)
    adj = (adj + adj.T) / 2
    G = nx.Graph(adj)
    # Use only largest connected component (maybe not correct, but ideally all graphs will be connected)
    if not nx.is_connected(G):
        largest_cc = max(nx.connected_components(G), key=len)
        G0 = G.subgraph(largest_cc)
        adj = nx.adjacency_matrix(G0).toarray()
    # BCT doesn't normalize by w_max, but it's a faster implementation, so manually calculate w_max
    w_max = adj[~np.eye(adj.shape[0], dtype=bool)].max()
    if w_max == 0:
        return 0.0
    # Use BCT and normalize manually
    C = np.mean(clustering_coef_wu(adj / w_max))
    return C

def SWP(adj, coords=None, itr=100):
    """
    Calculates the Small-World Propensity of a weighted network. Calculates 
    characteristic path length, L, and clustering coefficient, C, of inputted network. 
    Then, generates a latticized version of the inputted network using make_lattice_null,
    and generates a randomized version of the inputted network using bct.algorithms.reference.randmio_und.
    Calculates L and C for the generated networks.

    Inputs
    ---
    adj       : Weighted, undirected graph adjacency matrix.

    Returns
    ---
    phi       : Small-World Propensity, in [0, 1].

    delta_L   : Fractional deviance of path length from respective null model.

    delta_C   : Fractional deviance of clustering coefficient from respective null model.
    """
    # Ensure floats
    adj = adj.astype(float)
    # Ensure symmetric matrix
    adj = (adj + adj.T) / 2
    # Use only largest connected component (maybe not correct, but ideally all graphs will be connected)
    G = nx.Graph(adj)
    if not nx.is_connected(G):
        print('Graph is not fully connected.')
        largest_cc = max(nx.connected_components(G), key=len)
        G0 = G.subgraph(largest_cc)
        adj = nx.adjacency_matrix(G0).toarray()
    # Calculate path length and clustering of observed network
    L_obs = characteristic_path_length(adj)
    #print(f'L_obs: {L_obs}')
    C_obs = clustering_coefficient_bct(adj)
    #print(f'C_obs: {C_obs}')

    # Randomize network
    rand = make_random_null(adj)
    # Path length of randomized network
    L_rand = characteristic_path_length(rand)
    #print(f'L_rand: {L_rand}')
    # Clustering of randomized network
    C_rand = clustering_coefficient_bct(rand)
    #print(f'C_rand: {C_rand}')
    
    # Latticize network
    #latt = make_spatial_lattice(adj=adj, node_positions=coords)
    #latt = latmio_und_connected(R=adj, itr=itr)
    latt = make_lattice_null(adj)
    # Path length of latticized network
    L_latt = characteristic_path_length(latt)
    #print(f'L_latt: {L_latt}')
    # Clustering of latticized network
    C_latt = clustering_coefficient_bct(latt)
    #print(f'C_latt: {C_latt}')

    # Calculate delta_C, comparing observed clustering to randomized and latticized versions
    delta_C = (C_latt - C_obs) / (C_latt - C_rand)
    # Keep delta_C between [0, 1]
    delta_C = np.clip(delta_C, 0, 1)
    
    # Calculate delta_L, comparing observed path length to randomized and latticized versions
    delta_L = (L_obs - L_rand) / (L_latt - L_rand)
    # Keep delta_L between [0, 1]
    delta_L = np.clip(delta_L, 0, 1)

    phi = 1 - np.sqrt(((delta_C ** 2) + (delta_L ** 2)) / 2)
    # Keep phi between [0, 1]
    phi = np.clip(phi, 0, 1)

    return phi, delta_C, delta_L