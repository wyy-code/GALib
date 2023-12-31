import os
import numpy as np
import networkx as nx
import random
import pdb
import numpy as np
from scipy.io import loadmat

def print_graph_stats(G):
    print('# of nodes: %d, # of edges: %d' % (G.number_of_nodes(),
                                              G.number_of_edges()))
def construct_adjacency(G, id2idx):
    adjacency = np.zeros((len(G.nodes()), len(G.nodes())))
    for src_id, trg_id in G.edges():
        src_id = str(src_id)
        trg_id = str(trg_id)
        adjacency[id2idx[src_id], id2idx[trg_id]] = 1
        adjacency[id2idx[trg_id], id2idx[src_id]] = 1
    return adjacency

def build_degrees(G, id2idx):
    degrees = np.zeros(len(G.nodes()))
    for node in G.nodes():
        deg = len(G.neighbors(node))
        degrees[id2idx[node]] = deg
    return degrees

def build_clustering(G, id2idx):
    cluster = nx.clustering(G)
    # convert clustering from dict with keys are ids to array index-based
    clustering = [0] * len(G.nodes())
    for id, val in cluster.items():
        clustering[id2idx[id]] = val
    return clustering


def get_degree_similarity(A1, A2):
    """
    Computes the degree similarity matrix of two graphs based on their adjacency matrices.

    Parameters:
        A1 (numpy.ndarray): The adjacency matrix of the first graph.
        A2 (numpy.ndarray): The adjacency matrix of the second graph.

    Returns:
        numpy.ndarray: The degree similarity matrix.
    """
    degree_matrix_A1 = np.sum(A1, axis=1).reshape(-1, 1)
    degree_matrix_A2 = np.sum(A2, axis=1).reshape(1, -1)
    # min_degrees = np.minimum(degree_matrix_A1, degree_matrix_A2)
    # max_degrees = np.maximum(degree_matrix_A1, degree_matrix_A2)
    # degree_similarity_matrix = min_degrees / max_degrees
    degree_similarity_matrix = degree_matrix_A1 @ degree_matrix_A2


    return degree_similarity_matrix
    


def get_edges(G, id2idx):
    edges1 = [(id2idx[n1], id2idx[n2]) for n1, n2 in G.edges()]
    edges2 = [(id2idx[n2], id2idx[n1]) for n1, n2 in G.edges()]
    
    edges = edges1 + edges2
    edges = np.array(edges)
    return edges

def load_gt(path, id2idx_src, id2idx_trg, format='matrix', convert=False):    
    conversion_src = type(list(id2idx_src.keys())[0])
    conversion_trg = type(list(id2idx_trg.keys())[0])
    if format == 'matrix':
        gt = np.zeros((len(id2idx_src.keys()), len(id2idx_trg.keys())))
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()                
                gt[id2idx_src[conversion_src(src)], id2idx_trg[conversion_trg(trg)]] = 1
        return gt
    else:
        gt = {}
        with open(path) as file:
            for line in file:
                src, trg = line.strip().split()
                if convert:
                    gt[id2idx_src[conversion_src(src)]] = id2idx_trg[conversion_trg(trg)]
                else:
                    gt[conversion_src(src)] = conversion_trg(trg)
        return gt