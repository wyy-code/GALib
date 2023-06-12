import numpy as np
import numpy.matlib as matlib
from scipy.sparse.linalg import svds
from encoder.network_alignment_model import NetworkAlignmentModel
from input.dataset import Dataset
import networkx as nx

class BigAlign(NetworkAlignmentModel):
    def __init__(self, adjA, adjB, lamb=0.01):
        """
        data1: object of Dataset class, contains information of source network
        data2: object of Dataset class, contains information of target network
        lamb: lambda
        """
        self.adjA = adjA
        self.adjB = adjB
        self.lamb = lamb

        self.N1 = self._extract_features(adjA)
        self.N2 = self._extract_features(adjB)


    def _extract_features(self, adj):
        """
        Preprocess input for unialign algorithms
        """
        G = nx.from_numpy_matrix(adj)

        n_nodes = len(G.nodes())
        
        N = np.zeros((n_nodes,2))
        N[:,0] = self.build_degrees(G)
        N[:,1] = self.build_clustering(G)
        return N
        # N = np.zeros((n_nodes, 3))
        # N[:, 0] = dataset.get_nodes_degrees()
        # N[:, 1] = dataset.get_nodes_clustering()
        # if dataset.features is not None:
        #     N[:,2] = dataset.features.argmax(axis=1)
        # return N
    def build_degrees(self, G):
        degrees = np.zeros(len(G.nodes()))
        for node in G.nodes():
            deg = len(G.neighbors(node))
            degrees[node] = deg
        return degrees
    
    def build_clustering(self, G):
        cluster = nx.clustering(G)
        # convert clustering from dict with keys are ids to array index-based
        clustering = [0] * len(G.nodes())
        for id, val in cluster.items():
            clustering[id] = val
        return clustering

    def align(self):
        N1, N2, lamb = self.N1, self.N2, self.lamb
        n2 = N2.shape[0]
        d = N2.shape[1]
        u, s, _ = np.linalg.svd(N1, full_matrices=False)

        # transform s
        S = np.zeros((s.shape[0], s.shape[0]))

        for i in range(S.shape[1]):
            S[i, i] = s[i]
            S[i, i] = 1 / S[i, i] ** 2

        X = N1.T.dot(u).dot(S).dot(u.T)
        Y = lamb / 2 * np.sum(u.dot(S).dot(u.T), axis=0)
        P = N2.dot(X) - matlib.repmat(Y, n2, 1)
        return P.T  # map from source to target
