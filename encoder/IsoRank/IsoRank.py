import numpy as np
from numpy import inf, nan
from encoder.graph_alignment_model import GraphAlignmentModel
from utils.graph_utils import get_degree_similarity
from scipy.sparse import csr_matrix

class IsoRank(GraphAlignmentModel):

    """
    Description:
      The algorithm computes the alignment/similarity matrix by a random walk
      based method. This algorithm is for non-attributed networks.
    Input:
      - A1, A2: adjacency matrices of two networks
      - H: the prior node similarity matrix, e.g., degree similarity matrix
      - alpha: decay factor, i.e., how important the global topology
               consistency is
      - maxiter: maximum number of iterations
    Output:
      - S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
       x in A2 is aligned to node-y in A1
    Reference:
      Singh, Rohit, Jinbo Xu, and Bonnie Berger.
      Global alignment of multiple protein interaction networks with application to functional orthology detection.
      Proceedings of the National Academy of Sciences 105.35 (2008): 12763-12768.
    """

    def __init__(self, adjA, adjB, H=None, alpha=0.82, maxiter=30, tol=1e-4):
 
        self.alignment_matrix = None
        self.A1 = adjA
        self.A2 = adjB
        self.alpha = alpha
        self.maxiter = maxiter
        if H is not None:
            self.H = H
        else:
            self.H = get_degree_similarity(adjA, adjB)
        self.tol = tol

    def align(self):

        n1 = self.A1.shape[0]
        n2 = self.A2.shape[0]

        # normalize the adjacency matrices
        d1 = 1 / self.A1.sum(axis=1)
        d2 = 1 / self.A2.sum(axis=1)

        d1[d1 == inf] = 0
        d2[d2 == inf] = 0
        d1 = d1.reshape(-1,1)
        d2 = d2.reshape(-1,1)

        W1 = d1*self.A1
        W2 = d2*self.A2
        S = np.ones((n2,n1)) / (n1 * n2) # Map target to source
        # IsoRank Algorithm in matrix form
        for iter in range(1, self.maxiter + 1):
            prev = S.flatten()
            if self.H is not None:
                S = (self.alpha*W2.T).dot(S).dot(W1) + (1-self.alpha) * self.H
            else:
                S = W2.T.dot(S).dot(W1)
            delta = np.linalg.norm(S.flatten()-prev, 2)
            print("Iteration: ", iter, " with delta = ", delta)
            if delta < self.tol:
                break

        self.alignment_matrix = S.T

        return self.alignment_matrix

    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix