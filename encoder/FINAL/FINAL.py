import numpy as np
from sklearn.preprocessing import normalize
from numpy import inf, nan
from encoder.network_alignment_model import NetworkAlignmentModel
import argparse
import pdb

class FINAL(NetworkAlignmentModel):

    """
     Input:
       A1, A2: Input adjacency matrices with n1, n2 nodes
       N1, N2: Node attributes matrices, N1 is an n1*K matrix, N2 is an n2*K
             matrix, each row is a node, and each column represents an
             attribute. If the input node attributes are categorical, we can
             use one hot encoding to represent each node feature as a vector.
             And the input N1 and N2 are still n1*K and n2*K matrices.
             E.g., for node attributes as countries, including USA, China, Canada, 
             if a user is from China, then his node feature is (0, 1, 0).
             If N1 and N2 are emtpy, i.e., N1 = [], N2 = [], then no node
             attributes are input. 
    
       E1, E2: a L*1 cell, where E1{i} is the n1*n1 matrix and nonzero entry is
             the i-th attribute of edges. E2{i} is same. Similarly,  if the
             input edge attributes are categorical, we can use one hot
             encoding, i.e., E1{i}(a,b)=1 if edge (a,b) has categorical
             attribute i. If E1 and E2 are empty, i.e., E1 = {} or [], E2 = {}
             or [], then no edge attributes are input.
    
       H: a n2*n1 prior node similarity matrix, e.g., degree similarity. H
          should be normalized, e.g., sum(sum(H)) = 1.
       alpha: decay factor 
       maxiter, tol: maximum number of iterations and difference tolerance.
    
     Output: 
       S: an n2*n1 alignment matrix, entry (x,y) represents to what extend node-
        x in A2 is aligned to node-y in A1

    """

    def __init__(self, adjA, adjB, alpha=0.82, maxiter=30, tol=1e-4, attributes=None): 
        self.alignment_matrix = None
        self.A1 = adjA
        self.A2 = adjB
        self.alpha = alpha
        self.maxiter = maxiter
        self.H = self.get_H(adjA,adjB)
        self.tol = tol
        self.alignment_matrix = None
        self.attributes = attributes

        # % If no node attributes input, then initialize as a vector of 1
        # % so that all nodes are treated to have the save attributes which 
        # % is equivalent to no given node attribute.
        # E1 should be (2, A1.shape[0], A1.shape[1])

        if self.attributes == None:
            self.N1 = np.ones((self.A1.shape[0], 1))
            self.N2 = np.ones((self.A2.shape[0], 1))

        self.E1 = np.zeros((1, self.A1.shape[0], self.A1.shape[1]))
        self.E2 = np.zeros((1, self.A2.shape[0], self.A2.shape[1]))
        self.E1[0] = self.A1
        self.E2[0] = self.A2

    def align(self):
        
        L = self.E1.shape[0]
        K = self.N1.shape[1]
        T1 = np.zeros_like(self.A1)
        T2 = np.zeros_like(self.A2)
        # T1 = np.zeros([int(self.A1.shape[0]/2),int(self.A1.shape[1]/2)])
        # T2 = np.zeros([int(self.A2.shape[0]/2),int(self.A2.shape[1]/2)])
        # print(T1.shape,T2.shape)
        # print(self.E1[0].shape,self.E1[1].shape)

        # Normalize edge feature vectors
        for i in range(L):
            T1 += self.E1[i] ** 2
            T2 += self.E2[i] ** 2

        for i in range(T1.shape[0]):
            for j in range(T1.shape[1]):
                if T1[i, j] > 0:
                    T1[i, j] = 1./T1[i, j]

        for i in range(T2.shape[0]):
            for j in range(T2.shape[1]):
                if T2[i, j] > 0:
                    T2[i, j] = 1./T2[i, j]

        for i in range(L):
            self.E1[i] = self.E1[i]*T1
            self.E2[i] = self.E2[i]*T2

        # Normalize node feature vectors
        self.N1 = normalize(self.N1)
        self.N2 = normalize(self.N2)
        # Compute node feature cosine cross-similarity
        n1 = self.A1.shape[0]
        n2 = self.A2.shape[0]
        N = np.zeros(n1 * n2)
        
        for k in range(K):
            N += np.kron(self.N1[:, k], self.N2[:, k])

        # Compute the Kronecker degree vector
        d = np.zeros_like(N)
        for i in range(L):
            for k in range(K):
                d += np.kron(np.dot(self.E1[i]*self.A1, self.N1[:, k]), np.dot(self.E2[i]*self.A2, self.N2[:, k]))

        D = N*d
        DD = 1./np.sqrt(D)
        DD[DD == inf] = 0

        # fixed-point solution
        q = DD*N 
        h = self.H.flatten('F')
        s = h 

        for i in range(self.maxiter):
            print("iterations ", i + 1)
            prev = s
            M = (q*s).reshape((n2, n1), order='F' )
            S = np.zeros((n2, n1))
            for l in range(L):
                S += np.dot(np.dot(self.E2[l]*self.A2, M), self.E1[l]*self.A1)
            s = (1- self.alpha) * h + self.alpha * q * S.flatten('F')
            diff = np.sqrt(np.sum((s - prev)**2))
            print(diff)
            if diff < self.tol:
                break

        self.alignment_matrix = s.reshape((n1, n2))         
        return self.alignment_matrix

    def get_alignment_matrix(self):
        if self.alignment_matrix is None:
            raise Exception("Must calculate alignment matrix by calling 'align()' method first")
        return self.alignment_matrix
    
    def get_H(self, adjA, adjB):
        H = np.ones((adjA.shape[0], adjB.shape[0]))
        H = H*(1/adjB.shape[0])
        return H
