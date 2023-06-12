import numpy as np
import scipy as sci
import scipy.sparse as sps
import scipy.sparse.linalg as spsl
import networkx as nx
import time
import os
from sklearn.preprocessing import normalize
import argparse
import matplotlib.pyplot as plt
#import base_align as ba
#import munkres
from . import base_align as ba
#import base_align as ba
from encoder.network_alignment_model import NetworkAlignmentModel
from sklearn.neighbors import NearestNeighbors


class Grasp(NetworkAlignmentModel):
    def __init__(self, adjA, adjB, q=20, k=5, laa=1, lower_t=0.01, upper_t=1.0, linsteps=True, base_align=False):
        """
        laa：Linear assignment algorithm. 1=nn,2=sortgreedy, 3=jv
        lower_t：smallest timestep for corresponding functions
        upper_t：biggest timestep for corresponding functions
        linsteps：scaling of time steps of corresponding functions, logarithmically or linearly
        """
        self.adjA = adjA
        self.adjB = adjB
        self.q = q
        self.k =k
        self.laa = laa
        self.lower_t = lower_t
        self.upper_t = upper_t
        self.linsteps =linsteps
        self.n_eig = None
        self.base_align = base_align

    def align(self):
        Src = self.adjA
        Tar = self.adjB

        if self.n_eig is None:
            self.n_eig = Src.shape[0] - 1

        G1_emb, G2_emb = self.functional_maps_gen(Src, Tar, q=self.q, k=self.k, n_eig=self.n_eig, laa=self.laa, \
                                            lower_t=self.lower_t, upper_t=self.upper_t, linsteps=self.linsteps, base_align=self.base_align)

        return sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)


    def functional_maps_gen(self, A1, A2, q, k, n_eig, laa, lower_t, upper_t, linsteps, base_align):

        # 1:nn 2:sortgreedy 3: jv
        match = laa

        t = np.linspace(lower_t, upper_t, q)
        if (not linsteps):
            t = np.logspace(lower_t, upper_t, q)

        # n = np.shape(A1)[0]
        n = A1.shape[0]

        # decompose graph laplacians
        D1, V1 = self.decompose_laplacian(A1, True, n_eig)
        D2, V2 = self.decompose_laplacian(A2, True, n_eig)

        # calculate corresponding functions
        Cor1 = self.calc_corresponding_functions(n, q, t, D1, V1)
        Cor2 = self.calc_corresponding_functions(n, q, t, D2, V2)


        # calculate base alignment matrix
        if base_align:
            B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k)

            # align bases with base alignment matrix
            V1_rot = V1[:, 0:k]
            V2_rot = V2[:, 0:k] @ B

            # calculate correspondence matrix C
            C = self.calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1_rot, V2_rot, k, q)
            # print(np.diagonal(C))

            # use eigenvectors for alignment
            G1_emb = C @ V1_rot.T  # [:, 0: k].T;

            G2_emb = V2_rot.T  # [:, 0: k].T;
        else:
            A = self.calc_coefficient_matrix(Cor1, V1, k, q)

            B = self.calc_coefficient_matrix(Cor2, V2, k, q)

            C = self.calc_correspondence_matrix_ortho(A, B, k)
            #C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
            # print(np.diagonal(C))

            G1_emb = C @ V1[:, 0: k].T

            G2_emb = V2[:, 0: k].T

        return G1_emb, G2_emb


    def adj_to_laplacian(self, mat, normalized):
        """
        Converts a sparse or dence adjacency matrix to Laplacian.

        Parameters
        ----------
        mat : obj
            Input adjacency matrix. If it is a Laplacian matrix already, return it.
        normalized : bool
            Whether to use normalized Laplacian.
            Normalized and unnormalized Laplacians capture different properties of graphs, e.g. normalized Laplacian spectrum can determine whether a graph is bipartite, but not the number of its edges. We recommend using normalized Laplacian.
        Returns
        -------
        obj
            Laplacian of the input adjacency matrix
        Examples
        --------
        >>> mat_to_laplacian(numpy.array([[0, 1, 1], [1, 0, 1], [1, 1, 0]]), False)
        [[ 2, -1, -1], [-1,  2, -1], [-1, -1,  2]]
        """
        if sps.issparse(mat):
            if np.all(mat.diagonal() >= 0):  # Check diagonal
                if np.all((mat-sps.diags(mat.diagonal())).data <= 0):  # Check off-diagonal elements
                    return mat
        else:
            if np.all(np.diag(mat) >= 0):  # Check diagonal
                if np.all(mat - np.diag(mat) <= 0):  # Check off-diagonal elements
                    return mat
        deg = np.squeeze(np.asarray(mat.sum(axis=1)))
        if sps.issparse(mat):
            L = sps.diags(deg) - mat
        else:
            L = np.diag(deg) - mat
        if not normalized:
            return L
        with np.errstate(divide='ignore'):
            sqrt_deg = 1.0 / np.sqrt(deg)
        sqrt_deg[sqrt_deg == np.inf] = 0
        if sps.issparse(mat):
            sqrt_deg_mat = sps.diags(sqrt_deg)
        else:
            sqrt_deg_mat = np.diag(sqrt_deg)
        return sqrt_deg_mat.dot(L).dot(sqrt_deg_mat)


    def decompose_laplacian(self, A, normalized=True, n_eig=100):
        l = self.adj_to_laplacian(A, normalized)
        D, V = spsl.eigsh(l, n_eig, which='SM')
        return [D, V]


    def decompose_rw_normalized_laplacian(self, A):

        #  adjacency matrix

        Deg = np.diag((np.sum(A, axis=1)))

        n = np.shape(Deg)[0]

        L = np.identity(n) - np.linalg.inv(Deg) @ A

        D, V = np.linalg.eig(L)

        return [D, V]


    def calc_corresponding_functions(self, n, q, t, d, V):

        t = t[:, np.newaxis]
        d = d[:, np.newaxis]

        V_square = np.square(V)

        time_and_eigv = np.dot((d), np.transpose(t))

        time_and_eigv = np.exp(-1*time_and_eigv)

        Cores = np.dot(V_square, time_and_eigv)

        return Cores


    def calc_coefficient_matrix(self, Corr, V, k, q):
        coefficient_matrix = np.linalg.lstsq(V[:, 0:k], Corr, rcond=None)
        # print(type(coefficient_matrix))
        return coefficient_matrix[0]


    def calc_correspondence_matrix(self, A, B, k):
        C = np.zeros([k, k])
        At = A.T
        Bt = B.T

        for i in range(0, k):
            C[i, i] = np.linalg.lstsq(
                Bt[:, i].reshape(-1, 1), At[:, i].reshape(-1, 1), rcond=None)[0]

        return C


    def calc_correspondence_matrix_ortho_diag(self, A, B, k):
        C = np.zeros([k, k])
        At = A.T
        Bt = B.T

        for i in range(0, k):
            C[i, i] = np.sign(np.linalg.lstsq(
                Bt[:, i].reshape(-1, 1), At[:, i].reshape(-1, 1), rcond=None)[0])

        return C


    def calc_correspondence_matrix_ortho(self, A, B, k):
        #C = np.zeros([k,k])
        At = A.T
        Bt = B.T

        C = sci.linalg.orthogonal_procrustes(Bt, At)[0]

        C_norms = np.linalg.norm(C)

        C_normalized = normalize(C, axis=1)

        return C_normalized


    def calc_rotation_matrices(self, Cor1, Cor2, V1, V2, k, q):

        rotV1, _, rotV2 = np.linalg.svd(V1[:, 0:k].T@Cor1@Cor2.T@V2[:, 0:k])

        return rotV1, rotV2.T


    def calc_C_as_in_quasiharmonicpaper(self, Cor1, Cor2, V1, V2, k, q):
        leftside = Cor1.T@V1[:, 0:k]
        rightside = V2[:, 0:k].T@Cor2

        left = np.diag(leftside[0, :])

        right = rightside[:, 0]
        for i in range(1, q):
            left = np.concatenate((left, np.diag(leftside[i, :])))
            right = np.concatenate((right, rightside[:, i]))

    # print(np.shape(left))
    # print(np.shape(right))

        C_diag = np.linalg.lstsq(left, right, rcond=None)[0]
    # print(C_diag)
    #  print(np.shape(C_diag))
        return np.diag(C_diag)