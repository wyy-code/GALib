import numpy as np
import scipy as sci
import networkx as nx
import time
import lapjv
import os
from sklearn.preprocessing import normalize
import argparse
import math
import matplotlib.pyplot as plt
#import base_align as ba
#import munkres
from . import base_align_pymanopt as ba
from encoder.graph_alignment_model import GraphAlignmentModel
#import base_align as ba

from sklearn.neighbors import NearestNeighbors
# np.set_printoptions(precision=3)

# np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})


# 	#noise_level= [1]
# 	scores=np.zeros([5,reps])
# 	for i in noise_level:
# 		for j in range(1,reps+1):
#
#
# 			print('Noise level %d, round %d' %(i,j))


class Grasp(GraphAlignmentModel):
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

    def functional_maps_base_align(self, A1, A2, q, k, laa, lower_t, upper_t, linsteps):

        # 1:nn 2:sortgreedy 3: jv
        match = laa

        t = np.linspace(lower_t, upper_t, q)
        if (not linsteps):
            t = np.logspace(math.log(lower_t, 10), math.log(upper_t, 10), q)

        n = np.shape(A1)[0]

        start = time.time()
        D1, V1 = self.decompose_laplacian(A1)

        # # # #
        D2, V2 = self.decompose_laplacian(A2)

        Cor1 = self.calc_corresponding_functions(n, q, t, D1, V1)
        # print(Cor1)
        Cor2 = self.calc_corresponding_functions(n, q, t, D2, V2)
        # print(np.abs(Cor2-Cor1))

        B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k)

        V1_rot = V1[:, 0:k]
        V2_rot = V2[:, 0:k] @ B
        C = self.calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1_rot, V2_rot, k, q)
        # print(np.diagonal(C))

        G1_emb = V1_rot.T  # [:, 0: k].T;
        G2_emb = C @ V2_rot.T  # [:, 0: k].T;

        return G1_emb, G2_emb



    def functional_maps(self, A1, A2, q, k, laa, lower_t, upper_t, linsteps):

        # 1:nn 2:sortgreedy 3: jv
        match = laa

        t = np.linspace(lower_t, upper_t, q)
        if(not linsteps):
            t = np.logspace(lower_t, upper_t, q)

        n = np.shape(A1)[0]
        print(n)

        D1, V1 = self.decompose_laplacian(A1)
    # print(V1[:,0])
    #   D1, V1 = decompose_laplacian(A1)
    # # #  print(V1[20,200])
        D2, V2 = self.decompose_laplacian(A2)
    #   print(V2[20, 200])

        Cor1 = self.calc_corresponding_functions(n, q, t, D1, V1)
        Cor2 = self.calc_corresponding_functions(n, q, t, D2, V2)

    #    print(Cor1[20,25])
    #    print(Cor2[20, 25])

        # rotV1, rotV2=calc_rotation_matrices(Cor1, Cor2, V1, V2,k)
        # #
        # plt.imshow(rotV1[0:25,0:25])
        # plt.show()
        # V1=V1[:,0:k]@rotV1
        # V2 = V2[:,0:k] @ rotV2

        A = self.calc_coefficient_matrix(Cor1, V1, k, q)

        B = self.calc_coefficient_matrix(Cor2, V2, k, q)
    #    print(A[20,25])
    #    print(B[20,25])
        # C=calc_correspondence_matrix(A,B,k)

        # C=calc_correspondence_matrix_ortho(A,B,k)
        C = self.calc_C_as_in_quasiharmonicpaper(
            Cor1, Cor2, V1[:, 0:k], V2[:, 0:k], k, q)
        # plt.imshow(C)
        # plt.show()
        # print(np.diagonal(C))

        G1_emb = C @ V1[:, 0: k].T
        G2_emb = V2[:, 0: k].T

        return G1_emb, G2_emb

        # matching = []
        # if(icp):
        #     matching = iterative_closest_point(
        #         V1, V2, C, icp_its, k, match, Cor1, Cor2, q)
        # else:
        #     if match == 1:
        #         matching = greedyNN(G1_emb, G2_emb)
        #     if match == 2:
        #         matching = sort_greedy(G1_emb, G2_emb)
        #     if match == 3:
        #         matching = hungarian_matching(G1_emb, G2_emb)

        # matching = dict(matching.astype(int))

        # return matching, 0, V1, V2


    def edgelist_to_adjmatrix(self, edgeList_file):

        edge_list = np.loadtxt(edgeList_file, usecols=range(2))

        n = int(np.amax(edge_list)+1)
        #n = int(np.amax(edge_list))
    # print(n)

        e = np.shape(edge_list)[0]

        a = np.zeros((n, n))

        # make adjacency matrix A1

        for i in range(0, e):
            n1 = int(edge_list[i, 0])  # - 1

            n2 = int(edge_list[i, 1])  # - 1

            a[n1, n2] = 1.0
            a[n2, n1] = 1.0

        return a


    def decompose_laplacian(self, A):

        #  adjacency matrix

        Deg = np.diag((np.sum(A, axis=1)))

        n = np.shape(Deg)[0]

        Deg = sci.linalg.fractional_matrix_power(Deg, -0.5)

        L = np.identity(n) - Deg @ A @ Deg
    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
        # '[V1, D1] = eig(L1);

        D, V = np.linalg.eigh(L)

        return [D, V]


    def decompose_unnormalized_laplacian(self, A):

        #  adjacency matrix

        Deg = np.diag((np.sum(A, axis=1)))

        n = np.shape(Deg)[0]

        L = Deg - A

    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
        # '[V1, D1] = eig(L1);

        D, V = np.linalg.eig(L)

        return [D, V]


    def decompose_rw_normalized_laplacian(self, A):

        #  adjacency matrix

        Deg = np.diag((np.sum(A, axis=1)))

        n = np.shape(Deg)[0]

        L = np.identity(n) - np.linalg.inv(Deg) @ A

    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
        # '[V1, D1] = eig(L1);

        D, V = np.linalg.eig(L)

        return [D, V]


    def decompose_rw_laplacian(self, A):

        #  adjacency matrix

        Deg = np.diag((np.sum(A, axis=1)))

        n = np.shape(Deg)[0]

        L = np.linalg.inv(Deg) @ A

    # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
        # '[V1, D1] = eig(L1);

        D, V = np.linalg.eig(L)

        return [D, V]


    def calc_corresponding_functions(self, n, q, t, d, V):

        # corresponding functions are the heat kernel diagonals in each time step
        # t= time steps, d= eigenvalues, V= eigenvectors, n= number of nodes, q= number of corresponding functions
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
        # for i in range(0,k):
        # print(np.shape(C))
        # print(C)
        # print('\n')
        # print(C_normalized)

        # print(np.sum(C_normalized,axis=1))
        #print(np.sum(C, axis=1))
        # return C_normalized
        return C_normalized


    def nearest_neighbor_matching(self, G1_emb, G2_emb):
        n = np.shape(G1_emb)[1]
        nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(G1_emb.T)
        distances, indices = nbrs.kneighbors(G2_emb.T)
        indices = np.c_[np.linspace(0, n-1, n).astype(int), indices.astype(int)]
        return indices
    #


    def hungarian_matching(self, G1_emb, G2_emb):
        print('hungarian_matching: calculating distance matrix')

        dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
        n = np.shape(dist)[0]
        # print(np.shape(dist))
        print('hungarian_matching: calculating matching')
        cols, rows, _ = lapjv.lapjv(dist)
        print('fertig')
        matching = np.c_[cols, np.linspace(0, n-1, n).astype(int)]
        matching = matching[matching[:, 0].argsort()]
        return matching.astype(int)


    def iterative_closest_point(self, V1, V2, C, it, k, match, Cor1, Cor2, q):
        G1 = V1[:, 0: k].T
        G2_emb = V2[:, 0: k].T
        n = np.shape(G2_emb)[1]

        for i in range(0, it):

            print('icp iteration '+str(i))
            G1_emb = C@V1[:, 0:k].T
        # print('calculating hungarian in icp')
            M = []

            if (match == 1):
                M = self.nearest_neighbor_matching(G1_emb, G2_emb)
            if match == 2:
                M = self.sort_greedy(G1_emb, G2_emb)
            if match == 3:
                M = self.hungarian_matching(G1_emb, G2_emb)
            G2_cur = np.zeros([k, n])
        ## print('finding nearest neighbors in eigenvector matrix icp')
            for j in range(0, n):

                G2idx = M[j, 1]
                G2_cur[:, G2idx] = G2_emb[:, j]
        ## print('calculating correspondence matrix in icp')
            # C=calc_correspondence_matrix(G1,G2_cur,k)
            C = self.calc_correspondence_matrix_ortho(G1, G2_cur, k)
            #calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
            #C = calc_correspondence_matrix_ortho_diag(G1, G2_cur, k)
            C_show = C
            # C_show[C_show < 0.13] = 0.0
        # plt.imshow(np.abs(C_show))
        # plt.show()

        # print('calculated correspondence matrix in icp')
        # print('\n')
        G1_emb = C@V1[:, 0:k].T

        if (match == 1):
            M = self.nearest_neighbor_matching(G1_emb, G2_emb)
        if match == 2:
            M = self.sort_greedy(G1_emb, G2_emb)
        if match == 3:
            M = self.hungarian_matching(G1_emb, G2_emb)

        return dict(M.astype(int))


    def greedyNN(self, G1_emb, G2_emb):
        print('greedyNN: calculating distance matrix')

        dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
        n = np.shape(dist)[0]
        # print(np.shape(dist))
        print('greedyNN: calculating matching')
        idx = np.argsort(dist, axis=0)
        matching = np.ones([n, 1])*(n+1)
        for i in range(0, n):
            matched = False
            cur_idx = 0
            while(not matched):
                # print([cur_idx,i])
                if(not idx[cur_idx, i] in matching):
                    matching[i, 0] = idx[cur_idx, i]

                    matched = True
                else:
                    cur_idx += 1
                    # print(cur_idx)

        matching = np.c_[np.linspace(0, n-1, n).astype(int), matching]
        return matching.astype(int)


    def sort_greedy(self, G1_emb, G2_emb):
        print('sortGreedy: calculating distance matrix')

        dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
        n = np.shape(dist)[0]
        # print(np.shape(dist))
        print('sortGreedy: calculating matching')
        dist_platt = np.ndarray.flatten(dist)
        idx = np.argsort(dist_platt)
        k = idx//n
        r = idx % n
        idx_matr = np.c_[k, r]
    # print(idx_matr)
        G1_elements = set()
        G2_elements = set()
        i = 0
        j = 0
        matching = np.ones([n, 2])*(n+1)
        while(len(G1_elements) < n):
            if (not idx_matr[i, 0] in G1_elements) and (not idx_matr[i, 1] in G2_elements):
                # print(idx_matr[i,:])
                matching[j, :] = idx_matr[i, :]

                G1_elements.add(idx_matr[i, 0])
                G2_elements.add(idx_matr[i, 1])
                j += 1
                # print(len(G1_elements))

            i += 1

    # print(idx)
        matching = np.c_[matching[:, 1], matching[:, 0]]
        matching = matching[matching[:, 0].argsort()]
        return matching.astype(int)


    def top_x_matching(self, G1_emb, G2_emb, x):
        dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
        n = np.shape(dist)[0]

        idx = np.argsort(dist, axis=0)

        matches = idx[0:x, :]

        matching = np.c_[np.linspace(0, n-1, n).astype(int), matches.T]
        #matching = matching[matching[:, 0].argsort()]
        return matching.astype(int)


    def eval_top_x_matching(self, matching, gt):
        n = float(len(gt))
        nn = len(gt)
        acc = 0.0
        for i in range(0, nn):
            if i in gt and np.isin(gt[i], matching[i, :]):
                acc += 1.0
        # print(acc/n)
        return acc / n


    def eval_matching(self, matching, gt):

        n = float(len(gt))
        acc = 0.0
        for i in matching:
            if i in gt and matching[i] == gt[i]:
                acc += 1.0
    # print(acc/n)
        return acc/n


    def read_regal_matrix(self, file):

        nx_graph = nx.read_edgelist(file, nodetype=int, comments="%")
        A = nx.adjacency_matrix(nx_graph)
        n = int(np.shape(A)[0]/2)
        A1 = A[0:n, 0:n]
        A2 = A[n:2*n, n:2*n]
        return A1.todense(), A2.todense()


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


    def align(self):

        Src = self.adjA
        Tar = self.adjB

        if self.n_eig is None:
            self.n_eig = Src.shape[0] - 1

        if self.base_align:
            G1_emb, G2_emb = self.functional_maps_base_align(Src, Tar, q=self.q, k=self.k, laa=self.laa, \
                                                             lower_t=self.lower_t, upper_t=self.upper_t, linsteps=self.linsteps)
        else:
            G1_emb, G2_emb = self.functional_maps(Src, Tar, q=self.q, k=self.k, laa=self.laa, \
                                                             lower_t=self.lower_t, upper_t=self.upper_t, linsteps=self.linsteps)

        return sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
