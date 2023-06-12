import numpy as np
import scipy
from numpy.linalg import svd
from scipy.linalg import lu
from scipy.linalg._expm_frechet import vec
#original code https://github.com/nassarhuda/lowrank_spectral
from . import decomposeX, newbound_methods
from encoder.network_alignment_model import NetworkAlignmentModel
# from .. import bipartiteMatching
from .. import bipartitewrapper as bmw

class LREA(NetworkAlignmentModel):
    def __init__(self, adjA, adjB, iters=20, method="lowrank_svd_union", bmatch=3):
        """
 
        """
        self.adjA = adjA
        self.adjB = adjB
        self.iters = iters
        self.method = method
        self.bmatch = bmatch

    def find_parameters(self, A, B):
        nB = len(B[0])
        nA = len(A[0])
        myalpha = (nB ** 2 - np.sum(B)) / np.sum(B) + \
            (nA ** 2 - np.sum(A)) / np.sum(A)+1
        myeps = 0.001
        s1 = myalpha + myeps
        s2 = 1 + myeps
        s3 = myeps
        return s1, s2, s3
    
    def split_balanced_decomposition(self, Uk, Wk, Vk):
        P, L, U = scipy.linalg.lu(Wk, False)
        Ud = np.diag(np.sqrt(abs(np.diag(U))))
        L2 = np.dot(L, Ud)
        Utemp = np.sqrt(np.diag(U))
        Utemp2 = np.divide(1, Utemp)
        U2 = np.dot(np.diag(Utemp2), U)
        Un = np.dot(Uk, L2)
        Vn = np.dot(Vk, U2.transpose())

        return Un, Vn
    
    def split_svd(self, Uk, Wk, Vk):
        U, S, V = svd(Wk)
        D = np.diag(np.sqrt(S))
        Unew = Uk @ U @ D
        Vnew = Vk @ V @ D
        return Unew, Vnew
    
    def align(self, default_params=True):

        Src = self.adjA
        Tar = self.adjB

        # D = 0
        s1, s2, s3 = self.find_parameters(Src, Tar)

        if not default_params:
            s1 += 100
            s2 += 10
            s3 += 5
        c1 = s1 + s2 - 2 * s3
        c2 = s3 - s2
        c3 = s2
        Uk, Vk, Wk, W1, W2 = decomposeX.decomposeX_balance_allfactors(
            Src, Tar, self.iters + 1, c1, c2, c3)  # okay
        Un, Vn = self.split_balanced_decomposition(Uk, Wk, Vk)  # okay
        timematching = 0
        nA = len(Src[0])
        nB = len(Tar[0])

        if self.method == "lowrank_svd_union":

            U, S, Vtemp = np.linalg.svd(Wk)
            V = Vtemp.transpose()
            U1 = np.dot(np.dot(Uk, U), np.diag(np.sqrt(S)))
            V1 = np.dot(np.dot(Vk, V), np.diag(np.sqrt(S)))
            # X = newbound_methods.newbound_rounding_lowrank_evaluation_relaxed(U1, V1, bmatch) * (10 ** 8)  # 1
            X, nzi, nzj, nzv = newbound_methods.newbound_rounding_lowrank_evaluation_relaxed(
                U1, V1, self.bmatch)  # alternative
            # nzv = nzv * (10 ** 8)  # alternative
            # X = X * (10 ** 8)
            # X1 = X.toarray()
            # avgdeg = map(lambda x: sum(X1[x, :] != 0),
            #              np.arange(0, np.shape(X1)[0], 1))  # keep1
            # np.fromiter(avgdeg, dtype=np.float)#keep1
            # avgdeg = np.array(list(avgdeg))
            # avgdeg = np.mean(avgdeg)  # keep1
            # Matching = bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching(X))  # 1

            # m, n, val, noute, match1 = (
            #     bipartiteMatching.bipartite_matching(X, nzi, nzj, nzv))
            # ma, mb = bipartiteMatching.edge_list(m, n, val, noute, match1)

            # ma, mb = bmw.getmatchings(nzi, nzj, nzv)

            # Matching=bipartite_Matching.edge_list(bipartite_Matching.bipartite_matching1(nzi,nzj,nzv))
            # D = avgdeg  # nnz(X)/prod(size(X))
        else:
            print(
                "method should be one of the following: (1)eigenalign,(2)lowrank_unbalanced_best, (3)lowrank_unbalanced_union,(4)lowrank_balanced_best, (5)lowrank_balanced_union,(6)lowrank_Wkdecomposed_best, (7)lowrank_Wkdecomposed_union")
        Sim = scipy.sparse.csr_matrix((nzv, (nzi, nzj))).todense()

        return Sim
    