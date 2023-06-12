import numpy as np
import scipy.sparse as sps
from math import floor, log2
from encoder.network_alignment_model import NetworkAlignmentModel
#original code from https://github.com/nassarhuda/NetworkAlignment.jl/blob/master/src/NSD.jl

class NSD(NetworkAlignmentModel):
    def __init__(self, adjA, adjB, alpha=0.8, iters=20, weighted=True):
        """
 
        """
        self.adjA = adjA
        self.adjB = adjB
        self.alpha = alpha
        self.iters = iters
        self.weighted = weighted

    def create_L(self, A, B, lalpha=1, mind=None, weighted=True):
        n = A.shape[0]
        m = B.shape[0]

        if lalpha is None:
            return sps.csr_matrix(np.ones((n, m)))

        a = A.sum(1)
        b = B.sum(1)
        # print(a)
        # print(b)

        # a_p = [(i, m[0,0]) for i, m in enumerate(a)]
        a_p = list(enumerate(a))
        a_p.sort(key=lambda x: x[1])

        # b_p = [(i, m[0,0]) for i, m in enumerate(b)]
        b_p = list(enumerate(b))
        b_p.sort(key=lambda x: x[1])

        ab_m = [0] * n
        s = 0
        e = floor(lalpha * log2(m))
        for ap in a_p:
            while(e < m and
                abs(b_p[e][1] - ap[1]) <= abs(b_p[s][1] - ap[1])
                ):
                e += 1
                s += 1
            ab_m[ap[0]] = [bp[0] for bp in b_p[s:e]]

        # print(ab_m)

        li = []
        lj = []
        lw = []
        for i, bj in enumerate(ab_m):
            for j in bj:
                # d = 1 - abs(a[i]-b[j]) / a[i]
                d = 1 - abs(a[i]-b[j]) / max(a[i], b[j])
                if mind is None:
                    if d > 0:
                        li.append(i)
                        lj.append(j)
                        lw.append(d)
                else:
                    li.append(i)
                    lj.append(j)
                    lw.append(mind if d <= 0 else d)
                    # lw.append(0.0 if d <= 0 else d)
                    # lw.append(d)

                    # print(len(li))
                    # print(len(lj))
                    # print(len(lj))

        return sps.csr_matrix((lw, (li, lj)), shape=(n, m))

    def normout_rowstochastic(self, P):
        n = P.shape[0]
        # colsums = sum(P, 1)-1
        colsums = np.sum(P, axis=0)
        # pi, pj, pv = findnz_alt(P)
        pi, pj, pv = sps.find(P)
        pv = np.divide(pv, colsums[pi])
        Q = sps.csc_matrix((pv, (pi, pj)), shape=(n, n)).toarray()
        return Q


    def nsd(self, A, B, alpha, iters, Zvecs, Wvecs):
        # dtype = np.float16
        dtype = np.float32
        # dtype = np.float64

        A = self.normout_rowstochastic(A).T.astype(dtype)
        B = self.normout_rowstochastic(B).T.astype(dtype)
        Zvecs = Zvecs.astype(dtype=dtype)
        Wvecs = Wvecs.astype(dtype=dtype)
        nB = np.shape(B)[0]
        nA = np.shape(A)[0]

        Sim = np.zeros((nA, nB), dtype=dtype)
        for i in range(np.shape(Zvecs)[1]):
            print(f"<{i}>")
            z = Zvecs[:, i]
            w = Wvecs[:, i]
            z = z / sum(z)
            w = w / sum(w)
            Z = np.zeros((iters + 1, nA), dtype=dtype)  # A
            W = np.zeros((iters + 1, nB), dtype=dtype)  # B
            W[0] = w
            Z[0] = z

            print("dots")
            for k in range(1, iters + 1):
                print(k)
                np.dot(A, Z[k-1], out=Z[k])
                np.dot(B, W[k-1], out=W[k])

            factor = 1.0 - alpha
            print("krons")
            for k in range(iters + 1):
                print(k)

                if k == iters:
                    W[iters] *= alpha ** iters
                    # Z[iters] *= alpha ** iters
                else:
                    W[k] *= factor
                    # Z[k] *= factor
                    factor *= alpha

                # Sim += np.kron(Z[k], W[k]).reshape(nA, nB)

                # Sim += np.dot(
                #     Z[k].reshape(-1, 1), W[k].reshape(1, -1)
                # )

                # for i, w in enumerate(W[k]):
                #     Sim[:, i] += Z[k] * w

                intervals = 4
                for i in range(intervals):
                    start = i * nA // intervals
                    end = (i+1) * nA // intervals
                    Sim[:, start:end] += np.dot(
                        Z[k].reshape(-1, 1), W[k][start:end].reshape(1, -1)
                    )

        return Sim
    
    def align(self):
        Sim = self.nsd(
        self.adjA, self.adjB, self.alpha, self.iters,
        np.ones((self.adjA.shape[0], 1)),
        np.ones((self.adjB.shape[0], 1)),
        # np.ones(Src.shape),
        # np.ones(Tar.shape),
        # L,
        # L.T,
    )
        return Sim

