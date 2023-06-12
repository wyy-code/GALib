import scipy
import numpy as np


def bipartite_matching(A, nzi, nzj, nzv):
    # return bipartite_matching_primal_dual(bipartite_matching_setup(A,nzi,nzj,nzv))
    rp, ci, ai, tripi, m, n = bipartite_matching_setup(A, nzi, nzj, nzv)
    # print("hi7")
    return bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n)


def bipartite_matching_primal_dual(rp, ci, ai, tripi, m, n):
    # print(m, n)
    # print(rp.tolist())
    # print(ci.tolist())
    # print(ai.tolist())
    # print(tripi.tolist())
    # variables used for the primal-dual algorithm
    # normalize ai values # updated on 2-19-2019
    ai = ai/np.amax(abs(ai))
    alpha = np.zeros(m)
    bt = np.zeros(m+n)  # beta
    queue = np.zeros(m, int)
    t = np.zeros(m+n, int)
    match1 = np.zeros(m, int)
    match2 = np.zeros(m+n, int)
    tmod = np.zeros(m+n, int)
    ntmod = 0

    # initialize the primal and dual variables

    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if ai[rpi] > alpha[i]:
                alpha[i] = ai[rpi]

    # dual variables (bt) are initialized to 0 already
    # match1 and match2 are both 0, which indicates no matches

    i = 1
    while i < m:
        for j in range(1, ntmod+1):
            t[tmod[j]] = 0
        ntmod = 0
        # add i to the stack
        head = 1
        tail = 1
        queue[head] = i
        while head <= tail and match1[i] == 0:
            k = queue[head]
            for rpi in range(rp[k], rp[k+1]):
                j = ci[rpi]
                if ai[rpi] < alpha[k] + bt[j] - 1e-8:
                    continue

                if t[j] == 0:
                    tail = tail+1
                    if tail < m:
                        queue[tail] = match2[j]
                    t[j] = k
                    ntmod = ntmod+1
                    tmod[ntmod] = j
                    if match2[j] < 1:
                        while j > 0:
                            match2[j] = t[j]
                            k = t[j]
                            temp = match1[k]
                            match1[k] = j
                            j = temp
                        break
            head = head+1
        if match1[i] < 1:
            theta = np.inf
            for j in range(1, head):
                t1 = queue[j]
                for rpi in range(rp[t1], rp[t1+1]):
                    t2 = ci[rpi]
                    if t[t2] == 0 and alpha[t1] + bt[t2] - ai[rpi] < theta:
                        theta = alpha[t1] + bt[t2] - ai[rpi]
            for j in range(1, head):
                alpha[queue[j]] -= theta
            for j in range(1, ntmod+1):
                bt[tmod[j]] += theta
            continue
        i = i+1
        # print(f"{i} < {m}")
    val = 0
    # print("po")
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if ci[rpi] == match1[i]:
                val = val+ai[rpi]
    noute = 0
    for i in range(1, m):
        if match1[i] <= n:
            noute = noute+1
    return m, n, val, noute, match1


def bipartite_matching_setup(A, nzi, nzj, nzv, m=None, n=None):
    # (nzi,nzj,nzv) = bipartite_matching_setup_phase1(A,nzi,nzj,nzv)
    if A is not None:
        (nzi, nzj, nzv) = bipartite_matching_setup_phase1(A)
        (m, n) = np.shape(A)
        m = m+1  # ?
        n = n+1  # ?
    if m is None:
        m = max(nzi) + 1
    if n is None:
        n = max(nzj) + 1
    # print(nzi)
    # print(nzj)
    # print(nzv)
    # print(m, n)
    # print("hi-setup")
    nedges = len(nzi)
    rp = np.ones(m+2, int)  # csr matrix with extra edges
    ci = np.zeros(nedges+m, int)
    ai = np.zeros(nedges+m)
    tripi = np.zeros(nedges+m, int)

    rp[0] = 0
    rp[1] = 0
    for i in range(1, nedges):
        rp[nzi[i]+1] = rp[nzi[i]+1]+1
    rp = np.cumsum(rp)

    for i in range(1, nedges):
        tripi[rp[nzi[i]]+1] = i
        ai[rp[nzi[i]]+1] = nzv[i]
        ci[rp[nzi[i]]+1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]]+1

    for i in range(1, m+1):  # add the extra edges
        tripi[rp[i]+1] = -1
        ai[rp[i]+1] = 0
        ci[rp[i]+1] = n+i
        rp[i] = rp[i]+1

    # restore the row pointer array
    for i in range(m, 0, -1):
        rp[i+1] = rp[i]
    rp[1] = 0
    rp = rp+1

    # check for duplicates in the data
    colind = np.zeros(m+n, int)
    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]):
            if colind[ci[rpi]] == 1:
                print("bipartite_matching:duplicateEdge")
        colind[ci[rpi]] = 1

        for rpi in range(rp[i], rp[i+1]):
            colind[ci[rpi]] = 0
    return rp, ci, ai, tripi, m, n


def bipartite_matching_setup_phase1(A, nzi, nzj, nzv):
    temp = len(nzi)+1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i-1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = nzv[i - 1]
    nzi1 = nzi1+1
    nzj1 = nzj1+1
    return (nzi1, nzj1, nzv1)


def bipartite_matching_setup_phase1(A):
    nzi, nzj = scipy.sparse.csr_matrix.nonzero(A)
    temp = len(nzi) + 1
    nzi1 = np.zeros(temp, int)
    nzj1 = np.zeros(temp, int)
    nzv1 = np.zeros(temp, float)
    for i in range(1, temp):
        nzi1[i] = nzi[i - 1]
        nzj1[i] = nzj[i - 1]
        nzv1[i] = A[nzi1[i], nzj1[i]]
    nzi1 = nzi1 + 1
    nzj1 = nzj1 + 1

    return (nzi1, nzj1, nzv1)


def edge_list(m, n, weight, cardinality, match):
    m1 = np.zeros(cardinality, int)
    m2 = np.zeros(cardinality, int)
    noute = 0
    for i in range(1, m):
        if match[i] <= n:
            m1[noute] = i
            m2[noute] = match[i]
            noute = noute+1
    return m1, m2


def matching_indicator(rp, ci, match1, tripi, m, n):
    mi = np.zeros(len(tripi)-m, int)
    for i in range(1, m+1):
        for rpi in range(rp[i], rp[i+1]):
            if match1[i] <= n and ci[rpi] == match1[i]:
                mi[tripi[rpi]] = 1
    return mi


def bipartite_matching_setupc(x, ei, ej, m, n):
    (nzi, nzj, nzv) = (ei, ej, x)
    nedges = len(nzi)

    rp = np.ones(m+1, int)  # csr matrix with extra edges
    ci = np.zeros(nedges+m, int)
    ai = np.zeros(nedges+m)
    tripi = np.zeros(nedges+m, int)
    # 1. build csr representation with a set of extra edges from vertex i to
    # vertex m+i
    rp[0] = 0
    rp[1] = 0
    for i in range(1, nedges):
        rp[nzi[i]+1] = rp[nzi[i]+1]+1

    rp = np.cumsum(rp)
    for i in range(1, nedges):
        tripi[rp[nzi[i]]+1] = i
        ai[rp[nzi[i]]+1] = nzv[i]
        ci[rp[nzi[i]]+1] = nzj[i]
        rp[nzi[i]] = rp[nzi[i]]+1
    for i in range(1, m):  # add the extra edges
        tripi[rp[i]+1] = -1
        ai[rp[i]+1] = 0
        ci[rp[i]+1] = n+i
        rp[i] = rp[i]+1

    # restore the row pointer array
    for i in range(m, 0, -1):
        rp[i+1] = rp[i]
    rp[1] = 0
    rp = rp+1
    rp[0] = 0

    # check for duplicates in the data
    colind = np.zeros(m+n)

    for i in range(1, m):
        for rpi in range(rp[i], rp[i+1]-1):
            if colind[ci[rpi]] == 1:
                print("bipartite_matching:duplicateEdge")
        colind[ci[rpi]] = 1

        for rpi in range(rp[i], rp[i+1]-1):
            colind[ci[rpi]] = 0

    return rp, ci, ai, tripi, m, n


# def test1():
#     rp = [1, 1, 7, 13, 19, 25, 31, 37]
#     ci = [0, 1, 2, 3, 4, 5, 6, 1, 2, 3, 4, 5, 7, 1, 2, 3, 4, 5,
#           8, 1, 2, 3, 4, 5, 9, 1, 2, 3, 4, 5, 10, 1, 2, 3, 4, 5, 11]
#     ai = [0.0, 2.8, 0.19999999999999996, 0.5, 0.5, 0.5, 0.0, 3.3, 1.0, 0.5, 0.5, 1.8, 0.0, 0.4999999999999999, 1.4, 0.5, 0.9,
#           0.5, 0.0, 2.5, 1.0, 0.9, 0.5, 0.5, 0.0, 2.5, 0.9, 0.5, 0.09999999999999998, 0.5, 0.0, 1.4, 0.5, 0.5, 0.5, 0.5, 0.0]
#     # ai = [0.0, 2.8, 0.2, 0.5, 0.5, 0.5, 0.0, 3.3, 1.0, 0.5, 0.5, 1.8, 0.0, 0.5, 1.4, 0.5, 0.9,
#     #       0.5, 0.0, 2.5, 1.0, 0.9, 0.5, 0.5, 0.0, 2.5, 0.9, 0.5, 0.1, 0.5, 0.0, 1.4, 0.5, 0.5, 0.5, 0.5, 0.0]
#     tripi = [0, 1, 7, 13, 19, 25, -1, 2, 8, 14, 20, 26, -1, 3, 9, 15, 21, 27, -
#              1, 4, 10, 16, 22, 28, -1, 5, 11, 17, 23, 29, -1, 6, 12, 18, 24, 30, -1]

#     print(bipartite_matching_primal_dual(np.array(rp), np.array(
#         ci), np.array(ai), np.array(tripi), 7, 6))

#     #(7, 6, 7.4, 5, array([ 0,  1,  5,  2,  3, 10,  4]))


# if __name__ == "__main__":
#     test1()
