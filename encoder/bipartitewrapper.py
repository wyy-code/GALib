try:
    from . import bipartiteMatching as bm
except Exception:
    import bipartiteMatching as bm
import numpy as np
import scipy.sparse as sps

# def debugm(U):
#     uu = np.array(sps.find(U), float).T
#     debug(uu[:, uu[1].argsort()])


# def debug(arr):
#     print("###")
#     print(arr[:51])
#     print(arr[-50:])
#     print("$$$")

def to_matlab(array, diff=1):
    res = array.copy()
    res += diff
    return np.insert(res, [0], [0])


def to_python(array, diff=1):
    res = array.copy()
    res -= diff
    return np.delete(res, [0])


def bipartite_setup(li, lj, w):

    # print(li.tolist())
    # print(lj.tolist())
    # print(w.tolist())

    m = max(li) + 1
    n = max(lj) + 1

    rp, ci, ai, tripi = bm.bipartite_matching_setup(
        None, to_matlab(li), to_matlab(lj), to_matlab(w, 0), m, n)[:4]

    mperm = tripi[tripi > 0]

    return (rp, ci, ai, tripi, mperm), m, n


def round_messages(messages, S, w, alpha, beta, setup, m, n):
    # print(messages.size)
    # print(messages)
    rp, ci, _, tripi, mperm = setup

    ai = np.zeros(len(tripi))
    ai[tripi > 0] = messages[mperm-1]
    _, _, val, _, match1 = bm.bipartite_matching_primal_dual(
        rp, ci, ai, tripi, m+1, n+1)

    mi = bm.matching_indicator(rp, ci, match1, tripi, m, n)[1:]

    matchweight = sum(w[mi > 0])
    # matchweight = np.dot(mi, w)
    cardinality = sum(mi)

    overlap = np.dot(mi.T, (S*mi))/2
    f = alpha*matchweight + beta*overlap

    return f, matchweight, int(cardinality), int(overlap), val, mi


def getmatchings(matrix):
    coo = sps.coo_matrix(matrix)
    li = coo.row
    lj = coo.col
    lv = coo.data

    m, n, val, noute, match1 = bm.bipartite_matching(
        None, to_matlab(li), to_matlab(lj), to_matlab(lv, 0))
    ma, mb = bm.edge_list(m, n, val, noute, match1)

    return ma - 1, mb - 1


# if __name__ == "__main__":
#     li = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7,
#           8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
#     lj = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6,
#           6, 6, 7, 7, 7, 7, 7, 7, 7, 7, 7, 7, 8, 8, 8, 8, 8, 8, 8, 8, 8, 8, 9, 9, 9, 9, 9, 9, 9, 9, 9, 9]
#     w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,
#          1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

#     L = sps.csr_matrix((w, (li, lj)))

#     print(L.A)

#     (rp, ci, ai, tripi, mperm), m, n = bipartite_setup(
#         np.array(li, int), np.array(lj, int), np.array(w))

#     print(L.indptr)
#     print(L.indices)
#     print(L.data)
#     print()
#     print(rp)
#     print(ci)
#     print(ai)
#     # print(tripi)
#     # print(mperm)
