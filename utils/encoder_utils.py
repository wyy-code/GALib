import numpy as np
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree


def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)

    row = np.array([])
    col = np.array([])
    data = np.array([])

    dist, ind = kd_tree.query(emb1, k=num_top)
    # print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix(
        (data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    return sparse_align_matrix.tocsr()