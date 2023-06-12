import numpy as np
import sklearn.metrics.pairwise
import scipy.sparse as sps
import argparse
import time
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
from encoder.network_alignment_model import NetworkAlignmentModel
# from data import ReadFile
from . import unsup_align, embedding
#original code from https://github.com/GemsLab/CONE-Align

class CONE(NetworkAlignmentModel):
    def __init__(self, adjA, adjB, dim=64,window=10,negative=1.0,niter_init=10,reg_init=1.0, \
                 lr=1.0,bsz=10,nepoch=5,embsim="euclidean",numtop=10,reg_align=0.05,niter_align=10):
        """
        data1: object of Dataset class, contains information of source network
        data2: object of Dataset class, contains information of target network
        lamb: lambda
        """
        self.adjA = adjA
        self.adjB = adjB
        self.niter_init = niter_init
        self.niter_align = niter_align
        self.reg_init = reg_init
        self.lr = lr
        self.bsz = bsz
        self.nepoch = nepoch
        self.embsim = embsim
        self.numtop = numtop
        self.reg_align = reg_align
        self.emb_matrixA = self.get_embed(adjA,dim,window,negative)
        self.emb_matrixB = self.get_embed(adjB,dim,window,negative)

    def get_embed(self,adj,dim,window,negative):
        emb_matrix = embedding.netmf(adj, dim=dim, window=window, b=negative, normalize=True)
        return emb_matrix
    
    
    def align(self):

        # Convex Initialization
        corr = None
        if self.adjA is not None and self.adjB is not None:
            if not sps.issparse(self.adjA):
                adj1 = sps.csr_matrix(self.adjA)
            if not sps.issparse(self.adjB):
                adj2 = sps.csr_matrix(self.adjB)
            init_sim, corr_mat = unsup_align.convex_init_sparse(
                self.emb_matrixA, self.emb_matrixB, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=self.niter_init, reg=self.reg_init, P=corr)
        else:
            init_sim, corr_mat = unsup_align.convex_init(
                self.emb_matrixA, self.emb_matrixB, apply_sqrt=False, niter=self.niter_init, reg=self.reg_init, P=corr)

        # Stochastic Alternating Optimization
        dim_align_matrix, corr_mat = unsup_align.align(
            self.emb_matrixA, self.emb_matrixB, init_sim, lr=self.lr, bsz=self.bsz, nepoch=self.nepoch, niter=self.niter_align, reg=self.reg_align)

        # Align embedding spaces
        aligned_embed1 = self.emb_matrixA.dot(dim_align_matrix)

        alignment_matrix = kd_align(
            aligned_embed1, self.emb_matrixB, distance_metric=self.embsim, num_top=self.numtop)

        # sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, self.emb_matrixB)

        return alignment_matrix



def align_embeddings(embed1, embed2, CONE_args, adj1=None, adj2=None, struc_embed=None, struc_embed2=None):
    # Step 2: Align Embedding Spaces
    corr = None
    if struc_embed is not None and struc_embed2 is not None:
        if CONE_args['embsim'] == "cosine":
            corr = sklearn.metrics.pairwise.cosine_similarity(embed1, embed2)
        else:
            corr = sklearn.metrics.pairwise.euclidean_distances(embed1, embed2)
            corr = np.exp(-corr)

        # Take only top correspondences
        matches = np.zeros(corr.shape)
        matches[np.arange(corr.shape[0]), np.argmax(corr, axis=1)] = 1
        corr = matches

    # Convex Initialization
    if adj1 is not None and adj2 is not None:
        if not sps.issparse(adj1):
            adj1 = sps.csr_matrix(adj1)
        if not sps.issparse(adj2):
            adj2 = sps.csr_matrix(adj2)
        init_sim, corr_mat = unsup_align.convex_init_sparse(
            embed1, embed2, K_X=adj1, K_Y=adj2, apply_sqrt=False, niter=CONE_args['niter_init'], reg=CONE_args['reg_init'], P=corr)
    else:
        init_sim, corr_mat = unsup_align.convex_init(
            embed1, embed2, apply_sqrt=False, niter=CONE_args['niter_init'], reg=CONE_args['reg_init'], P=corr)
    # print(corr_mat)
    # print(np.max(corr_mat, axis=0))
    # print(np.max(corr_mat, axis=1))

    # Stochastic Alternating Optimization
    dim_align_matrix, corr_mat = unsup_align.align(
        embed1, embed2, init_sim, lr=CONE_args['lr'], bsz=CONE_args['bsz'], nepoch=CONE_args['nepoch'], niter=CONE_args['niter_align'], reg=CONE_args['reg_align'])
    # print(dim_align_matrix.shape, corr_mat.shape)

    # Step 3: Match Nodes with Similar Embeddings
    # Align embedding spaces
    aligned_embed1 = embed1.dot(dim_align_matrix)
    # Greedily match nodes
    # greedily align each embedding to most similar neighbor
    # if CONE_args['alignmethod'] == 'greedy':
    #     # KD tree with only top similarities computed
    #     if CONE_args['numtop'] is not None:
    alignment_matrix = kd_align(
        aligned_embed1, embed2, distance_metric=CONE_args['embsim'], num_top=CONE_args['numtop'])
    # # All pairwise distance computation
    # else:
    #     if CONE_args['embsim'] == "cosine":
    #         alignment_matrix = sklearn.metrics.pairwise.cosine_similarity(
    #             aligned_embed1, embed2)
    #     else:
    #         alignment_matrix = sklearn.metrics.pairwise.euclidean_distances(
    #             aligned_embed1, embed2)
    #         alignment_matrix = np.exp(-alignment_matrix)
    return alignment_matrix, sklearn.metrics.pairwise.euclidean_distances(aligned_embed1, embed2)




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


def main(data, **args):

    Src = data['Src']
    Tar = data['Tar']

    # min_dim = min(Src.shape[0] - 1, Tar.shape[0] - 1)
    # if args['dim'] > min_dim:
    #     args['dim'] = min_dim

    if args['dim'] > Src.shape[0] - 1:
        args['dim'] = Src.shape[0] - 1

    # global args
    # args = parse_args()

    # node_num = int(adj.shape[0] / 2)
    # Tar = adj[:node_num, :node_num]
    # Src = adj[node_num:, node_num:]
    # Tar= ReadFile.edgelist_to_adjmatrix1("data/noise_level_1/arenas_orig.txt")
    # Src = ReadFile.edgelist_to_adjmatrix1("data/noise_level_1/edges_4.txt")
    start = time.time()
   # step1: obtain normalized proximity-preserving node embeddings
    # if (args.embmethod == "netMF"):
    print("0")
    emb_matrixA = embedding.netmf(
        Src, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
    print("A")
    emb_matrixB = embedding.netmf(
        Tar, dim=args['dim'], window=args['window'], b=args['negative'], normalize=True)
    print("B")

    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    alignment_matrix, cost_matrix = align_embeddings(
        emb_matrixA,
        emb_matrixB,
        args,
        adj1=csr_matrix(Src),
        adj2=csr_matrix(Tar),
        struc_embed=None,
        struc_embed2=None
    )
    total_time = time.time() - start
    print(("time for CONE-align (in seconds): %f" % total_time))

    return alignment_matrix, cost_matrix
