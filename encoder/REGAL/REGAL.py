import numpy as np
import sklearn.metrics.pairwise
from utils.encoder_utils import kd_align
from scipy.linalg import block_diag
from .xnetmf_config import *
import encoder.REGAL.xnetmf as xnetmf
from encoder.graph_alignment_model import GraphAlignmentModel


class REGAL(GraphAlignmentModel):
    def __init__(self, adjA, adjB, attributes=None, untillayer=2, buckets=2, alpha=0.01, 
                 k=10, gammastruc=1, gammaattr=1, graph_split_idx=None):

        '''
        attribute: File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.
        untillayer: Calculation until the layer for xNetMF.
        buckets: base of log for degree (node feature) binning.
        k: Controls of landmarks to sample. Default is 10.
        alpha: Discount factor for further layers
        gammastruc: Weight on structural similarity
        gammaattr: Weight on attribute similarity
        '''

        self.adjA = adjA
        self.adjB = adjB
        self.attributes = attributes
        self.untillayer = untillayer
        self.buckets = buckets
        self.alpha = alpha
        self.k = k
        self.gammastruc = gammastruc
        self.gammaattr = gammaattr
        self.graph_split_idx = graph_split_idx
        

    # xnetMF generate graph embedding
    def get_embed(self):

        
        print("Generating xnetMF embeddings for REGAL")
        adj = block_diag(self.adjA, self.adjB)
        graph = Graph(adj, node_attributes = self.attributes)
        max_layer = self.untillayer
        if self.untillayer == 0:
            max_layer = None
        if self.buckets == 1:
            self.buckets = None
        rep_method = RepMethod(max_layer = max_layer, alpha = self.alpha, k = self.k, num_buckets = self.buckets, #BASE OF LOG FOR LOG SCALE
            normalize = True, gammastruc = self.gammastruc, gammaattr = self.gammaattr)
        if max_layer is None:
            max_layer = 1000
        print("Learning representations with max layer %d and alpha = %f" % (max_layer, self.alpha))
        embed = xnetmf.get_representations(graph, rep_method)

        return embed
    
    def get_split_embeddings(self, combined_embed, graph_split_idx = None):
        if graph_split_idx is None:
            graph_split_idx = int(combined_embed.shape[0] / 2)
        dim = combined_embed.shape[1]
        embed1 = combined_embed[:graph_split_idx]
        embed2 = combined_embed[graph_split_idx:]

        return embed1, embed2
    
    def get_embedding_similarities(self, embed, embed2 = None, sim_measure = "euclidean", num_top = None):
        if embed2 is None:
            embed2 = embed

        if num_top is not None: #KD tree with only top similarities computed
            kd_sim = kd_align(embed, embed2, distance_metric = sim_measure, num_top = num_top)
            return kd_sim

        #All pairwise distance computation
        if sim_measure == "cosine":
            similarity_matrix = sklearn.metrics.pairwise.cosine_similarity(embed, embed2)
        else:
            similarity_matrix = sklearn.metrics.pairwise.euclidean_distances(embed, embed2)
            similarity_matrix = np.exp(-similarity_matrix)

        return similarity_matrix
    
    def align(self):

        embed = self.get_embed()
        emb1, emb2 = self.get_split_embeddings(embed, graph_split_idx = self.graph_split_idx)

        alignment_matrix = self.get_embedding_similarities(emb1, emb2, num_top = None)

        return alignment_matrix