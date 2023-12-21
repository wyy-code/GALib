import numpy as np
import torch
from numpy import inf, nan
from decoder.NAdecoder import NADecoder
import scipy.sparse as sp
from sklearn.preprocessing import normalize

import time
from collections import defaultdict
import math

from decoder.refina_utils import threshold_alignment_matrix, score_alignment_matrix, kd_align, score_MNC
import pdb

class RefiNA(NADecoder):

    """
     Input:
        A1, A2: Input adjacency matrices with n1, n2 nodes
        alignment_matrix: alignment matrix from encoder
        K: number of iterations
        epsilon:  token match score
    
     Output: 
        S: Refinement alignment matrix through MNC

    """

    def __init__(self, alignment_matrix, adj1, adj2, token_match=-1, n_update=-1, iter=50, true_alignments = None):
        self.true_alignments = true_alignments
        self.alignment_matrix = None
        self.adj1 = adj1
        self.adj2 = adj2
        self.token_match = token_match
        self.iter = iter
        self.n_update = n_update
        self.alignment_matrix = alignment_matrix


    def refine_align(self):
        '''Automatically set token match'''
        if self.token_match < 0: #automatically select
            #reciprocal of smallest power of 10 larger than largest graph #nodes
            pow_10 = math.log(max(self.adj1.shape[0], self.adj2.shape[0]), 10)
            self.token_match = 10**-int(math.ceil(pow_10))	

        #alignment_matrix = threshold_alignment_matrix(alignment_matrix, topk = args.init_threshold)
        for i in range(self.iter):
            '''DIAGNOSTIC/DEMO ONLY: keep track of alignment quality'''
            if self.alignment_matrix.shape[0] < 20000: #don't bother with intermediate diagnostics for big matrices
                print(("Scores after %d refinement iterations" % i))
                if self.true_alignments is not None:
                    score, _ = score_alignment_matrix(self.alignment_matrix, true_alignments = self.true_alignments)
                    print("Top 1 accuracy: %.5f" % score)
                mnc = score_MNC(self.alignment_matrix, self.adj1, self.adj2)
                print("MNC: %.5f" % mnc)

            '''Step 1: compute MNC-based update'''
            '''n_update: How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement'''

            update = self.compute_update(self.adj1, self.adj2, self.alignment_matrix, self.n_update)
            update = self.compute_update(self.adj1, self.adj2, self.alignment_matrix, self.n_update)#min( int(5*(i+1)), adj1.shape[0]) )
            
            '''Step 2: apply update and token match'''
            if self.n_update > 0: #add token match score here so we can selectively update
                if sp.issparse(self.alignment_matrix):
                    nonzero_updates = update.nonzero() #Indices of alignments to update
                    updated_data = np.asarray(self.alignment_matrix[nonzero_updates]) #Alignment values we want to update
                    updated_data += self.token_match #Add token match
                    updated_data *= update.data #Multiplicatively update them

                    self.alignment_matrix = self.alignment_matrix.tolil()
                    self.alignment_matrix[nonzero_updates] = updated_data
                    self.alignment_matrix.tocsr()
                else:
                    self.alignment_matrix[update != 0] += self.token_match
                    self.alignment_matrix[update != 0] *= update[update != 0]
            else:
                self.alignment_matrix = self.alignment_matrix * update
                self.alignment_matrix += self.token_match

            '''Step 3: normalize'''
            self.alignment_matrix = self.normalize_alignment_matrix(self.alignment_matrix)

        return self.alignment_matrix

    def compute_update(self, adj1, adj2, alignment_matrix, n_update):
        update_matrix = adj1.dot(alignment_matrix).dot(adj2.T) #row i: counterparts of neighbors of i

        if n_update > 0 and n_update < adj1.shape[0]:
            if sp.issparse(update_matrix): 
                if update_matrix.shape[0] < 120000: update_matrix = update_matrix.toarray() #still fits in memory and dense is faster 
                update_matrix = threshold_alignment_matrix(update_matrix, topk = n_update, keep_dist = True)
                update_matrix = sp.csr_matrix(update_matrix)
            else:
                update_matrix = threshold_alignment_matrix(update_matrix, topk = n_update, keep_dist = True)
        return update_matrix

    def normalize_alignment_matrix(self, alignment_matrix):
        alignment_matrix = normalize(alignment_matrix, norm = "l1", axis = 1)
        alignment_matrix = normalize(alignment_matrix, norm = "l1", axis = 0)
        return alignment_matrix

