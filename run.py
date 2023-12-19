import numpy as np
import argparse
import networkx as nx
import time
import os
import sys
import pickle
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
from encoder.REGAL.xnetmf_config import *
from scipy.linalg import block_diag
import scipy.sparse as sps
import encoder.REGAL.xnetmf as xnetmf
import decoder.RefiNA.refina as refina
import encoder.REGAL.regal_utils as regal_utils
from decoder.RefiNA.RefiNA import RefiNA
# import decoder.RefiNA.refina as refina
import decoder.refina_utils as refina_utils
from encoder.FINAL.FINAL import FINAL
from encoder.CONE.CONE import CONE
from encoder.Grampa.Grampa import Grampa
from encoder.IsoRank.IsoRank import IsoRank
from encoder.BigAlign.BigAlign import BigAlign
from encoder.NSD.NSD import NSD
from encoder.LREA.LREA import LREA
from encoder.Grasp.Grasp import Grasp
import math
from encoder.gwl import gwl_model
import torch.optim as optim
from torch.optim import lr_scheduler
from matcher import matcher,metrics
import scipy

def parse_args():
    parser = argparse.ArgumentParser(description="Run CONE Align.")

    parser.add_argument('--true_align', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_edges-mapping-permutation.txt',
                        help='True alignment file.')
    parser.add_argument('--combined_graph', nargs='?', default='data/synthetic-combined/arenas/arenas950-1/arenas_combined_edges.txt', help='Edgelist of combined input graph.')
    parser.add_argument("--level", default=3, type=int, help='Number of levels for coarseing')
    parser.add_argument('--output_alignment', nargs='?', default='output/alignment_matrix/arenas/arenas950-1', help='Output path for alignment matrix.')
    # Embedding Method
    parser.add_argument('--embmethod', nargs='?', default='netMF', help='Node embedding method.')
    # xnetmf parameters
    parser.add_argument('--attributes', nargs='?', default=None,help='File with saved numpy matrix of node attributes, or int of number of attributes to synthetically generate.  Default is 5 synthetic.')
    parser.add_argument('--attrvals', type=int, default=2,help='Number of attribute values. Only used if synthetic attributes are generated')
    parser.add_argument('--k', type=int, default=10,help='Controls of landmarks to sample. Default is 10.')
    parser.add_argument('--untillayer', type=int, default=2,help='Calculation until the layer for xNetMF.')
    parser.add_argument('--alpha', type=float, default = 0.01, help = "Discount factor for further layers")
    parser.add_argument('--gammastruc', type=float, default = 1, help = "Weight on structural similarity")
    parser.add_argument('--gammaattr', type=float, default = 1, help = "Weight on attribute similarity")
    parser.add_argument('--buckets', default=2, type=float, help="base of log for degree (node feature) binning")
    # REFINA parameters
    parser.add_argument('--n-iter', type=int, default=100, help='Maximum #iter for RefiNA. Default is 20.') 
    parser.add_argument('--token-match', type=float, default = -1, help = "Token match score for each node.  Default of -1 sets it to reciprocal of largest graph #nodes rounded up to smallest power of 10")
    parser.add_argument('--n-update', type=int, default=-1, help='How many possible updates per node. Default is -1, or dense refinement.  Positive value uses sparse refinement')   
    # Alignment methods
    parser.add_argument('--alignmethod', nargs='?', default='REGAL', help='Network alignment method.')
    # Refinement methods
    parser.add_argument('--refinemethod', nargs='?', default=None, help='Network refinement method, to overcome the shortcoming of MILE')

    return parser.parse_args()


def main(args):
    true_align_name = args.true_align
    with open(true_align_name, "rb") as true_alignments_file:
        # for python3, you need to use latin1 as the encoding method
        true_align = pickle.load(true_alignments_file, encoding = "latin1")

    ##################### Load data ######################################
    # running normal graph alignment methods
    combined_graph_name = args.combined_graph
    graph = nx.read_edgelist(combined_graph_name, nodetype=int, comments="%")
    adj = nx.adjacency_matrix(graph, nodelist = range(graph.number_of_nodes()) ).todense().astype(float)
    node_num = int(adj.shape[0] / 2)
    adjA = np.array(adj[:node_num, :node_num])
    split_idx = adjA.shape[0]
    adjB = np.array(adj[node_num:, node_num:])

    # print statistics data
    print("---------------")
    print(f"The number of nodes in a single graph is {node_num}")
    print(f"The number of edges in a the graph A is {nx.from_numpy_matrix(adjA).number_of_edges()}")
    print(f"The number of edges in a the graph B is {nx.from_numpy_matrix(adjB).number_of_edges()}")
    print("---------------")

    ##################### Proprocess if needed ######################################
    if (args.embmethod == "xnetMF"):
        print("Generating xnetMF embeddings for REGAL")
        adj = block_diag(adjA, adjB)
        graph = Graph(adj, node_attributes = args.attributes)
        max_layer = args.untillayer
        if args.untillayer == 0:
            max_layer = None
        if args.buckets == 1:
            args.buckets = None
        rep_method = RepMethod(max_layer = max_layer, alpha = args.alpha, k = args.k, num_buckets = args.buckets, #BASE OF LOG FOR LOG SCALE
            normalize = True, gammastruc = args.gammastruc, gammaattr = args.gammaattr)
        if max_layer is None:
            max_layer = 1000
        print("Learning representations with max layer %d and alpha = %f" % (max_layer, args.alpha))
        embed = xnetmf.get_representations(graph, rep_method)
        # if (args.store_emb):
        #     np.save(args.embeddingA, embed, allow_pickle=False)
        #     np.save(args.embeddingB, embed, allow_pickle=False)
    elif (args.embmethod == "gwl"):
        # parse the data to be gwl readable format
        print("Parse the data to be gwl readable format")
        data_gwl = {}
        data_gwl['src_index'] = {}
        data_gwl['tar_index'] = {}
        data_gwl['src_interactions'] = []
        data_gwl['tar_interactions'] = []
        data_gwl['mutual_interactions'] = []
        for i in range(adjA.shape[0]):
            data_gwl['src_index'][float(i)] = i
        for i in range(adjB.shape[0]):
            data_gwl['tar_index'][float(i)] = i
        ma,mb = adjA.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['src_interactions'].append([ma[i], mb[i]])
        ma,mb = adjB.nonzero()
        for i in range(ma.shape[0]):
            data_gwl['tar_interactions'].append([ma[i], mb[i]])
        after_emb = time.time()
    else:
        print("No preprocessing needed for FINAL")
        after_emb = time.time()

    ##################### Alignment ######################################
    before_align = time.time()
    # step2 and 3: align embedding spaces and match nodes with similar embeddings
    if args.alignmethod == 'REGAL':
        emb1, emb2 = regal_utils.get_embeddings(embed, graph_split_idx=split_idx)
        alignment_matrix = regal_utils.get_embedding_similarities(emb1, emb2, num_top = None)
    elif args.alignmethod == 'FINAL':
        encoder = FINAL(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'IsoRank':
        encoder = IsoRank(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'BigAlign':
        encoder = BigAlign(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'CONE':
        encoder = CONE(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'Grampa':
        encoder = Grampa(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'NSD':
        encoder = NSD(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'LREA':
        encoder = LREA(adjA, adjB)
        alignment_matrix = encoder.align()
    elif args.alignmethod == 'Grasp':
        encoder = Grasp(adjA, adjB)
        alignment_matrix = encoder.align()
        
    elif args.alignmethod == "gwl":
        result_folder = 'gwl_test'
        cost_type = ['cosine']
        method = ['proximal']
        opt_dict = {'epochs': 30,
                    'batch_size': 57000,
                    'use_cuda': False,
                    'strategy': 'soft',
                    'beta': 1e-2,
                    'outer_iteration': 200,
                    'inner_iteration': 1,
                    'sgd_iteration': 500,
                    'prior': False,
                    'prefix': result_folder,
                    'display': False}
        for m in method:
            for c in cost_type:
                hyperpara_dict = {'src_number': len(data_gwl['src_index']),
                                'tar_number': len(data_gwl['tar_index']),
                                'dimension': 256,
                                'loss_type': 'L2',
                                'cost_type': c,
                                'ot_method': m}
                gwd_model = gwl_model.GromovWassersteinLearning(hyperpara_dict)

                # initialize optimizer
                optimizer = optim.Adam(gwd_model.gwl_model.parameters(), lr=1e-2)
                scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

                # Gromov-Wasserstein learning
                gwd_model.train_without_prior(data_gwl, optimizer, opt_dict, scheduler=None)
                # save model
                gwd_model.save_model('{}/model_{}_{}.pt'.format(result_folder, m, c))
                gwd_model.save_recommend('{}/result_{}_{}.pkl'.format(result_folder, m, c))
                alignment_matrix = gwd_model.trans

    ##################### Refine Alignment embeddings ######################################
    if args.refinemethod is not None:
        if args.refinemethod == "RefiNA":
            if sps.issparse(alignment_matrix):
                alignment_matrix = np.array(alignment_matrix.todense())
            if args.n_update > 0:
                alignment_matrix = sps.csr_matrix(alignment_matrix)
                adjA = sps.csr_matrix(adjA)
                adjB = sps.csr_matrix(adjB)
                # alignment_matrix = refina.refina(alignment_matrix, adjA, adjB, true_alignments = true_align) 
                decoder = RefiNA(alignment_matrix, adjA, adjB, n_update=args.n_update,true_alignments = true_align)
                alignment_matrix = decoder.refine_align()  
                print(f"args.refinemethod is {args.refinemethod}")    
    node_num = alignment_matrix.shape[0]
    after_align = time.time()


    if true_align is not None:
        score, _ = refina_utils.score_alignment_matrix(alignment_matrix, topk = 1, true_alignments = true_align)
        mnc = refina_utils.score_MNC(alignment_matrix, adjA, adjB)
        print("Top 1 accuracy: %.5f" % score)
        print("MNC: %.5f" % mnc)

        # pred = matcher.greedy_match(alignment_matrix)
        # print(pred.shape)
        # print(len(true_align))

        # groundtruth_matrix = matcher.load_gt(true_align)
        # metrics.get_statistics(pred, groundtruth_matrix)

        # matcher.sinkhorn_match(alignment_matrix)
        # greedy_match_acc = metrics.get_statistics(pred, groundtruth_matrix)
        # print("Accuracy: %.4f" % greedy_match_acc)


    # evaluation
    # total_time = (after_align - before_align) + (after_emb - before_emb)
    # print(("score for NA: %f" % score))
    # print(("time (in seconds): %f" % total_time))


    # with open(args.output_stats, "w") as log:
    #     log.write("score: %f\n" % score)
    #     log.writelines("time(in seconds): %f\n"% total_time)

if __name__ == "__main__":
    args = parse_args()
    main(args)
